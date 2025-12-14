import os
from dotenv import load_dotenv
import json
import base64
import audioop
from flask import Flask, request
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

ngrok_url = "..." # asta o luati din terminal dupa comanda: ngrok http 5000 -> ex: https://catarina-tensed-berry.ngrok-free.dev

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

active_agent = None # aici ar trebui sa vina instanta de agent care face stt-llm-tts si foloseste tool urile

@app.route('/dial')
def dial_hotel():
    call = client.calls.create(
        url=f"{ngrok_url}/voice",
        to="...", # adaugati nr vostru aici
        from_="+15703545228",) # aici e numarul primit de la twilio
    return call.sid

"""
    Pt TWILIO: 
    -> aici adaugati numarul vostru : https://console.twilio.com/us1/develop/phone-numbers/manage/verified
    -> aici setati acces pt Romania : https://console.twilio.com/us1/develop/voice/settings/geo-permissions?frameUrl=%2Fconsole%2Fvoice%2Fcalls%2Fgeo-permissions%3Fx-target-region%3Dus1&currentFrameUrl=%2Fconsole%2Fvoice%2Fcalls%2Fgeo-permissions%2Flow-risk%3F__override_layout__%3Dembed%26countryIsoCode%3Droman%26x-target-region%3Dus1%26bifrost%3Dtrue

    -> de aici va cumparati nr de twilio (e gratis, dar cu limita ) : https://console.twilio.com/us1/develop/phone-numbers/manage/incoming

    Pt NGROK:
    -> va faceti cont gratis
    -> la Setup&Installation o sa va apara cv de genutl ngrok config add-authtoken ... pe care o rulati in terminal si apoi ngrok http 5000 pt url
"""

@app.route("/voice", methods=['POST'])
def voice():
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f'wss://{request.host}/stream')
    response.append(connect)
    return str(response), 200, {'Content-Type': 'application/xml'}

# websocket handler for real-time audio
@sock.route('/stream')
def stream(ws):
    print("Connected with Twilio")

    while True:
        try:
            message = ws.receive()
            if message is None: break
            
            data = json.loads(message)

            if data['event'] == 'start':
                stream_sid = data['start']['streamSid']
                print(f"Stream started: {stream_sid}")

            elif data['event'] == 'media':
                # decode audio and pass it to agent
                if active_agent:
                    payload = data['media']['payload']
                    chunk_mulaw = base64.b64decode(payload)
                    chunk_pcm = audioop.ulaw2lin(chunk_mulaw, 2)
                    response_pcm = active_agent.handle_audio_stream(chunk_pcm)
                    
                    # if agent responded, send it back to twilio in the needed format
                    if response_pcm:
                        mulaw = audioop.lin2ulaw(response_pcm, 2)
                        b64_audio = base64.b64encode(mulaw).decode("utf-8")
                        
                        msg = {
                            "event": "media",
                            "streamSid": stream_sid,
                            "media": {
                                "payload": b64_audio
                            }
                        }
                        ws.send(json.dumps(msg))


        except Exception as e:
            print(f"Telephony Error: {e}")
            break

if __name__ == '__main__':
    app.run(port=5000)

