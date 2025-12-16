import os
from dotenv import load_dotenv
import json
import base64
import audioop
from flask import Flask, request
from flask_sock import Sock
from twilio.twiml.voice_response import VoiceResponse, Connect
from twilio.rest import Client
from interactor import VoiceAgent

load_dotenv()

app = Flask(__name__)
sock = Sock(app)


lighting_url = "https://5000-01kck4zt7we0gagz901x4r8gzs.cloudspaces.litng.ai" 

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

called_hotel = "Hotel_1"
active_agent = VoiceAgent(called_hotel) 


@app.route('/dial')
def dial_hotel() -> str:
    call = client.calls.create(
        url=f"{lighting_url}/voice",
        to="+40764067966",
        from_="+15703545228", # Twilio number
    )
    
    return str(call.sid)


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

            if data['event'] == 'media':
                payload = data['media']['payload']
                chunk_mulaw = base64.b64decode(payload)
                
                # Twilio sends mulaw 8000Hz. Convert to PCM 8000Hz
                chunk_pcm = audioop.ulaw2lin(chunk_mulaw, 2)
                
                # Feed to Interactor
                response_pcm = active_agent.handle_audio_stream(chunk_pcm)
                
                if response_pcm:
                    # Agent replied! Convert PCM 8000Hz back to mulaw
                    mulaw = audioop.lin2ulaw(response_pcm, 2)
                    b64_audio = base64.b64encode(mulaw).decode("utf-8")
                    
                    ws.send(json.dumps({
                        "event": "media",
                        "streamSid": data['streamSid'],
                        "media": {"payload": b64_audio}
                    }))
        

        except Exception as e:
            print(f"Stream Error: {e}")
            break
    



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)