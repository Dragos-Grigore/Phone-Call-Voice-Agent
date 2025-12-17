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
from agent_tools import update_data

# Optimized entity extraction
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()

app = Flask(__name__)
sock = Sock(app)

lighting_url = "https://5000-01kck4zt7we0gagz901x4r8gzs.cloudspaces.litng.ai"

account_sid = os.environ["TWILIO_ACCOUNT_SID"]
auth_token = os.environ["TWILIO_AUTH_TOKEN"]
client = Client(account_sid, auth_token)

called_hotel = "Hotel_1"
active_agent = VoiceAgent(called_hotel)

# --- Load Model Once ---
print("Loading Model (Flan-T5)...")
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

hf_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=64,
    do_sample=False,
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer ONLY using the context. "
        "If the answer is not in the context, say 'Not found'.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


qa_runnable = prompt | llm

def extract_entities_from_dialog(text: str) -> dict:
    """
    Extracts hotel_name, address, email, phone from text.
    """
    entities = {}
    queries = {
        "hotel_name": "What is the name of the hotel?",
        "adress": "What is the address of the hotel?",
        "email_address": "What is the email address?",
        "phone": "What is the phone number?"
    }

    for field, question in queries.items():
        try:
            res = qa_runnable.invoke({"context": text, "question": question})
            clean = res.strip()
            if "not found" not in clean.lower() and len(clean) > 2:
                entities[field] = clean
        except Exception as e:
            print(f"Error ({field}): {e}")

    return entities

# --- Flask / Twilio endpoints ---
@app.route('/dial')
def dial_hotel() -> str:
    call = client.calls.create(
        url=f"{lighting_url}/voice",
        to="+40775380182",
        from_="+14058177187",  # Twilio number
    )
    return str(call.sid)


@app.route("/voice", methods=['POST'])
def voice():
    response = VoiceResponse()
    connect = Connect()
    connect.stream(url=f'wss://{request.host}/stream')
    response.append(connect)
    return str(response), 200, {'Content-Type': 'application/xml'}


@sock.route('/stream')
def stream(ws):
    print("Connected with Twilio")

    while True:
        try:
            message = ws.receive()
            if message is None:
                break
            data = json.loads(message)

            if data['event'] == 'media':
                payload = data['media']['payload']
                chunk_mulaw = base64.b64decode(payload)

                # Twilio sends mulaw 8000Hz. Convert to PCM 8000Hz
                chunk_pcm = audioop.ulaw2lin(chunk_mulaw, 2)

                # Feed to Interactor
                response_pcm, final_transcript = active_agent.handle_audio_stream(chunk_pcm)

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

    print("Final transcript:")
    print(final_transcript)
    # Single LLM call for structured extraction
    answers = extract_entities_from_dialog(final_transcript)
    print(answers)
    update_data('Hotel_1', answers)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

