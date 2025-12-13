# Phone-Call-Voice-Agent

Voice-driven assistant for hotel calls. It listens to the microphone, detects speech, transcribes with Whisper, asks/collects hotel details (name, address, email) using a local Hugging Face model, and speaks responses with neural TTS (Coqui) or a local fallback (pyttsx3).

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
# Install a torch build that matches your hardware (GPU strongly recommended).
pip install torch --index-url https://download.pytorch.org/whl/cu121  # adjust/cuXXX or cpu
pip install -r requirements.txt
```

Run the voice loop:
```bash
python interactor.py
```

### What happens
- Captures 16 kHz mono audio from the mic, segments speech with WebRTC VAD.
- Transcribes each chunk with Whisper (`AppConfig.whisper_model_name`, default `small`).
- Builds concise prompts for `microsoft/Phi-3-mini-4k-instruct` to ask/confirm hotel name, address, email.
- Speaks replies with Coqui TTS if available, otherwise pyttsx3.
- Prints extracted entities for debugging via `ner_tool.extract_entities_from_dialog`.

### Tuning
- Adjust defaults in `AppConfig` (models, VAD aggressiveness, timeouts, generation settings).
- Swap the Hugging Face model ID if you prefer another instruct model.
- If TTS audio is missing, ensure PortAudio devices work or fall back to pyttsx3 by uninstalling `TTS`.

### Firebase + dataset (optional)
`fill_dataset.py` reads the provided Excel and writes hotel docs to Firestore. Set `FIREBASE_CREDENTIALS_PATH` in `.env` to point to your service account JSON before running it.
