import torch
import numpy as np
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import scipy.signal
import sys
import wave, os
import subprocess
from typing import Union, List, Any, cast

from agent_tools import get_hotel_info, verify_with_hotel, update_data
from agent import extract_entities_from_dialog

class VoiceAgent:
    def __init__(self, hotel_id):
        self.hotel_id = hotel_id
        
        # 1. STT (Whisper)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Whisper on {self.device}...")
        self.stt_model = whisper.load_model("base", device=self.device)

        # 2. VAD (Silero)
        print("Loading Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
            trust_repo=True
        ) # type: ignore
        self.vad_model.to(self.device)

        # 3. LLM (Qwen2-1.5B-Instruct)
        model_id = "microsoft/Phi-4-mini-instruct"
        print(f"Loading LLM: {model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            device_map=self.device, 
            trust_remote_code=False,
            low_cpu_mem_usage=True
        )

        # --- AUDIO CONFIGURATION ---
        self.SAMPLE_RATE = 8000 
        
        # Main Buffer (Accumulates speech for Whisper)
        self.audio_buffer = bytearray()
        
        # VAD Buffer (Accumulates tiny chunks for VAD check)
        self.vad_buffer = bytearray()
        
        # --- FIX: WINDOW SIZE MUST BE 256 FOR 8000HZ ---
        self.VAD_WINDOW = 256 
        self.VAD_WINDOW_BYTES = self.VAD_WINDOW * 2 # 16-bit PCM = 2 bytes per sample
        
        # State
        self.silence_chunks = 0
        self.is_speaking = False
        self.SPEECH_THRESHOLD = 0.5
        self.MAX_SILENCE_CHUNKS = 20 
        self.final_transcript=""
        self.call_ended = False

    def _resample_audio(self, audio_data, orig_sr, target_sr) -> np.ndarray:
        number_of_samples = round(len(audio_data) * float(target_sr) / orig_sr)
        resampled = scipy.signal.resample(audio_data, number_of_samples)
        if isinstance(resampled, tuple):
            return np.array(resampled[0])
        return np.array(resampled)

    def process_llm(self, user_text):
        current_db = get_hotel_info(self.hotel_id)
        extracted = extract_entities_from_dialog(user_text)
        system_note = ""
        
        if extracted:
            print(f"Tool extracted: {extracted}")
            verification = verify_with_hotel(current_db, extracted)
            if verification["needs_update"]:
                updates = verification["updates"]
                update_data(self.hotel_id, updates)
                current_db.update(updates)
                system_note = f"(SYSTEM NOTE: You updated the database with: {updates}.)"
                print(f"Database Updated: {updates}")
           
        info_sentences = [f"{k}: {v}" for k, v in current_db.items()]
        db_context = "\n".join(info_sentences)

        messages = [
            {"role": "system", "content": f"""You are Sofia, a professional voice agent. You need to ask the hotel manager if the phone number, address and email address are correct
            CURRENT DATA:
            {db_context}
            {system_note}
            
            YOUR JOB:
            1. Verify guest details (Email, Phone).
            2. Be concise (under 2 sentences).
            3. NEVER ask user to type.
            """},
            {"role": "user", "content": "Hello. <Hotel name>, how can i help you?"},
            {"role": "assistant", "content": "Hello! I'm Sofia, calling to verify your details."},
            {"role": "user", "content": user_text},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.llm_model.generate(
            **inputs, 
            max_new_tokens=80,
            do_sample=True, 
            temperature=0.7,
        )
        
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response_text = response_text.strip()

        if "goodbye" in response_text.lower() or "bye" in response_text.lower():
            self.call_ended = True
            
        return response_text

    def text_to_speech_pcm(self, text):
        filename = "temp_response.wav"
        try:
            subprocess.run(["espeak", "-w", filename, "-v", "en-us", "-s", "150", text], check=True)
        except Exception as e:
            print(f"TTS Error: {e}")
            return None

        data = None
        if os.path.exists(filename):
            with wave.open(filename, 'rb') as wav:
                sr = wav.getframerate()
                raw = wav.readframes(wav.getnframes())
                np_wav = np.frombuffer(raw, dtype=np.int16)
                if sr != 8000:
                    resampled = self._resample_audio(np_wav, sr, 8000)
                    data = resampled.astype(np.int16).tobytes()
                else:
                    data = raw
            os.remove(filename)
        return data

    def handle_audio_stream(self, chunk_pcm):
        if self.call_ended: return None

        # 1. ADD INCOMING DATA TO VAD BUFFER
        self.vad_buffer.extend(chunk_pcm)
        response_audio = None

        # 2. PROCESS ONLY COMPLETE CHUNKS (256 samples / 512 bytes)
        while len(self.vad_buffer) >= self.VAD_WINDOW_BYTES:
            
            # Pop strictly 256 samples
            vad_chunk_bytes = self.vad_buffer[:self.VAD_WINDOW_BYTES]
            self.vad_buffer = self.vad_buffer[self.VAD_WINDOW_BYTES:]
            
            audio_int16 = np.frombuffer(vad_chunk_bytes, dtype=np.int16)
            
            # Normalize for VAD
            audio_float = audio_int16.astype(np.float32) / 32768.0
            audio_tensor = torch.from_numpy(audio_float).to(self.device)
            
            # Check speech
            speech_prob = float(self.vad_model(audio_tensor, self.SAMPLE_RATE).item())
            
            if speech_prob > self.SPEECH_THRESHOLD:
                if not self.is_speaking:
                    print(f"Speech started (Prob: {speech_prob:.2f})")
                    self.is_speaking = True
                self.silence_chunks = 0
                self.audio_buffer.extend(vad_chunk_bytes)
            else:
                if self.is_speaking:
                    self.silence_chunks += 1
                    self.audio_buffer.extend(vad_chunk_bytes)

            # 3. TRIGGER RESPONSE IF SILENCE PERSISTS
            if self.is_speaking and self.silence_chunks > self.MAX_SILENCE_CHUNKS:
                print("Processing speech...")
                full_audio = np.frombuffer(self.audio_buffer, dtype=np.int16)
                
                # Reset
                self.audio_buffer = bytearray()
                self.is_speaking = False
                self.silence_chunks = 0
                
                # Transcribe (8k -> 16k)
                audio_f = full_audio.astype(np.float32) / 32768.0
                audio_16k = self._resample_audio(audio_f, self.SAMPLE_RATE, 16000)
                
                try:
                    result = self.stt_model.transcribe(audio_16k, fp16=False)
                    raw_text = result['text']
                    transcript = str(raw_text[0] if isinstance(raw_text, list) else raw_text).strip()
                    self.final_transcript += transcript + " "
                    print(f"User said: {transcript}")
                    
                    if transcript and len(transcript) >= 2:
                        llm_reply = self.process_llm(transcript)
                        self.final_transcript += llm_reply + " "
                        print(f"Agent replying: {llm_reply}")
                        response_audio = self.text_to_speech_pcm(llm_reply)
                        
                        # Stop processing remaining buffer to avoid double-replying
                        break 
                    
                except Exception as e:
                    print(f"Processing Error: {e}")

        return response_audio, self.final_transcript