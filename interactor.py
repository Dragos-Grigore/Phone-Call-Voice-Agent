import torch
import numpy as np
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer
import pyttsx3
import scipy.signal
import audioop
import sys
import re
import wave, os
from agent_tools import get_hotel_info, verify_with_hotel, update_data
from agent import extract_entities_from_dialog

class VoiceAgent:
    def __init__(self, hotel_id):
        self.hotel_id = hotel_id
        # stt with whisper
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.stt_model = whisper.load_model("base", device=self.device)

        # llm with tiny llama
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float32, 
            device_map="cpu", 
            low_cpu_mem_usage=True
        )

        # tts with coqui
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[1].id)

        # Audio Buffer Logic
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.is_speaking = False
        self.SILENCE_THRESHOLD = 20
        self.ENERGY_THRESHOLD = 300

        self.call_ended = False

        self.tools = [
            get_hotel_info, 
            verify_with_hotel, 
            update_data,
            extract_entities_from_dialog
        ]

    def _resample_audio(self, audio_data, orig_sr, target_sr):
        number_of_samples = round(len(audio_data) * float(target_sr) / orig_sr)
        return scipy.signal.resample(audio_data, number_of_samples)

    def process_llm(self, user_text):
        current_db = get_hotel_info(self.hotel_id)
        
        # 2. Extract Entities (Using Flan-T5 from agent.py)
        extracted = extract_entities_from_dialog(user_text)
        
        system_note = ""
        
        if extracted:
            print(f"Tool extracted: {extracted}")
            
            # 3. Verify
            verification = verify_with_hotel(current_db, extracted)
            
            # 4. Update
            if verification["needs_update"]:
                updates = verification["updates"]
                update_data(self.hotel_id, updates)
                current_db.update(updates)
                system_note = f"[SYSTEM: The database successfully updated with: {updates}. Confirm this to the user.]"
                print(f"Database Updated: {updates}")
           
        info_sentences = [f"The {k} is {v}." for k, v in current_db.items()]
        db_natural_text = " ".join(info_sentences)

        conversation = (
            f"<|system|>\n"
            f"You are Sofia, a polite hotel verification assistant.\n"
            f"Current Hotel Info: {db_natural_text}\n"
            f"{system_note}\n"
            f"IMPORTANT RULES:\n"
            f"1. Respond ONLY with natural spoken English\n"
            f"2. Never generate code, symbols, or scripts\n"
            f"3. Never output backticks, brackets, or special formatting\n"
            f"4. Keep responses under 3 sentences\n"
            f"5. Be conversational and warm\n"
            f"Goal: Verify hotel details naturally with the user.</s>\n"
            f"<|user|>\n{user_text}</s>\n"
            f"<|assistant|>\n"
        )

        # generate response
        input_ids = self.tokenizer(conversation, return_tensors="pt").input_ids
        outputs = self.llm_model.generate(
            input_ids, 
            max_new_tokens=80,
            do_sample=True, 
            temperature=0.5,
            repetition_penalty=1.2
        )
        response_text = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        response_text = response_text.replace("<|assistant|>", "").strip()

        
        # end call
        if "goodbye" in response_text.lower() or "have a great day" in response_text.lower():
            self.call_ended = True
            
        return response_text

    def text_to_speech_pcm(self, text):
        filename = "temp_response.wav"
        self.engine.save_to_file(text, filename)
        self.engine.runAndWait()
        
        
        with wave.open(filename, 'rb') as wav:
            sr = wav.getframerate()
            raw = wav.readframes(wav.getnframes())
            np_wav = np.frombuffer(raw, dtype=np.int16)
            
            if sr != 8000:
                np_8k = self._resample_audio(np_wav, sr, 8000).astype(np.int16)
                data = np_8k.tobytes()
            else:
                data = raw
                
        os.remove(filename)
        return data

    def handle_audio_stream(self, chunk_pcm):
        if self.call_ended:
            return None

        # convert the audio in a good format
        audio_data = np.frombuffer(chunk_pcm, dtype=np.int16)
        if len(audio_data) == 0: return None
        audio_float = audio_data.astype(np.float32)
        energy = np.sqrt(np.mean(audio_float**2))

        # detecting if the hotel has stopped talking or is still speaking
        if energy > self.ENERGY_THRESHOLD:
            if not self.is_speaking:
                self.is_speaking = True
            self.silence_frames = 0
            self.audio_buffer.extend(chunk_pcm)
        else:
            if self.is_speaking:
                self.silence_frames += 1
                self.audio_buffer.extend(chunk_pcm)

        # agent will start talking after a moment
        if self.is_speaking and self.silence_frames > self.SILENCE_THRESHOLD:
            full_audio = np.frombuffer(self.audio_buffer, dtype=np.int16)
            self.audio_buffer = bytearray()
            self.is_speaking = False
            self.silence_frames = 0
            
            # getting the hotel response ready for the agent
            audio_f = full_audio.astype(np.float32) / 32768.0
            audio_16k = self._resample_audio(audio_f, 8000, 16000)
            result = self.stt_model.transcribe(audio_16k, fp16=False)
            transcript = result['text'].strip()
            print(f"Hotel response: {transcript}")

            if not transcript: return None

            # getting the agent's response
            response = self.process_llm(transcript)
            print(f"Agent response: {response}")
            return self.text_to_speech_pcm(response)

        return None