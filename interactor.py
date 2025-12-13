from __future__ import annotations

import collections
import queue
import re
import threading
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pyttsx3
import sounddevice as sd
import torch
import webrtcvad
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from ner_tool import extract_entities_from_dialog

try:
    from TTS.api import TTS as CoquiTTS
except Exception:
    CoquiTTS = None


@dataclass
class AppConfig:
    """Runtime configuration for the voice agent."""

    # Audio + VAD
    sample_rate: int = 16000
    frame_ms: int = 30
    energy_threshold: float = 0.001
    vad_aggressiveness: int = 2  # 0–3
    pre_speech_ms: int = 300
    trailing_silence_ms: int = 600
    min_speech_seconds: float = 0.7
    max_speech_seconds: float = 12.0
    post_speech_cooldown_ms: int = 800

    # Models
    whisper_model_name: str = "small"
    hf_model_id: str = "microsoft/Phi-3-mini-4k-instruct"
    system_prompt: str = (
        "You are a friendly hotel reservations agent on a phone call. "
        "Ask concise questions to collect hotel_name, address, and email. "
        "Confirm what you have and ask only for missing details. Keep answers short."
    )
    max_history_turns: int = 6
    max_new_tokens: int = 256
    temperature: float = 0.6
    top_p: float = 0.9

    # TTS
    coqui_model_id: str = "tts_models/en/vctk/vits"

    def frames_from_ms(self, milliseconds: float) -> int:
        return max(1, int(np.ceil(milliseconds / self.frame_ms)))

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * self.frame_ms / 1000)


def pick_voice(engine: pyttsx3.Engine) -> None:
    voices = engine.getProperty("voices")
    preferred_voice = None
    for voice in voices:
        name = voice.name.lower()
        languages = ",".join(
            lang.decode("utf-8", "ignore") if isinstance(lang, bytes) else str(lang)
            for lang in (voice.languages or [])
        ).lower()
        if "english" in name or "en" in languages or "en_" in voice.id.lower():
            preferred_voice = voice.id
            break
    if preferred_voice is None and voices:
        preferred_voice = voices[0].id
    if preferred_voice:
        engine.setProperty("voice", preferred_voice)


def init_tts_engine(rate: int = 170, volume: float = 0.9) -> pyttsx3.Engine:
    """Configure pyttsx3 with a clear English voice and a comfortable speaking rate."""
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.setProperty("volume", volume)
    pick_voice(engine)
    return engine


def clean_tts_text(text: str) -> str:
    """Strip emojis/control chars Coqui can't handle; keep readable ASCII/punctuation."""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"[^\w\s.,!?;:'\"-]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class SpeechSynthesizer:
    """Simple wrapper that prefers neural TTS (Coqui) and falls back to pyttsx3."""

    def __init__(self, speaking_event: threading.Event, config: AppConfig):
        self.mode = "pyttsx3"
        self.pyttsx = init_tts_engine()
        self.coqui = None
        self.coqui_rate = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.default_speaker = None
        self.speaking_event = speaking_event
        self.config = config

        if CoquiTTS is not None:
            try:
                self.coqui = CoquiTTS(
                    config.coqui_model_id,
                    progress_bar=False,
                )
                try:
                    self.coqui.to(self.device)
                except Exception:
                    # Older TTS versions may not support .to(); ignore.
                    pass
                self.coqui_rate = getattr(
                    getattr(self.coqui, "synthesizer", None),
                    "output_sample_rate",
                    config.sample_rate,
                )
                try:
                    speakers: List[str] = []
                    if hasattr(self.coqui, "speaker_manager") and getattr(
                        self.coqui.speaker_manager, "speakers", None
                    ):
                        speakers = list(self.coqui.speaker_manager.speakers)
                    elif hasattr(self.coqui, "speakers"):
                        speakers = list(self.coqui.speakers)
                    if speakers:
                        self.default_speaker = speakers[0]
                        print("Using Coqui speaker:", self.default_speaker)
                except Exception:
                    pass
                self.mode = "coqui"
                print(f"Using Coqui TTS model '{config.coqui_model_id}' on {self.device}.")
            except Exception as exc:
                print(f"Coqui TTS unavailable ({exc}); falling back to pyttsx3.")

    def speak(self, text: str):
        clean_text = clean_tts_text(text)
        if not clean_text:
            print("Skipping TTS: empty/unsupported text after cleaning.")
            return

        if self.mode == "coqui" and self.coqui is not None:
            tts_kwargs = {}
            if self.default_speaker is not None:
                tts_kwargs["speaker"] = self.default_speaker
            audio = np.asarray(self.coqui.tts(clean_text, **tts_kwargs), dtype=np.float32)
            self.speaking_event.set()
            try:
                sd.play(audio, self.coqui_rate)
                sd.wait()
            finally:
                self.speaking_event.clear()
        else:
            self.speaking_event.set()
            try:
                self.pyttsx.say(text)
                self.pyttsx.runAndWait()
            finally:
                self.speaking_event.clear()


class HotelLLMResponder:
    """Lightweight LangChain-free responder using a local HF model."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.hf_model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.hf_model_id,
            device_map="auto",
            torch_dtype="auto",
        )
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.history: List[Tuple[str, str]] = []

    def _messages(self, user_text: str) -> List[dict]:
        messages: List[dict] = [{"role": "system", "content": self.config.system_prompt}]
        for role, content in self.history[-self.config.max_history_turns :]:
            messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _dialog_text(self) -> str:
        lines = []
        for role, content in self.history:
            prefix = "User" if role == "user" else "Agent"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def respond(self, user_text: str) -> Optional[str]:
        self.history.append(("user", user_text))
        messages = self._messages(user_text)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        try:
            result = self.generator(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]["generated_text"]
        except Exception as exc:
            print(f"Generation failed: {exc}")
            return None

        reply = result[len(prompt) :].strip()
        self.history.append(("assistant", reply))

        entities = extract_entities_from_dialog(self._dialog_text())
        print("[entities]", entities)
        return reply or None


class ConversationApp:
    """
    Coordinates microphone capture, VAD-based segmentation, Whisper STT,
    hotel-domain LLM responses, and TTS playback.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        self.audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self.speech_q: queue.Queue[Optional[str]] = queue.Queue()
        self.speaking = threading.Event()
        self.resume_listening_at = 0.0
        self.vad = webrtcvad.Vad(self.config.vad_aggressiveness)

        self.whisper_model = whisper.load_model(self.config.whisper_model_name)
        self.tts = SpeechSynthesizer(self.speaking, self.config)
        self.llm = HotelLLMResponder(self.config)

    def audio_callback(self, indata, frames, time_info, status):
        self.audio_q.put(indata[:, 0].copy())

    def _tts_worker(self):
        while True:
            text = self.speech_q.get()
            if text is None:
                break
            self.tts.speak(text)
            self.resume_listening_at = time.time() + (
                self.config.post_speech_cooldown_ms / 1000.0
            )

    def _drain_audio_queue(self):
        while True:
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def _reset_buffers(
        self,
        buffer: np.ndarray,
        pre_speech_frames: collections.deque[np.ndarray],
        speech_frames: List[np.ndarray],
    ) -> np.ndarray:
        self._drain_audio_queue()
        buffer = np.zeros(0, dtype=np.float32)
        pre_speech_frames.clear()
        speech_frames.clear()
        return buffer

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        result = self.whisper_model.transcribe(
            audio,
            fp16=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
        )
        text = result.get("text", "").strip()
        return text or None

    def _respond(self, text: str) -> Optional[str]:
        return self.llm.respond(text)

    def _process_segment(self, frames: Iterable[np.ndarray]) -> None:
        audio = np.concatenate(list(frames))
        text = self._transcribe(audio)
        if not text:
            return
        print("\nUser:", text)

        reply = self._respond(text)
        if not reply:
            return
        print("Agent:", reply)
        self.speech_q.put(reply)

    def run(self):
        tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        tts_thread.start()

        print("Listening... Speak naturally (Ctrl+C to stop)")

        buffer = np.zeros(0, dtype=np.float32)
        pre_speech_frames = collections.deque(
            maxlen=self.config.frames_from_ms(self.config.pre_speech_ms)
        )
        speech_frames: List[np.ndarray] = []
        silence_frames = 0
        min_speech_frames = self.config.frames_from_ms(self.config.min_speech_seconds * 1000)
        trailing_silence_frames = self.config.frames_from_ms(self.config.trailing_silence_ms)
        max_speech_frames = self.config.frames_from_ms(self.config.max_speech_seconds * 1000)
        in_speech = False

        with sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.config.frame_size,
            callback=self.audio_callback,
        ):
            try:
                while True:
                    if self.speaking.is_set() or time.time() < self.resume_listening_at:
                        buffer = self._reset_buffers(buffer, pre_speech_frames, speech_frames)
                        silence_frames = 0
                        in_speech = False
                        time.sleep(0.05)
                        continue

                    while not self.audio_q.empty():
                        buffer = np.concatenate((buffer, self.audio_q.get()))

                    processed_segment = False

                    while len(buffer) >= self.config.frame_size:
                        frame = buffer[: self.config.frame_size]
                        buffer = buffer[self.config.frame_size :]
                        pre_speech_frames.append(frame)

                        frame_energy = np.mean(np.abs(frame))
                        voiced = False
                        if frame_energy >= self.config.energy_threshold:
                            pcm = (frame * 32768).astype(np.int16).tobytes()
                            voiced = self.vad.is_speech(pcm, self.config.sample_rate)

                        if not in_speech and voiced:
                            in_speech = True
                            speech_frames = list(pre_speech_frames)
                            silence_frames = 0
                        elif in_speech:
                            speech_frames.append(frame)
                            silence_frames = 0 if voiced else silence_frames + 1

                        should_finalize = False
                        if in_speech:
                            if silence_frames >= trailing_silence_frames:
                                should_finalize = True
                            elif len(speech_frames) >= max_speech_frames:
                                should_finalize = True

                        if should_finalize:
                            in_speech = False
                            pre_speech_frames.clear()
                            if len(speech_frames) >= min_speech_frames:
                                self._process_segment(speech_frames)
                                processed_segment = True

                            speech_frames = []
                            silence_frames = 0

                        if processed_segment:
                            break

                    time.sleep(0.01)

            except KeyboardInterrupt:
                print("\nStopped.")
            finally:
                self.speech_q.put(None)
                tts_thread.join(timeout=2)


def main():
    ConversationApp().run()


if __name__ == "__main__":
    main()
