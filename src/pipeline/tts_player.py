"""
English text-to-speech player for navigation output.

This module uses the local system voice when available. It is designed to
fail gracefully so that lack of TTS dependencies never breaks the pipeline.
"""
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    import pyttsx3
except Exception:  # pragma: no cover
    pyttsx3 = None


class TTSPlayer:
    """Play English navigation text through a local system TTS engine."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", True))
        self.backend = self.config.get("backend", "system")
        self.language = self.config.get("language", "en")
        self.voice_hint = str(self.config.get("voice_hint", "English"))
        self.rate = self.config.get("rate")
        self.volume = self.config.get("volume")
        self.engine = None

        if self.enabled:
            self._init_engine()

    def _init_engine(self):
        if pyttsx3 is None:
            logger.warning("pyttsx3 is not installed; TTS playback will be skipped.")
            return

        try:
            self.engine = pyttsx3.init()
            self._configure_voice()
            if self.rate is not None:
                self.engine.setProperty("rate", self.rate)
            if self.volume is not None:
                self.engine.setProperty("volume", self.volume)
        except Exception as e:
            logger.warning("Failed to initialize TTS engine: %s", e)
            self.engine = None

    def _configure_voice(self):
        if self.engine is None:
            return

        try:
            voices = self.engine.getProperty("voices")
        except Exception as e:
            logger.warning("Unable to query system voices: %s", e)
            return

        best_voice_id = None
        fallback_voice_id = None
        english_keywords = ["en-us", "en_gb", "english", "en-uk", "en_us", "en gb", "en us"]

        for voice in voices:
            raw = " ".join(
                str(part)
                for part in [
                    getattr(voice, "id", ""),
                    getattr(voice, "name", ""),
                    getattr(voice, "languages", ""),
                ]
            ).lower()

            if self.voice_hint.lower() in raw and fallback_voice_id is None:
                fallback_voice_id = voice.id

            if any(keyword in raw for keyword in english_keywords):
                best_voice_id = voice.id
                break

        selected_voice_id = best_voice_id or fallback_voice_id
        if selected_voice_id:
            try:
                self.engine.setProperty("voice", selected_voice_id)
            except Exception as e:
                logger.warning("Failed to set selected English voice: %s", e)

    def speak(self, text: str) -> Dict[str, Any]:
        if not self.enabled:
            return {
                "tts_enabled": False,
                "tts_backend": self.backend,
                "tts_status": "disabled",
                "tts_error": "",
            }

        if not text or not text.strip():
            return {
                "tts_enabled": True,
                "tts_backend": self.backend,
                "tts_status": "skipped",
                "tts_error": "empty_navigation_sentence",
            }

        if self.engine is None:
            return {
                "tts_enabled": True,
                "tts_backend": self.backend,
                "tts_status": "skipped",
                "tts_error": "tts_engine_unavailable",
            }

        try:
            self.engine.say(text)
            self.engine.runAndWait()
            return {
                "tts_enabled": True,
                "tts_backend": self.backend,
                "tts_status": "played",
                "tts_error": "",
            }
        except Exception as e:
            logger.warning("TTS playback failed: %s", e)
            return {
                "tts_enabled": True,
                "tts_backend": self.backend,
                "tts_status": "error",
                "tts_error": str(e),
            }

    def release(self):
        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
            self.engine = None
