"""Audio processing (speech-to-text, text-to-speech) via LLMProviderService."""

from typing import Union, Optional
import logging

from ..pipeline.llm_provider_service import LLMProviderService

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process audio using speech models via LLMProviderService."""

    def __init__(
        self,
        llm_service: LLMProviderService,
        model: str = "whisper-1"
    ):
        """
        Initialize audio processor.

        Args:
            llm_service: LLMProviderService instance (required)
            model: Whisper model to use
        """
        self.llm_service = llm_service
        self.model = model

    async def transcribe(
        self,
        audio: Union[bytes, str],
        language: Optional[str] = None
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio: Audio bytes or file path
            language: Optional language code

        Returns:
            Transcribed text
        """
        return await self.llm_service.transcribe(
            audio=audio,
            model=self.model,
            language=language
        )

    async def synthesize(
        self,
        text: str,
        voice: str = "alloy",
        output_format: str = "mp3"
    ) -> bytes:
        """
        Convert text to speech.

        Args:
            text: Text to synthesize
            voice: Voice to use (alloy, echo, fable, onyx, nova, shimmer)
            output_format: Output format (mp3, opus, aac, flac)

        Returns:
            Audio bytes
        """
        return await self.llm_service.synthesize_speech(
            text=text,
            voice=voice,
            output_format=output_format
        )

    async def detect_language(self, audio: Union[bytes, str]) -> str:
        """
        Detect language of audio.

        Args:
            audio: Audio bytes or file path

        Returns:
            Language code or "unknown"
        """
        transcription = await self.transcribe(audio)

        try:
            from langdetect import detect
            return detect(transcription)
        except ImportError:
            logger.warning("langdetect not installed, returning 'unknown'")
            return "unknown"
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "unknown"

    def get_supported_formats(self) -> list:
        """Get supported audio formats."""
        return ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
