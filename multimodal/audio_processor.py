"""Audio processing (speech-to-text, text-to-speech)."""

from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Process audio using speech models."""

    def __init__(self, model: str):
        """
        Initialize audio processor.

        Args:
            model: Whisper model to use (required)

        Raises:
            ImportError: If openai library is not installed
        """
        self.model = model
        self._init_client()

    def _init_client(self) -> None:
        """Initialize OpenAI client for Whisper."""
        try:
            import openai
            self.client = openai.AsyncOpenAI()
            logger.info("Audio processor initialized with Whisper")
        except ImportError as e:
            raise ImportError(
                "OpenAI library required for audio processing. "
                "Install with: pip install openai"
            ) from e

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
        # Load audio
        if isinstance(audio, bytes):
            # Create temp file for API
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                f.write(audio)
                audio_path = f.name
        else:
            audio_path = audio

        # Transcribe
        with open(audio_path, 'rb') as audio_file:
            response = await self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language
            )

        return response.text

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
            voice: Voice to use
            output_format: Output format

        Returns:
            Audio bytes
        """
        response = await self.client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
            response_format=output_format
        )

        return response.content

    async def detect_language(self, audio: Union[bytes, str]) -> str:
        """
        Detect language of audio.

        Args:
            audio: Audio bytes or file path

        Returns:
            Language code or "unknown"

        Raises:
            ImportError: If langdetect library is not installed
        """
        # Transcribe without language specified
        transcription = await self.transcribe(audio)

        # Use language detection library
        try:
            from langdetect import detect
            return detect(transcription)
        except ImportError as e:
            raise ImportError(
                "langdetect required for language detection. "
                "Install with: pip install langdetect"
            ) from e

    def get_supported_formats(self) -> list:
        """Get supported audio formats."""
        return ["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
