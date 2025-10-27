"""
Multi-Modal Processor - Main interface for multi-modal AI.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModalityType(Enum):
    """Types of modalities supported."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultiModalInput:
    """Input for multi-modal processing."""
    content: Any  # bytes, str, PIL.Image, etc.
    modality: ModalityType
    metadata: Dict[str, Any] = field(default_factory=dict)
    prompt: Optional[str] = None  # Optional prompt for this specific input


@dataclass
class MultiModalOutput:
    """Output from multi-modal processing."""
    result: str  # Main textual result
    modality_results: Dict[ModalityType, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class MultiModalConfig:
    """Configuration for multi-modal processing."""
    supported_modalities: List[ModalityType] = field(
        default_factory=lambda: list(ModalityType)
    )
    image_model: Optional[str] = None  # Required for image processing
    audio_model: Optional[str] = None  # Required for audio processing
    vision_provider: str = "openai"  # "openai", "anthropic"
    max_image_size: int = 2048
    audio_format: str = "mp3"
    video_fps: int = 1  # Frames per second for video processing
    enable_fusion: bool = True  # Cross-modal fusion
    max_concurrent: int = 3  # Max concurrent modal processing


class MultiModalProcessor:
    """
    Main processor for multi-modal AI operations.

    Coordinates processing across different modalities and
    optionally fuses results.
    """

    def __init__(
        self,
        config: Optional[MultiModalConfig] = None,
        llm_provider: Optional[Any] = None
    ):
        """
        Initialize multi-modal processor.

        Args:
            config: Configuration
            llm_provider: LLM provider for text processing
        """
        self.config = config or MultiModalConfig()
        self.llm_provider = llm_provider

        # Initialize processors on-demand (lazy initialization)
        self._image_processor = None
        self._audio_processor = None
        self._video_processor = None

        if self.config.enable_fusion:
            from .fusion import ModalityFusion
            self.fusion = ModalityFusion(llm_provider=llm_provider)
        else:
            self.fusion = None

    @property
    def image_processor(self):
        """Lazy-load image processor."""
        if self._image_processor is None:
            if not self.config.image_model:
                raise ValueError(
                    "image_model must be configured for image processing. "
                    "Example: MultiModalConfig(image_model='gpt-4-vision-preview')"
                )
            from .image_processor import ImageProcessor
            self._image_processor = ImageProcessor(
                model=self.config.image_model,
                provider=self.config.vision_provider,
                max_size=self.config.max_image_size
            )
        return self._image_processor

    @property
    def audio_processor(self):
        """Lazy-load audio processor."""
        if self._audio_processor is None:
            if not self.config.audio_model:
                raise ValueError(
                    "audio_model must be configured for audio processing. "
                    "Example: MultiModalConfig(audio_model='whisper-1')"
                )
            from .audio_processor import AudioProcessor
            self._audio_processor = AudioProcessor(model=self.config.audio_model)
        return self._audio_processor

    @property
    def video_processor(self):
        """Lazy-load video processor."""
        if self._video_processor is None:
            from .video_processor import VideoProcessor
            self._video_processor = VideoProcessor(fps=self.config.video_fps)
        return self._video_processor

    async def process(
        self,
        inputs: List[MultiModalInput],
        global_prompt: Optional[str] = None
    ) -> MultiModalOutput:
        """
        Process multiple modalities.

        Args:
            inputs: List of multi-modal inputs
            global_prompt: Optional prompt for all inputs

        Returns:
            Multi-modal output with fused results
        """
        logger.info(f"Processing {len(inputs)} multi-modal inputs")

        # Process each modality
        modality_results = {}

        for input_item in inputs:
            logger.info(f"Processing {input_item.modality.value}")

            if input_item.modality == ModalityType.TEXT:
                result = await self._process_text(input_item, global_prompt)
            elif input_item.modality == ModalityType.IMAGE:
                result = await self.process_image(
                    input_item.content,
                    input_item.prompt or global_prompt or "Describe this image"
                )
            elif input_item.modality == ModalityType.AUDIO:
                result = await self.process_audio(input_item.content)
            elif input_item.modality == ModalityType.VIDEO:
                result = await self.process_video(
                    input_item.content,
                    input_item.prompt or global_prompt
                )
            else:
                logger.warning(f"Unsupported modality: {input_item.modality}")
                continue

            modality_results[input_item.modality] = result

        # Fuse results if enabled
        if self.fusion and len(modality_results) > 1:
            fused_result = await self.fusion.fuse(
                modality_results,
                global_prompt
            )
        else:
            # Just concatenate results
            fused_result = "\n\n".join(
                f"**{modality.value.title()}**: {result}"
                for modality, result in modality_results.items()
            )

        return MultiModalOutput(
            result=fused_result,
            modality_results=modality_results,
            metadata={"num_modalities": len(modality_results)}
        )

    async def process_image(
        self,
        image: Union[bytes, str],
        prompt: str = "Describe this image in detail"
    ) -> str:
        """
        Process an image with vision model.

        Args:
            image: Image bytes or file path
            prompt: Prompt for the vision model

        Returns:
            Image description/analysis
        """
        return await self.image_processor.process(image, prompt)

    async def process_audio(self, audio: Union[bytes, str]) -> str:
        """
        Process audio (speech-to-text).

        Args:
            audio: Audio bytes or file path

        Returns:
            Transcribed text
        """
        return await self.audio_processor.transcribe(audio)

    async def process_video(
        self,
        video: Union[bytes, str],
        prompt: Optional[str] = None
    ) -> str:
        """
        Process video by extracting frames and analyzing.

        Args:
            video: Video bytes or file path
            prompt: Optional prompt for frame analysis

        Returns:
            Video analysis
        """
        return await self.video_processor.process(
            video,
            prompt,
            self.image_processor
        )

    async def _process_text(
        self,
        input_item: MultiModalInput,
        global_prompt: Optional[str]
    ) -> str:
        """Process text input."""
        text = input_item.content
        if isinstance(text, bytes):
            text = text.decode('utf-8')

        # If there's a prompt, process with LLM
        if global_prompt and self.llm_provider:
            combined_prompt = f"{global_prompt}\n\nText: {text}"
            response = await self.llm_provider.generate(prompt=combined_prompt)
            return response.get("content", response.get("text", text))

        return text

    async def generate_multimodal_embeddings(
        self,
        inputs: List[MultiModalInput]
    ) -> List[List[float]]:
        """
        Generate embeddings for multi-modal inputs.

        Args:
            inputs: List of inputs

        Returns:
            List of embedding vectors
        """
        embeddings = []

        for input_item in inputs:
            if input_item.modality == ModalityType.TEXT:
                # Use text embedding
                embedding = await self._get_text_embedding(input_item.content)
            elif input_item.modality == ModalityType.IMAGE:
                # Use vision embedding
                embedding = await self.image_processor.get_embedding(
                    input_item.content
                )
            else:
                # Convert to text first, then embed
                text = await self.process([input_item])
                embedding = await self._get_text_embedding(text.result)

            embeddings.append(embedding)

        return embeddings

    async def _get_text_embedding(self, text: str) -> List[float]:
        """
        Get text embedding.

        Raises:
            ImportError: If sentence-transformers is not installed
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embedding = model.encode(text)
            return embedding.tolist()
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for text embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e
