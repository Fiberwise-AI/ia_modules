"""Image processing with vision models via LLMProviderService."""

from typing import Union, List, Optional
import base64
import logging
import io

from ..pipeline.llm_provider_service import LLMProviderService

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images using vision models via LLMProviderService."""

    def __init__(
        self,
        llm_service: LLMProviderService,
        model: str = "gpt-4-vision-preview",
        max_size: int = 2048,
        provider_name: Optional[str] = None
    ):
        """
        Initialize image processor.

        Args:
            llm_service: LLMProviderService instance (required)
            model: Vision model to use
            max_size: Maximum image dimension in pixels
            provider_name: Provider name to use with llm_service
        """
        self.llm_service = llm_service
        self.model = model
        self.max_size = max_size
        self.provider_name = provider_name

    async def process(
        self,
        image: Union[bytes, str],
        prompt: str = "Describe this image"
    ) -> str:
        """
        Process image with vision model.

        Args:
            image: Image bytes or file path
            prompt: Prompt for image analysis

        Returns:
            Model's description/analysis of the image
        """
        image_data = await self._load_image(image)
        image_data = self._resize_image(image_data)

        return await self.llm_service.generate_vision(
            image=image_data,
            prompt=prompt,
            provider_name=self.provider_name,
            model=self.model
        )

    async def _load_image(self, image: Union[bytes, str]) -> bytes:
        """Load image from bytes or file path."""
        if isinstance(image, bytes):
            return image

        with open(image, 'rb') as f:
            return f.read()

    def _resize_image(self, image_data: bytes) -> bytes:
        """Resize image if needed."""
        try:
            from PIL import Image
        except ImportError:
            logger.debug("PIL not available, skipping resize")
            return image_data

        try:
            img = Image.open(io.BytesIO(image_data))

            if max(img.size) > self.max_size:
                ratio = self.max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

                output = io.BytesIO()
                img.save(output, format='JPEG', quality=85)
                return output.getvalue()

            return image_data
        except Exception as e:
            logger.warning(f"Resize failed: {e}, using original image")
            return image_data

    async def get_embedding(self, image: Union[bytes, str]) -> List[float]:
        """
        Get embedding for image.

        Args:
            image: Image bytes or file path

        Returns:
            Embedding vector
        """
        description = await self.process(image, "Describe this image in detail")

        # Use LLMProviderService for embedding if available
        import litellm
        response = await litellm.aembedding(
            model="text-embedding-ada-002",
            input=description
        )

        return response.data[0].embedding
