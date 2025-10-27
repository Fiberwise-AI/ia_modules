"""Image processing with vision models."""

from typing import Union, List, Optional
import base64
import logging
import io

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Process images using vision models."""

    def __init__(
        self,
        model: str,
        provider: str = "openai",
        max_size: int = 2048
    ):
        """
        Initialize image processor.

        Args:
            model: Vision model to use (required)
            provider: Provider name ('openai' or 'anthropic')
            max_size: Maximum image dimension in pixels

        Raises:
            ValueError: If provider is not supported
            ImportError: If provider library is not installed
        """
        self.model = model
        self.provider = provider
        self.max_size = max_size
        self._init_provider()

    def _init_provider(self) -> None:
        """Initialize vision provider."""
        if self.provider == "openai":
            try:
                import openai
                self.client = openai.AsyncOpenAI()
                logger.info("Using OpenAI vision")
            except ImportError as e:
                raise ImportError(
                    "OpenAI library not installed. Install with: pip install openai"
                ) from e
        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.AsyncAnthropic()
                logger.info("Using Anthropic vision")
            except ImportError as e:
                raise ImportError(
                    "Anthropic library not installed. Install with: pip install anthropic"
                ) from e
        else:
            raise ValueError(
                f"Unsupported vision provider: {self.provider}. "
                f"Supported providers: openai, anthropic"
            )

    async def process(
        self,
        image: Union[bytes, str],
        prompt: str = "Describe this image"
    ) -> str:
        """Process image with vision model."""
        # Load and resize image
        image_data = await self._load_image(image)
        image_data = self._resize_image(image_data)

        # Encode to base64
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Call vision API based on provider
        if self.provider == "openai":
            return await self._process_openai(base64_image, prompt)
        elif self.provider == "anthropic":
            return await self._process_anthropic(base64_image, prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    async def _process_openai(self, base64_image: str, prompt: str) -> str:
        """Process with OpenAI vision."""
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }],
            max_tokens=500
        )
        return response.choices[0].message.content

    async def _process_anthropic(self, base64_image: str, prompt: str) -> str:
        """Process with Anthropic vision."""
        message = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        return message.content[0].text

    async def _load_image(self, image: Union[bytes, str]) -> bytes:
        """Load image from bytes or file path."""
        if isinstance(image, bytes):
            return image

        # Load from file
        with open(image, 'rb') as f:
            return f.read()

    def _resize_image(self, image_data: bytes) -> bytes:
        """
        Resize image if needed.

        Args:
            image_data: Image bytes

        Returns:
            Resized image bytes (or original if resize not needed/fails)

        Note:
            If PIL is not available, returns original image with a warning.
            This is a legitimate degraded mode since images can still be processed.
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning(
                "PIL not available, skipping resize. "
                "Install with: pip install Pillow"
            )
            return image_data

        try:
            img = Image.open(io.BytesIO(image_data))

            # Resize if larger than max_size
            if max(img.size) > self.max_size:
                ratio = self.max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)

                # Convert back to bytes
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

        Raises:
            ImportError: If sentence-transformers is not installed

        Note:
            Currently uses text embeddings of image description.
            For production use, consider using CLIP or similar vision models.
        """
        # Simplified: convert image description to embedding
        description = await self.process(image, "Describe this image briefly")

        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            return model.encode(description).tolist()
        except ImportError as e:
            raise ImportError(
                "sentence-transformers required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e
