"""
Unit tests for multimodal processing.

Tests image, audio, video processing and cross-modal fusion.
"""

import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import base64
import io

from ia_modules.multimodal.processor import (
    MultiModalProcessor,
    MultiModalInput,
    MultiModalOutput,
    MultiModalConfig,
    ModalityType
)
from ia_modules.multimodal.image_processor import ImageProcessor
from ia_modules.multimodal.fusion import ModalityFusion


class TestModalityType:
    """Test ModalityType enum."""

    def test_modality_types(self):
        """ModalityType enum has expected values."""
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"


class TestMultiModalInput:
    """Test MultiModalInput dataclass."""

    def test_creation_minimal(self):
        """MultiModalInput can be created with minimal fields."""
        input_data = MultiModalInput(
            content="test content",
            modality=ModalityType.TEXT
        )

        assert input_data.content == "test content"
        assert input_data.modality == ModalityType.TEXT
        assert input_data.metadata == {}
        assert input_data.prompt is None

    def test_creation_full(self):
        """MultiModalInput can be created with all fields."""
        input_data = MultiModalInput(
            content=b"image data",
            modality=ModalityType.IMAGE,
            metadata={"source": "camera"},
            prompt="Describe this image"
        )

        assert input_data.content == b"image data"
        assert input_data.modality == ModalityType.IMAGE
        assert input_data.metadata == {"source": "camera"}
        assert input_data.prompt == "Describe this image"


class TestMultiModalOutput:
    """Test MultiModalOutput dataclass."""

    def test_creation(self):
        """MultiModalOutput can be created."""
        output = MultiModalOutput(
            result="Combined result",
            modality_results={
                ModalityType.TEXT: "Text result",
                ModalityType.IMAGE: "Image description"
            },
            metadata={"processed": True},
            confidence=0.95
        )

        assert output.result == "Combined result"
        assert len(output.modality_results) == 2
        assert output.metadata["processed"] is True
        assert output.confidence == 0.95

    def test_defaults(self):
        """MultiModalOutput has proper defaults."""
        output = MultiModalOutput(result="test")

        assert output.modality_results == {}
        assert output.metadata == {}
        assert output.confidence == 1.0


class TestMultiModalConfig:
    """Test MultiModalConfig dataclass."""

    def test_creation_defaults(self):
        """MultiModalConfig has proper defaults."""
        config = MultiModalConfig()

        assert len(config.supported_modalities) == 4
        assert config.image_model == "gpt-4-vision-preview"
        assert config.audio_model == "whisper-1"
        assert config.vision_provider == "openai"
        assert config.max_image_size == 2048
        assert config.enable_fusion is True
        assert config.max_concurrent == 3

    def test_creation_custom(self):
        """MultiModalConfig can be customized."""
        config = MultiModalConfig(
            image_model="gpt-4-vision",
            audio_model="whisper-large",
            vision_provider="anthropic",
            max_image_size=1024,
            enable_fusion=False
        )

        assert config.image_model == "gpt-4-vision"
        assert config.audio_model == "whisper-large"
        assert config.vision_provider == "anthropic"
        assert config.max_image_size == 1024
        assert config.enable_fusion is False


@pytest.mark.asyncio
class TestImageProcessor:
    """Test ImageProcessor functionality."""

    @pytest.fixture
    def image_processor(self):
        """Create image processor instance."""
        return ImageProcessor(
            model="gpt-4-vision-preview",
            provider="openai",
            max_size=2048
        )

    @pytest.fixture
    def mock_openai_client(self):
        """Create mock OpenAI client."""
        client = Mock()

        # Mock completion response
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "A beautiful landscape with mountains"
        mock_choice = Mock()
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]

        client.chat = Mock()
        client.chat.completions = Mock()
        client.chat.completions.create = AsyncMock(return_value=mock_response)

        return client

    def test_creation(self, image_processor):
        """ImageProcessor can be created."""
        assert image_processor.model == "gpt-4-vision-preview"
        assert image_processor.provider == "openai"
        assert image_processor.max_size == 2048

    async def test_process_image_bytes(self, image_processor, mock_openai_client):
        """Can process image from bytes."""
        image_processor.client = mock_openai_client

        # Create fake image bytes
        image_bytes = b"fake image data"

        result = await image_processor.process(image_bytes, "Describe this")

        assert isinstance(result, str)
        assert mock_openai_client.chat.completions.create.called

    async def test_process_without_client(self):
        """Processing without client returns fallback message."""
        processor = ImageProcessor(provider="none")

        result = await processor.process(b"data", "test")

        assert "not available" in result.lower()

    async def test_load_image_bytes(self, image_processor):
        """Can load image from bytes."""
        image_bytes = b"test image data"

        loaded = await image_processor._load_image(image_bytes)

        assert loaded == image_bytes

    async def test_resize_image_no_pil(self, image_processor):
        """Resize handles missing PIL gracefully."""
        image_bytes = b"test data"

        with patch('ia_modules.multimodal.image_processor.Image', side_effect=ImportError):
            resized = image_processor._resize_image(image_bytes)
            assert resized == image_bytes

    async def test_get_embedding_fallback(self, image_processor):
        """Can get embedding using fallback method."""
        # Mock the process method
        image_processor.process = AsyncMock(return_value="A test image")

        embedding = await image_processor.get_embedding(b"image data")

        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, float) for x in embedding)

    def test_provider_initialization_openai(self):
        """OpenAI provider initialization."""
        with patch('ia_modules.multimodal.image_processor.openai'):
            processor = ImageProcessor(provider="openai")
            assert processor.provider == "openai"

    def test_provider_initialization_anthropic(self):
        """Anthropic provider initialization."""
        with patch('ia_modules.multimodal.image_processor.anthropic'):
            processor = ImageProcessor(provider="anthropic")
            assert processor.provider == "anthropic"


@pytest.mark.asyncio
class TestModalityFusion:
    """Test ModalityFusion functionality."""

    @pytest.fixture
    def fusion(self):
        """Create fusion instance."""
        return ModalityFusion()

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate = AsyncMock(
            return_value={"content": "Synthesized result"}
        )
        return provider

    async def test_creation(self, fusion):
        """ModalityFusion can be created."""
        assert fusion.llm_provider is None

    async def test_fuse_without_llm(self, fusion):
        """Can fuse results without LLM (simple concatenation)."""
        modality_results = {
            ModalityType.TEXT: "Text content",
            ModalityType.IMAGE: "Image shows a cat"
        }

        result = await fusion.fuse(modality_results)

        assert "Text content" in result
        assert "Image shows a cat" in result
        assert "text" in result.lower()
        assert "image" in result.lower()

    async def test_fuse_with_llm(self, mock_llm_provider):
        """Can fuse results with LLM."""
        fusion = ModalityFusion(llm_provider=mock_llm_provider)

        modality_results = {
            ModalityType.TEXT: "Text content",
            ModalityType.IMAGE: "Image description"
        }

        result = await fusion.fuse(modality_results)

        assert result == "Synthesized result"
        assert mock_llm_provider.generate.called

    async def test_fuse_with_prompt(self, mock_llm_provider):
        """Fusion includes context prompt."""
        fusion = ModalityFusion(llm_provider=mock_llm_provider)

        modality_results = {
            ModalityType.TEXT: "Content"
        }

        await fusion.fuse(modality_results, prompt="Analyze this data")

        # Check that prompt was included in call
        call_args = mock_llm_provider.generate.call_args
        assert "Analyze this data" in call_args.kwargs["prompt"]

    async def test_fuse_llm_error_fallback(self, fusion):
        """Fusion falls back on LLM error."""
        mock_provider = Mock()
        mock_provider.generate = AsyncMock(side_effect=Exception("LLM error"))
        fusion.llm_provider = mock_provider

        modality_results = {
            ModalityType.TEXT: "Content"
        }

        result = await fusion.fuse(modality_results)

        # Should still return concatenated result
        assert "Content" in result

    async def test_calculate_modality_weights(self, fusion):
        """Can calculate weights based on information content."""
        modality_results = {
            ModalityType.TEXT: "Short",
            ModalityType.IMAGE: "Much longer description here"
        }

        weights = fusion.calculate_modality_weights(modality_results)

        assert len(weights) == 2
        assert all(0 <= w <= 1 for w in weights.values())
        # Longer content should have more weight
        assert weights[ModalityType.IMAGE] > weights[ModalityType.TEXT]

    async def test_calculate_modality_weights_equal(self, fusion):
        """Equal length results get equal weights."""
        modality_results = {
            ModalityType.TEXT: "Same",
            ModalityType.IMAGE: "Same"
        }

        weights = fusion.calculate_modality_weights(modality_results)

        assert abs(weights[ModalityType.TEXT] - weights[ModalityType.IMAGE]) < 0.01

    async def test_calculate_modality_weights_empty(self, fusion):
        """Empty results handled gracefully."""
        modality_results = {
            ModalityType.TEXT: "",
            ModalityType.IMAGE: ""
        }

        weights = fusion.calculate_modality_weights(modality_results)

        # Should give equal weights
        assert len(weights) == 2
        assert all(abs(w - 0.5) < 0.01 for w in weights.values())

    async def test_cross_modal_attention_without_llm(self, fusion):
        """Cross-modal attention without LLM returns query result."""
        query_result = "Primary information"
        context_results = {
            ModalityType.IMAGE: "Additional context"
        }

        result = await fusion.cross_modal_attention(
            ModalityType.TEXT,
            query_result,
            context_results
        )

        assert result == query_result

    async def test_cross_modal_attention_with_llm(self, mock_llm_provider):
        """Cross-modal attention with LLM enhances result."""
        fusion = ModalityFusion(llm_provider=mock_llm_provider)

        query_result = "Primary info"
        context_results = {
            ModalityType.IMAGE: "Context"
        }

        result = await fusion.cross_modal_attention(
            ModalityType.TEXT,
            query_result,
            context_results
        )

        assert result == "Synthesized result"
        assert mock_llm_provider.generate.called

    async def test_cross_modal_attention_empty_context(self, fusion):
        """Cross-modal attention with no context returns query."""
        query_result = "Primary"

        result = await fusion.cross_modal_attention(
            ModalityType.TEXT,
            query_result,
            {}
        )

        assert result == query_result

    async def test_cross_modal_attention_llm_error(self, fusion):
        """Cross-modal attention handles LLM error."""
        mock_provider = Mock()
        mock_provider.generate = AsyncMock(side_effect=Exception("Error"))
        fusion.llm_provider = mock_provider

        query_result = "Primary"
        context_results = {ModalityType.IMAGE: "Context"}

        result = await fusion.cross_modal_attention(
            ModalityType.TEXT,
            query_result,
            context_results
        )

        assert result == query_result


@pytest.mark.asyncio
class TestMultiModalProcessor:
    """Test MultiModalProcessor functionality."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return MultiModalConfig(
            enable_fusion=False  # Disable for simpler testing
        )

    @pytest.fixture
    def processor(self, config):
        """Create processor instance."""
        return MultiModalProcessor(config=config)

    @pytest.fixture
    def mock_llm_provider(self):
        """Create mock LLM provider."""
        provider = Mock()
        provider.generate = AsyncMock(
            return_value={"content": "Text result"}
        )
        return provider

    def test_creation(self, processor):
        """MultiModalProcessor can be created."""
        assert processor.config is not None
        assert processor.image_processor is not None
        assert processor.audio_processor is not None
        assert processor.video_processor is not None

    def test_creation_with_fusion(self):
        """Processor with fusion enabled."""
        config = MultiModalConfig(enable_fusion=True)
        processor = MultiModalProcessor(config=config)

        assert processor.fusion is not None

    async def test_process_single_text_input(self, processor, mock_llm_provider):
        """Can process single text input."""
        processor.llm_provider = mock_llm_provider

        inputs = [
            MultiModalInput(
                content="Test text",
                modality=ModalityType.TEXT
            )
        ]

        result = await processor.process(inputs)

        assert isinstance(result, MultiModalOutput)
        assert result.result is not None

    async def test_process_multiple_inputs(self, processor):
        """Can process multiple modality inputs."""
        # Mock the processors
        processor.image_processor.process = AsyncMock(return_value="Image result")
        processor.audio_processor.transcribe = AsyncMock(return_value="Audio text")

        inputs = [
            MultiModalInput(content=b"image", modality=ModalityType.IMAGE),
            MultiModalInput(content=b"audio", modality=ModalityType.AUDIO)
        ]

        result = await processor.process(inputs)

        assert isinstance(result, MultiModalOutput)
        assert ModalityType.IMAGE in result.modality_results
        assert ModalityType.AUDIO in result.modality_results

    async def test_process_with_global_prompt(self, processor):
        """Process with global prompt."""
        processor.image_processor.process = AsyncMock(return_value="Image result")

        inputs = [
            MultiModalInput(content=b"image", modality=ModalityType.IMAGE)
        ]

        result = await processor.process(
            inputs,
            global_prompt="Analyze this carefully"
        )

        # Check that prompt was used
        assert processor.image_processor.process.called
        call_args = processor.image_processor.process.call_args
        assert "Analyze this carefully" in call_args[0]

    async def test_process_input_specific_prompt(self, processor):
        """Input-specific prompt overrides global."""
        processor.image_processor.process = AsyncMock(return_value="Image result")

        inputs = [
            MultiModalInput(
                content=b"image",
                modality=ModalityType.IMAGE,
                prompt="Specific prompt"
            )
        ]

        await processor.process(inputs, global_prompt="Global prompt")

        call_args = processor.image_processor.process.call_args
        assert "Specific prompt" in call_args[0]

    async def test_process_image(self, processor):
        """Can process image directly."""
        processor.image_processor.process = AsyncMock(return_value="Image description")

        result = await processor.process_image(b"image data", "Describe")

        assert result == "Image description"

    async def test_process_audio(self, processor):
        """Can process audio directly."""
        processor.audio_processor.transcribe = AsyncMock(return_value="Transcribed text")

        result = await processor.process_audio(b"audio data")

        assert result == "Transcribed text"

    async def test_process_video(self, processor):
        """Can process video directly."""
        processor.video_processor.process = AsyncMock(return_value="Video analysis")

        result = await processor.process_video(b"video data", "Analyze")

        assert result == "Video analysis"

    async def test_process_with_fusion_enabled(self):
        """Process with fusion combines results."""
        config = MultiModalConfig(enable_fusion=True)
        processor = MultiModalProcessor(config=config)

        # Mock processors
        processor.image_processor.process = AsyncMock(return_value="Image result")
        processor.audio_processor.transcribe = AsyncMock(return_value="Audio result")
        processor.fusion.fuse = AsyncMock(return_value="Fused result")

        inputs = [
            MultiModalInput(content=b"image", modality=ModalityType.IMAGE),
            MultiModalInput(content=b"audio", modality=ModalityType.AUDIO)
        ]

        result = await processor.process(inputs)

        assert result.result == "Fused result"
        assert processor.fusion.fuse.called

    async def test_process_unsupported_modality(self, processor):
        """Unsupported modality is handled gracefully."""
        # Create invalid input (will be ignored)
        inputs = [
            MultiModalInput(
                content="test",
                modality=ModalityType.TEXT
            )
        ]

        # Mock _process_text to raise error
        processor._process_text = AsyncMock(side_effect=Exception("Not supported"))

        # Should not crash
        result = await processor.process(inputs)
        assert isinstance(result, MultiModalOutput)


class TestEdgeCases:
    """Test edge cases and error handling."""

    @pytest.mark.asyncio
    async def test_fusion_empty_modality_results(self):
        """Fusion handles empty modality results."""
        fusion = ModalityFusion()

        result = await fusion.fuse({})

        assert result == ""

    @pytest.mark.asyncio
    async def test_fusion_single_modality(self):
        """Fusion handles single modality."""
        fusion = ModalityFusion()

        modality_results = {
            ModalityType.TEXT: "Only text"
        }

        result = await fusion.fuse(modality_results)

        assert "Only text" in result

    @pytest.mark.asyncio
    async def test_processor_empty_inputs(self):
        """Processor handles empty input list."""
        config = MultiModalConfig()
        processor = MultiModalProcessor(config=config)

        result = await processor.process([])

        assert isinstance(result, MultiModalOutput)
        assert len(result.modality_results) == 0

    @pytest.mark.asyncio
    async def test_image_processor_invalid_image_data(self):
        """Image processor handles invalid image data."""
        processor = ImageProcessor(provider="none")

        # Should not crash on invalid data
        result = await processor.process(b"invalid", "test")

        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_modality_weights_single_result(self):
        """Weight calculation with single result."""
        fusion = ModalityFusion()

        modality_results = {
            ModalityType.TEXT: "Content"
        }

        weights = fusion.calculate_modality_weights(modality_results)

        assert weights[ModalityType.TEXT] == 1.0

    @pytest.mark.asyncio
    async def test_processor_metadata_preservation(self):
        """Processor preserves metadata from inputs."""
        config = MultiModalConfig(enable_fusion=False)
        processor = MultiModalProcessor(config=config)

        processor.image_processor.process = AsyncMock(return_value="Result")

        inputs = [
            MultiModalInput(
                content=b"data",
                modality=ModalityType.IMAGE,
                metadata={"source": "camera"}
            )
        ]

        result = await processor.process(inputs)

        assert "num_modalities" in result.metadata
        assert result.metadata["num_modalities"] == 1


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests combining multiple components."""

    async def test_end_to_end_multimodal_processing(self):
        """End-to-end test with mocked components."""
        config = MultiModalConfig(enable_fusion=True)
        processor = MultiModalProcessor(config=config)

        # Mock all processors
        processor.image_processor.process = AsyncMock(return_value="An image of a cat")
        processor.audio_processor.transcribe = AsyncMock(return_value="Hello world")
        processor.fusion.fuse = AsyncMock(return_value="Combined: cat image and hello")

        inputs = [
            MultiModalInput(content=b"img", modality=ModalityType.IMAGE),
            MultiModalInput(content=b"aud", modality=ModalityType.AUDIO)
        ]

        result = await processor.process(
            inputs,
            global_prompt="Analyze these inputs"
        )

        assert result.result == "Combined: cat image and hello"
        assert len(result.modality_results) == 2
        assert ModalityType.IMAGE in result.modality_results
        assert ModalityType.AUDIO in result.modality_results

    async def test_cross_modal_enhancement_workflow(self):
        """Test cross-modal enhancement workflow."""
        mock_llm = Mock()
        mock_llm.generate = AsyncMock(return_value={"content": "Enhanced"})

        fusion = ModalityFusion(llm_provider=mock_llm)

        # Primary text with image context
        result = await fusion.cross_modal_attention(
            ModalityType.TEXT,
            "The object is blue",
            {ModalityType.IMAGE: "Shows a blue car"}
        )

        assert result == "Enhanced"
        assert mock_llm.generate.called
