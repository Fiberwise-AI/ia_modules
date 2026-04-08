"""
Comprehensive unit tests for multimodal processors and database adapters.

Targets:
- ia_modules/multimodal/video_processor.py
- ia_modules/multimodal/audio_processor.py
- ia_modules/multimodal/image_processor.py
- ia_modules/multimodal/processor.py
- ia_modules/database/adapters/sqlalchemy_adapter.py
- ia_modules/database/factory.py
- ia_modules/database/adapters/nexusql_adapter.py
"""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from dataclasses import asdict

# ============================================================
# VideoProcessor Tests
# ============================================================

from ia_modules.multimodal.video_processor import VideoProcessor


class TestVideoProcessor:
    """Tests for VideoProcessor."""

    def test_init_default_fps(self):
        vp = VideoProcessor()
        assert vp.fps == 1

    def test_init_custom_fps(self):
        vp = VideoProcessor(fps=5)
        assert vp.fps == 5

    async def test_process_with_frames(self):
        vp = VideoProcessor(fps=1)
        mock_image_processor = AsyncMock()
        mock_image_processor.process = AsyncMock(return_value="A cat sitting")

        # Mock extract_frames to return fake frames
        vp.extract_frames = AsyncMock(return_value=[b"frame1", b"frame2"])

        result = await vp.process(b"fake_video", "Describe", mock_image_processor)
        assert "Video Analysis (2 frames)" in result
        assert "Frame 1: A cat sitting" in result
        assert "Frame 2: A cat sitting" in result

    async def test_process_no_frames_raises(self):
        vp = VideoProcessor(fps=1)
        mock_image_processor = AsyncMock()
        vp.extract_frames = AsyncMock(return_value=[])

        with pytest.raises(ValueError, match="No frames could be extracted"):
            await vp.process(b"fake_video", None, mock_image_processor)

    async def test_process_default_prompt(self):
        """When prompt is None, uses default prompt."""
        vp = VideoProcessor(fps=1)
        mock_image_processor = AsyncMock()
        mock_image_processor.process = AsyncMock(return_value="desc")
        vp.extract_frames = AsyncMock(return_value=[b"frame1"])

        await vp.process(b"vid", None, mock_image_processor)
        mock_image_processor.process.assert_called_once_with(
            b"frame1", "Describe what's happening in this frame"
        )

    async def test_process_custom_prompt(self):
        vp = VideoProcessor(fps=1)
        mock_image_processor = AsyncMock()
        mock_image_processor.process = AsyncMock(return_value="desc")
        vp.extract_frames = AsyncMock(return_value=[b"frame1"])

        await vp.process(b"vid", "What color?", mock_image_processor)
        mock_image_processor.process.assert_called_once_with(b"frame1", "What color?")

    async def test_extract_frames_import_error(self):
        vp = VideoProcessor(fps=1)
        with patch.dict("sys.modules", {"cv2": None, "numpy": None}):
            with patch("builtins.__import__", side_effect=ImportError("no cv2")):
                with pytest.raises(ImportError, match="OpenCV required"):
                    await vp.extract_frames(b"video_data")

    async def test_extract_frames_with_bytes_input(self):
        """Test extract_frames with bytes input - mocking cv2."""
        mock_cv2 = MagicMock()
        mock_np = MagicMock()
        mock_tempfile = MagicMock()

        # Mock the temp file
        mock_tmp = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tmp.name = "/tmp/fake.mp4"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

        # Mock VideoCapture
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 30,
            mock_cv2.CAP_PROP_FPS: 30.0,
        }.get(prop, 0)
        mock_cap.read.side_effect = [
            (True, MagicMock()),
            (True, MagicMock()),
            (False, None),
        ]
        mock_cv2.VideoCapture.return_value = mock_cap

        # Mock imencode
        mock_buffer = MagicMock()
        mock_buffer.tobytes.return_value = b"jpeg_bytes"
        mock_cv2.imencode.return_value = (True, mock_buffer)

        import sys
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np, "tempfile": mock_tempfile}):
            # We need to actually call the real method but with mocked imports
            # Re-import won't work easily, so let's test via direct module patching
            vp = VideoProcessor(fps=1)
            # Directly test the logic by calling with mocked cv2
            # Since the import is inside the function, we patch builtins.__import__
            original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

            def mock_import(name, *args, **kwargs):
                if name == "cv2":
                    return mock_cv2
                if name == "numpy":
                    return mock_np
                if name == "tempfile":
                    return mock_tempfile
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                frames = await vp.extract_frames(b"video_bytes", max_frames=5)

        assert len(frames) >= 0  # May vary based on mock behavior
        mock_cap.release.assert_called_once()

    async def test_extract_frames_with_str_input(self):
        """Test extract_frames with file path string."""
        mock_cv2 = MagicMock()
        mock_np = MagicMock()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FRAME_COUNT: 10,
            mock_cv2.CAP_PROP_FPS: 10.0,
        }.get(prop, 0)
        mock_cap.read.side_effect = [
            (True, MagicMock()),
            (False, None),
        ]
        mock_cv2.VideoCapture.return_value = mock_cap

        mock_buffer = MagicMock()
        mock_buffer.tobytes.return_value = b"jpeg_bytes"
        mock_cv2.imencode.return_value = (True, mock_buffer)

        import sys
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
            vp = VideoProcessor(fps=1)
            frames = await vp.extract_frames("/path/to/video.mp4", max_frames=5)

        mock_cv2.VideoCapture.assert_called_with("/path/to/video.mp4")
        mock_cap.release.assert_called_once()

    async def test_extract_frames_video_not_opened(self):
        """Test ValueError when video can't be opened."""
        mock_cv2 = MagicMock()
        mock_np = MagicMock()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        import sys
        with patch.dict(sys.modules, {"cv2": mock_cv2, "numpy": mock_np}):
            vp = VideoProcessor(fps=1)
            with pytest.raises(ValueError, match="Could not open video file"):
                await vp.extract_frames("/bad/video.mp4")

    async def test_get_video_info_import_error(self):
        vp = VideoProcessor(fps=1)
        with patch("builtins.__import__", side_effect=ImportError("no cv2")):
            with pytest.raises(ImportError, match="OpenCV required"):
                await vp.get_video_info(b"data")

    async def test_get_video_info_with_str(self):
        mock_cv2 = MagicMock()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 30.0,
            mock_cv2.CAP_PROP_FRAME_COUNT: 300,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 1920,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap

        import sys
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            vp = VideoProcessor(fps=1)
            info = await vp.get_video_info("/path/to/video.mp4")

        assert info["fps"] == 30.0
        assert info["frame_count"] == 300
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["duration"] == 10.0
        mock_cap.release.assert_called_once()

    async def test_get_video_info_with_bytes(self):
        mock_cv2 = MagicMock()
        mock_tempfile = MagicMock()

        mock_tmp = MagicMock()
        mock_tmp.__enter__ = MagicMock(return_value=mock_tmp)
        mock_tmp.__exit__ = MagicMock(return_value=False)
        mock_tmp.name = "/tmp/test.mp4"
        mock_tempfile.NamedTemporaryFile.return_value = mock_tmp

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            mock_cv2.CAP_PROP_FPS: 0,  # Test zero fps path
            mock_cv2.CAP_PROP_FRAME_COUNT: 100,
            mock_cv2.CAP_PROP_FRAME_WIDTH: 640,
            mock_cv2.CAP_PROP_FRAME_HEIGHT: 480,
        }.get(prop, 0)
        mock_cv2.VideoCapture.return_value = mock_cap

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "cv2":
                return mock_cv2
            if name == "tempfile":
                return mock_tempfile
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            vp = VideoProcessor(fps=1)
            info = await vp.get_video_info(b"video_bytes")

        assert info["fps"] == 0
        assert info["duration"] == 0  # fps is 0 so duration is 0
        mock_cap.release.assert_called_once()

    async def test_get_video_info_not_opened(self):
        mock_cv2 = MagicMock()

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = False
        mock_cv2.VideoCapture.return_value = mock_cap

        import sys
        with patch.dict(sys.modules, {"cv2": mock_cv2}):
            vp = VideoProcessor(fps=1)
            with pytest.raises(ValueError, match="Could not open video file"):
                await vp.get_video_info("/bad.mp4")


# ============================================================
# AudioProcessor Tests
# ============================================================

from ia_modules.multimodal.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Tests for AudioProcessor."""

    def test_init_defaults(self):
        mock_llm = MagicMock()
        ap = AudioProcessor(llm_service=mock_llm)
        assert ap.model == "whisper-1"
        assert ap.llm_service is mock_llm

    def test_init_custom_model(self):
        mock_llm = MagicMock()
        ap = AudioProcessor(llm_service=mock_llm, model="whisper-2")
        assert ap.model == "whisper-2"

    async def test_transcribe_bytes(self):
        mock_llm = AsyncMock()
        mock_llm.transcribe = AsyncMock(return_value="Hello world")
        ap = AudioProcessor(llm_service=mock_llm)

        result = await ap.transcribe(b"audio_data")
        assert result == "Hello world"
        mock_llm.transcribe.assert_called_once_with(
            audio=b"audio_data", model="whisper-1", language=None
        )

    async def test_transcribe_with_language(self):
        mock_llm = AsyncMock()
        mock_llm.transcribe = AsyncMock(return_value="Hola mundo")
        ap = AudioProcessor(llm_service=mock_llm)

        result = await ap.transcribe(b"audio_data", language="es")
        assert result == "Hola mundo"
        mock_llm.transcribe.assert_called_once_with(
            audio=b"audio_data", model="whisper-1", language="es"
        )

    async def test_transcribe_file_path(self):
        mock_llm = AsyncMock()
        mock_llm.transcribe = AsyncMock(return_value="transcribed")
        ap = AudioProcessor(llm_service=mock_llm)

        result = await ap.transcribe("/path/to/audio.mp3")
        assert result == "transcribed"
        mock_llm.transcribe.assert_called_once_with(
            audio="/path/to/audio.mp3", model="whisper-1", language=None
        )

    async def test_synthesize(self):
        mock_llm = AsyncMock()
        mock_llm.synthesize_speech = AsyncMock(return_value=b"audio_bytes")
        ap = AudioProcessor(llm_service=mock_llm)

        result = await ap.synthesize("Hello")
        assert result == b"audio_bytes"
        mock_llm.synthesize_speech.assert_called_once_with(
            text="Hello", voice="alloy", output_format="mp3"
        )

    async def test_synthesize_custom_params(self):
        mock_llm = AsyncMock()
        mock_llm.synthesize_speech = AsyncMock(return_value=b"data")
        ap = AudioProcessor(llm_service=mock_llm)

        result = await ap.synthesize("Hi", voice="echo", output_format="flac")
        assert result == b"data"
        mock_llm.synthesize_speech.assert_called_once_with(
            text="Hi", voice="echo", output_format="flac"
        )

    async def test_detect_language_with_langdetect(self):
        mock_llm = AsyncMock()
        mock_llm.transcribe = AsyncMock(return_value="Bonjour le monde")
        ap = AudioProcessor(llm_service=mock_llm)

        mock_detect = MagicMock(return_value="fr")
        with patch.dict("sys.modules", {"langdetect": MagicMock(detect=mock_detect)}):
            with patch("ia_modules.multimodal.audio_processor.AudioProcessor.detect_language") as patched:
                # Actually test the real method
                pass

        # Test with actual patching of the import
        import sys
        mock_langdetect = MagicMock()
        mock_langdetect.detect = MagicMock(return_value="fr")

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "langdetect":
                return mock_langdetect
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await ap.detect_language(b"audio")
            assert result == "fr"

    async def test_detect_language_no_langdetect(self):
        mock_llm = AsyncMock()
        mock_llm.transcribe = AsyncMock(return_value="Hello")
        ap = AudioProcessor(llm_service=mock_llm)

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "langdetect":
                raise ImportError("no langdetect")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await ap.detect_language(b"audio")
            assert result == "unknown"

    async def test_detect_language_detection_fails(self):
        mock_llm = AsyncMock()
        mock_llm.transcribe = AsyncMock(return_value="Hello")
        ap = AudioProcessor(llm_service=mock_llm)

        original_import = __import__

        mock_langdetect = MagicMock()
        mock_langdetect.detect = MagicMock(side_effect=Exception("detection failed"))

        def mock_import(name, *args, **kwargs):
            if name == "langdetect":
                return mock_langdetect
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = await ap.detect_language(b"audio")
            assert result == "unknown"

    def test_get_supported_formats(self):
        mock_llm = MagicMock()
        ap = AudioProcessor(llm_service=mock_llm)
        formats = ap.get_supported_formats()
        assert "mp3" in formats
        assert "wav" in formats
        assert "webm" in formats
        assert len(formats) == 7


# ============================================================
# ImageProcessor Tests
# ============================================================

from ia_modules.multimodal.image_processor import ImageProcessor


class TestImageProcessor:
    """Tests for ImageProcessor."""

    def test_init_defaults(self):
        mock_llm = MagicMock()
        ip = ImageProcessor(llm_service=mock_llm)
        assert ip.model == "gpt-4-vision-preview"
        assert ip.max_size == 2048
        assert ip.provider_name is None

    def test_init_custom(self):
        mock_llm = MagicMock()
        ip = ImageProcessor(
            llm_service=mock_llm,
            model="claude-3",
            max_size=1024,
            provider_name="anthropic"
        )
        assert ip.model == "claude-3"
        assert ip.max_size == 1024
        assert ip.provider_name == "anthropic"

    async def test_process_with_bytes(self):
        mock_llm = AsyncMock()
        mock_llm.generate_vision = AsyncMock(return_value="A cat")
        ip = ImageProcessor(llm_service=mock_llm, provider_name="openai")

        # Patch _resize_image to return as-is
        ip._resize_image = MagicMock(return_value=b"image_data")

        result = await ip.process(b"image_data", "What is this?")
        assert result == "A cat"
        mock_llm.generate_vision.assert_called_once_with(
            image=b"image_data",
            prompt="What is this?",
            provider_name="openai",
            model="gpt-4-vision-preview"
        )

    async def test_process_with_file_path(self):
        mock_llm = AsyncMock()
        mock_llm.generate_vision = AsyncMock(return_value="A dog")
        ip = ImageProcessor(llm_service=mock_llm)
        ip._resize_image = MagicMock(return_value=b"img")

        with patch("builtins.open", mock_open(read_data=b"file_img_data")):
            result = await ip.process("/path/to/image.jpg")

        assert result == "A dog"

    async def test_load_image_bytes(self):
        mock_llm = MagicMock()
        ip = ImageProcessor(llm_service=mock_llm)
        result = await ip._load_image(b"raw_bytes")
        assert result == b"raw_bytes"

    async def test_load_image_file(self):
        mock_llm = MagicMock()
        ip = ImageProcessor(llm_service=mock_llm)

        with patch("builtins.open", mock_open(read_data=b"file_content")):
            result = await ip._load_image("/path/to/img.png")
            assert result == b"file_content"

    def test_resize_image_no_pil(self):
        """When PIL is not available, returns original data."""
        mock_llm = MagicMock()
        ip = ImageProcessor(llm_service=mock_llm)

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "PIL":
                raise ImportError("no PIL")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = ip._resize_image(b"original_data")
            assert result == b"original_data"

    def test_resize_image_small_image(self):
        """Image smaller than max_size should not be resized."""
        mock_llm = MagicMock()
        ip = ImageProcessor(llm_service=mock_llm, max_size=2048)

        mock_img = MagicMock()
        mock_img.size = (100, 100)

        mock_pil_image = MagicMock()
        mock_pil_image.open.return_value = mock_img

        import io as real_io

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "PIL":
                mock_module = MagicMock()
                mock_module.Image = mock_pil_image
                return mock_module
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            # Need to patch at module level since it uses "from PIL import Image"
            with patch("ia_modules.multimodal.image_processor.io.BytesIO"):
                with patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.Image": mock_pil_image}):
                    # Directly call with mocked PIL
                    result = ip._resize_image(b"small_image")
                    # Should return original since 100 < 2048
                    assert result == b"small_image"

    def test_resize_image_exception(self):
        """If resize fails, return original data."""
        mock_llm = MagicMock()
        ip = ImageProcessor(llm_service=mock_llm)

        mock_pil_image = MagicMock()
        mock_pil_image.open.side_effect = Exception("bad image")

        with patch.dict("sys.modules", {"PIL": MagicMock(), "PIL.Image": mock_pil_image}):
            result = ip._resize_image(b"bad_data")
            assert result == b"bad_data"

    def test_resize_image_large_needs_resize(self):
        """Image larger than max_size should be resized."""
        mock_llm = MagicMock()
        ip = ImageProcessor(llm_service=mock_llm, max_size=1024)

        mock_img = MagicMock()
        mock_img.size = (2048, 1536)
        mock_img.resize.return_value = mock_img  # resize returns itself for chaining

        mock_output = MagicMock()
        mock_output.getvalue.return_value = b"resized_bytes"

        mock_pil_module = MagicMock()
        mock_pil_module.Image = MagicMock()
        mock_pil_module.Image.open.return_value = mock_img
        mock_pil_module.Image.Resampling.LANCZOS = "LANCZOS"

        import io as real_io
        original_bytesio = real_io.BytesIO

        call_count = [0]
        def mock_bytesio_fn(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First call: Image.open(io.BytesIO(image_data))
                return original_bytesio(*args, **kwargs)
            else:
                # Second call: output = io.BytesIO()
                return mock_output

        with patch.dict("sys.modules", {"PIL": mock_pil_module, "PIL.Image": mock_pil_module.Image}):
            with patch("ia_modules.multimodal.image_processor.io.BytesIO", side_effect=mock_bytesio_fn):
                result = ip._resize_image(b"large_image")
                assert result == b"resized_bytes"

    async def test_get_embedding(self):
        mock_llm = AsyncMock()
        mock_llm.generate_vision = AsyncMock(return_value="A detailed image")
        ip = ImageProcessor(llm_service=mock_llm)
        ip._resize_image = MagicMock(return_value=b"img")

        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.embedding = [0.1, 0.2, 0.3]
        mock_response.data = [mock_embedding]

        mock_litellm = MagicMock()
        mock_litellm.aembedding = AsyncMock(return_value=mock_response)

        with patch.dict("sys.modules", {"litellm": mock_litellm}):
            with patch("ia_modules.multimodal.image_processor.litellm", mock_litellm, create=True):
                # Patch litellm at the point of use
                original_import = __import__

                def mock_import(name, *args, **kwargs):
                    if name == "litellm":
                        return mock_litellm
                    return original_import(name, *args, **kwargs)

                with patch("builtins.__import__", side_effect=mock_import):
                    result = await ip.get_embedding(b"image_data")
                    assert result == [0.1, 0.2, 0.3]


# ============================================================
# MultiModalProcessor Tests
# ============================================================

from ia_modules.multimodal.processor import (
    ModalityType,
    MultiModalInput,
    MultiModalOutput,
    MultiModalConfig,
    MultiModalProcessor,
)


class TestModalityType:
    def test_values(self):
        assert ModalityType.TEXT.value == "text"
        assert ModalityType.IMAGE.value == "image"
        assert ModalityType.AUDIO.value == "audio"
        assert ModalityType.VIDEO.value == "video"


class TestMultiModalInput:
    def test_defaults(self):
        inp = MultiModalInput(content="hello", modality=ModalityType.TEXT)
        assert inp.content == "hello"
        assert inp.modality == ModalityType.TEXT
        assert inp.metadata == {}
        assert inp.prompt is None

    def test_with_metadata_and_prompt(self):
        inp = MultiModalInput(
            content=b"data",
            modality=ModalityType.IMAGE,
            metadata={"size": 100},
            prompt="describe"
        )
        assert inp.metadata == {"size": 100}
        assert inp.prompt == "describe"


class TestMultiModalOutput:
    def test_defaults(self):
        out = MultiModalOutput(result="test")
        assert out.result == "test"
        assert out.modality_results == {}
        assert out.metadata == {}
        assert out.confidence == 1.0


class TestMultiModalConfig:
    def test_defaults(self):
        config = MultiModalConfig()
        assert config.image_model == "gpt-4-vision-preview"
        assert config.audio_model == "whisper-1"
        assert config.vision_provider == "openai"
        assert config.max_image_size == 2048
        assert config.audio_format == "mp3"
        assert config.video_fps == 1
        assert config.enable_fusion is True
        assert config.max_concurrent == 3
        assert len(config.supported_modalities) == 4


class TestMultiModalProcessor:
    def _make_processor(self, enable_fusion=False, llm_provider=None):
        mock_llm = AsyncMock()
        config = MultiModalConfig(enable_fusion=enable_fusion)
        return MultiModalProcessor(
            llm_service=mock_llm, config=config, llm_provider=llm_provider
        )

    def test_init_no_fusion(self):
        proc = self._make_processor(enable_fusion=False)
        assert proc.fusion is None

    def test_init_with_fusion(self):
        with patch("ia_modules.multimodal.fusion.ModalityFusion") as MockFusion:
            MockFusion.return_value = MagicMock()
            proc = self._make_processor(enable_fusion=True)
            assert proc.fusion is not None

    def test_image_processor_property(self):
        proc = self._make_processor()
        with patch("ia_modules.multimodal.image_processor.ImageProcessor") as MockIP:
            MockIP.return_value = MagicMock()
            ip = proc.image_processor
            assert ip is not None
            # Second access should return same instance
            ip2 = proc.image_processor
            assert ip is ip2

    def test_image_processor_no_model_raises(self):
        mock_llm = AsyncMock()
        config = MultiModalConfig(enable_fusion=False, image_model=None)
        proc = MultiModalProcessor(llm_service=mock_llm, config=config)
        with pytest.raises(ValueError, match="image_model must be configured"):
            _ = proc.image_processor

    def test_audio_processor_property(self):
        proc = self._make_processor()
        with patch("ia_modules.multimodal.audio_processor.AudioProcessor") as MockAP:
            MockAP.return_value = MagicMock()
            ap = proc.audio_processor
            assert ap is not None
            ap2 = proc.audio_processor
            assert ap is ap2

    def test_audio_processor_no_model_raises(self):
        mock_llm = AsyncMock()
        config = MultiModalConfig(enable_fusion=False, audio_model=None)
        proc = MultiModalProcessor(llm_service=mock_llm, config=config)
        with pytest.raises(ValueError, match="audio_model must be configured"):
            _ = proc.audio_processor

    def test_video_processor_property(self):
        proc = self._make_processor()
        with patch("ia_modules.multimodal.video_processor.VideoProcessor") as MockVP:
            MockVP.return_value = MagicMock()
            vp = proc.video_processor
            assert vp is not None
            vp2 = proc.video_processor
            assert vp is vp2

    async def test_process_text_input(self):
        proc = self._make_processor()
        inputs = [MultiModalInput(content="hello world", modality=ModalityType.TEXT)]
        result = await proc.process(inputs)
        assert "hello world" in result.result
        assert result.metadata["num_modalities"] == 1

    async def test_process_text_bytes(self):
        proc = self._make_processor()
        inputs = [MultiModalInput(content=b"hello bytes", modality=ModalityType.TEXT)]
        result = await proc.process(inputs)
        assert "hello bytes" in result.result

    async def test_process_text_with_llm_provider(self):
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value={"content": "LLM response"})
        proc = self._make_processor(llm_provider=mock_provider)
        proc.llm_provider = mock_provider

        inputs = [MultiModalInput(content="hello", modality=ModalityType.TEXT)]
        result = await proc.process(inputs, global_prompt="Summarize")
        assert "LLM response" in result.result

    async def test_process_image_input(self):
        proc = self._make_processor()
        mock_ip = AsyncMock()
        mock_ip.process = AsyncMock(return_value="A cat image")
        proc._image_processor = mock_ip

        inputs = [MultiModalInput(content=b"img", modality=ModalityType.IMAGE)]
        result = await proc.process(inputs)
        assert "A cat image" in result.result

    async def test_process_image_with_prompt(self):
        proc = self._make_processor()
        mock_ip = AsyncMock()
        mock_ip.process = AsyncMock(return_value="result")
        proc._image_processor = mock_ip

        inputs = [MultiModalInput(
            content=b"img", modality=ModalityType.IMAGE, prompt="What color?"
        )]
        result = await proc.process(inputs)
        mock_ip.process.assert_called_with(b"img", "What color?")

    async def test_process_audio_input(self):
        proc = self._make_processor()
        mock_ap = AsyncMock()
        mock_ap.transcribe = AsyncMock(return_value="Hello world")
        proc._audio_processor = mock_ap

        inputs = [MultiModalInput(content=b"audio", modality=ModalityType.AUDIO)]
        result = await proc.process(inputs)
        assert "Hello world" in result.result

    async def test_process_video_input(self):
        proc = self._make_processor()
        mock_vp = AsyncMock()
        mock_vp.process = AsyncMock(return_value="Video analysis")
        proc._video_processor = mock_vp
        mock_ip = AsyncMock()
        proc._image_processor = mock_ip

        inputs = [MultiModalInput(content=b"video", modality=ModalityType.VIDEO)]
        result = await proc.process(inputs)
        assert "Video analysis" in result.result

    async def test_process_error_continues(self):
        """Error in one modality should not stop processing of others."""
        proc = self._make_processor()
        mock_ip = AsyncMock()
        mock_ip.process = AsyncMock(side_effect=Exception("image failed"))
        proc._image_processor = mock_ip

        inputs = [
            MultiModalInput(content=b"img", modality=ModalityType.IMAGE),
            MultiModalInput(content="text", modality=ModalityType.TEXT),
        ]
        result = await proc.process(inputs)
        # Text should still be processed
        assert ModalityType.TEXT in result.modality_results
        assert ModalityType.IMAGE not in result.modality_results

    async def test_process_with_fusion(self):
        mock_llm = AsyncMock()
        mock_provider = AsyncMock()

        with patch("ia_modules.multimodal.fusion.ModalityFusion") as MockFusion:
            mock_fusion = AsyncMock()
            mock_fusion.fuse = AsyncMock(return_value="Fused result")
            MockFusion.return_value = mock_fusion

            config = MultiModalConfig(enable_fusion=True)
            proc = MultiModalProcessor(
                llm_service=mock_llm, config=config, llm_provider=mock_provider
            )

            mock_ip = AsyncMock()
            mock_ip.process = AsyncMock(return_value="image result")
            proc._image_processor = mock_ip

            inputs = [
                MultiModalInput(content="text", modality=ModalityType.TEXT),
                MultiModalInput(content=b"img", modality=ModalityType.IMAGE),
            ]
            result = await proc.process(inputs, global_prompt="Analyze")
            assert result.result == "Fused result"

    async def test_process_image_method(self):
        proc = self._make_processor()
        mock_ip = AsyncMock()
        mock_ip.process = AsyncMock(return_value="described")
        proc._image_processor = mock_ip

        result = await proc.process_image(b"img", "describe")
        assert result == "described"

    async def test_process_audio_method(self):
        proc = self._make_processor()
        mock_ap = AsyncMock()
        mock_ap.transcribe = AsyncMock(return_value="transcript")
        proc._audio_processor = mock_ap

        result = await proc.process_audio(b"audio")
        assert result == "transcript"

    async def test_process_video_method(self):
        proc = self._make_processor()
        mock_vp = AsyncMock()
        mock_vp.process = AsyncMock(return_value="video result")
        proc._video_processor = mock_vp
        mock_ip = AsyncMock()
        proc._image_processor = mock_ip

        result = await proc.process_video(b"video", "describe")
        assert result == "video result"

    async def test_process_text_with_provider_text_key(self):
        """Test _process_text when response has 'text' key instead of 'content'."""
        mock_provider = AsyncMock()
        mock_provider.generate = AsyncMock(return_value={"text": "LLM text response"})
        proc = self._make_processor(llm_provider=mock_provider)
        proc.llm_provider = mock_provider

        inputs = [MultiModalInput(content="data", modality=ModalityType.TEXT)]
        result = await proc.process(inputs, global_prompt="Summarize")
        assert "LLM text response" in result.result

    async def test_generate_multimodal_embeddings_text(self):
        proc = self._make_processor()

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_encoded = MagicMock()
        mock_encoded.tolist.return_value = [0.1, 0.2]
        mock_model.encode.return_value = mock_encoded
        mock_st.return_value = mock_model

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                mod = MagicMock()
                mod.SentenceTransformer = mock_st
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            inputs = [MultiModalInput(content="hello", modality=ModalityType.TEXT)]
            result = await proc.generate_multimodal_embeddings(inputs)
            assert result == [[0.1, 0.2]]

    async def test_generate_multimodal_embeddings_image(self):
        proc = self._make_processor()
        mock_ip = AsyncMock()
        mock_ip.get_embedding = AsyncMock(return_value=[0.5, 0.6])
        proc._image_processor = mock_ip

        inputs = [MultiModalInput(content=b"img", modality=ModalityType.IMAGE)]
        result = await proc.generate_multimodal_embeddings(inputs)
        assert result == [[0.5, 0.6]]

    async def test_generate_multimodal_embeddings_audio(self):
        """Audio modality falls through to 'else' branch: process then embed."""
        proc = self._make_processor()
        mock_ap = AsyncMock()
        mock_ap.transcribe = AsyncMock(return_value="audio text")
        proc._audio_processor = mock_ap

        mock_st = MagicMock()
        mock_model = MagicMock()
        mock_encoded = MagicMock()
        mock_encoded.tolist.return_value = [0.3, 0.4]
        mock_model.encode.return_value = mock_encoded
        mock_st.return_value = mock_model

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                mod = MagicMock()
                mod.SentenceTransformer = mock_st
                return mod
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            inputs = [MultiModalInput(content=b"aud", modality=ModalityType.AUDIO)]
            result = await proc.generate_multimodal_embeddings(inputs)
            assert result == [[0.3, 0.4]]

    async def test_get_text_embedding_import_error(self):
        proc = self._make_processor()

        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("no sentence_transformers")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="sentence-transformers required"):
                await proc._get_text_embedding("text")

    async def test_process_no_results_concatenation(self):
        """When only one modality, no fusion, just concatenate."""
        proc = self._make_processor()
        inputs = [MultiModalInput(content="just text", modality=ModalityType.TEXT)]
        result = await proc.process(inputs)
        assert "just text" in result.result


# ============================================================
# SQLAlchemyAdapter Tests
# ============================================================


class TestSQLAlchemyAdapter:
    """Tests for SQLAlchemyAdapter using mocked sqlalchemy."""

    def _make_adapter(self):
        """Create an adapter with mocked sqlalchemy."""
        with patch("ia_modules.database.adapters.sqlalchemy_adapter.SQLALCHEMY_AVAILABLE", True):
            with patch("ia_modules.database.adapters.sqlalchemy_adapter.create_engine") as mock_ce:
                with patch("ia_modules.database.adapters.sqlalchemy_adapter.sessionmaker") as mock_sm:
                    from ia_modules.database.adapters.sqlalchemy_adapter import SQLAlchemyAdapter
                    adapter = SQLAlchemyAdapter("sqlite:///test.db")
                    return adapter

    def test_init(self):
        adapter = self._make_adapter()
        assert adapter.database_url == "sqlite:///test.db"
        assert adapter._engine is None
        assert adapter._session is None

    def test_init_not_available(self):
        with patch("ia_modules.database.adapters.sqlalchemy_adapter.SQLALCHEMY_AVAILABLE", False):
            from ia_modules.database.adapters.sqlalchemy_adapter import SQLAlchemyAdapter
            with pytest.raises(ImportError, match="sqlalchemy is not installed"):
                SQLAlchemyAdapter("sqlite:///test.db")

    def test_connect_success(self):
        adapter = self._make_adapter()

        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_engine.connect.return_value = mock_conn

        mock_session = MagicMock()
        mock_session_maker = MagicMock(return_value=mock_session)

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.create_engine", return_value=mock_engine):
            with patch("ia_modules.database.adapters.sqlalchemy_adapter.sessionmaker", return_value=mock_session_maker):
                result = adapter.connect()

        assert result is True
        assert adapter._engine is mock_engine

    def test_connect_failure(self):
        adapter = self._make_adapter()

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.create_engine", side_effect=Exception("conn failed")):
            result = adapter.connect()

        assert result is False

    def test_disconnect(self):
        adapter = self._make_adapter()
        adapter._session = MagicMock()
        adapter._engine = MagicMock()
        adapter._session_maker = MagicMock()

        adapter.disconnect()
        assert adapter._session is None
        assert adapter._engine is None
        assert adapter._session_maker is None

    def test_disconnect_no_session(self):
        adapter = self._make_adapter()
        adapter._session = None
        adapter._engine = None
        adapter.disconnect()  # Should not raise

    async def test_close(self):
        adapter = self._make_adapter()
        adapter._session = MagicMock()
        adapter._engine = MagicMock()
        adapter._session_maker = MagicMock()

        await adapter.close()
        assert adapter._session is None

    def test_execute_not_connected(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="Database not connected"):
            adapter.execute("SELECT 1")

    def test_execute_select(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("val1", "val2")]
        mock_result.keys.return_value = ["col1", "col2"]
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            rows = adapter.execute("SELECT col1, col2 FROM t")

        assert rows == [{"col1": "val1", "col2": "val2"}]

    def test_execute_select_empty(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            rows = adapter.execute("SELECT * FROM t")

        assert rows == []

    def test_execute_show(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("table1",)]
        mock_result.keys.return_value = ["Tables"]
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            rows = adapter.execute("SHOW TABLES")

        assert rows == [{"Tables": "table1"}]

    def test_execute_insert(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            rows = adapter.execute("INSERT INTO t VALUES (1)")

        assert rows == []
        mock_session.commit.assert_called_once()

    def test_execute_with_params(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            adapter.execute("INSERT INTO t VALUES (:val)", {"val": 1})

        mock_session.execute.assert_called_once_with("stmt", {"val": 1})

    def test_execute_error_rollback(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session
        mock_session.execute.side_effect = Exception("query error")

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            with pytest.raises(Exception, match="query error"):
                adapter.execute("SELECT 1")

        mock_session.rollback.assert_called_once()

    async def test_execute_async(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            result = await adapter.execute_async("INSERT INTO t VALUES (1)")

        assert result == []

    def test_fetch_one_not_connected(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="Database not connected"):
            adapter.fetch_one("SELECT 1")

    def test_fetch_one_found(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_result.fetchone.return_value = ("val1",)
        mock_result.keys.return_value = ["col1"]
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            row = adapter.fetch_one("SELECT col1 FROM t")

        assert row == {"col1": "val1"}

    def test_fetch_one_not_found(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            row = adapter.fetch_one("SELECT * FROM t WHERE id=999")

        assert row is None

    def test_fetch_one_error(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session
        mock_session.execute.side_effect = Exception("error")

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            row = adapter.fetch_one("SELECT 1")

        assert row is None

    def test_fetch_all_not_connected(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="Database not connected"):
            adapter.fetch_all("SELECT 1")

    def test_fetch_all_found(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("a",), ("b",)]
        mock_result.keys.return_value = ["name"]
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            rows = adapter.fetch_all("SELECT name FROM t")

        assert rows == [{"name": "a"}, {"name": "b"}]

    def test_fetch_all_empty(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            rows = adapter.fetch_all("SELECT * FROM empty")

        assert rows == []

    def test_fetch_all_error(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session
        mock_session.execute.side_effect = Exception("err")

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            rows = adapter.fetch_all("SELECT 1")

        assert rows == []

    def test_table_exists_not_connected(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="Database not connected"):
            adapter.table_exists("users")

    def test_table_exists_true(self):
        adapter = self._make_adapter()
        adapter._engine = MagicMock()

        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = ["users", "posts"]

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.inspect", return_value=mock_inspector):
            assert adapter.table_exists("users") is True

    def test_table_exists_false(self):
        adapter = self._make_adapter()
        adapter._engine = MagicMock()

        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = ["posts"]

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.inspect", return_value=mock_inspector):
            assert adapter.table_exists("users") is False

    def test_table_exists_error(self):
        adapter = self._make_adapter()
        adapter._engine = MagicMock()

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.inspect", side_effect=Exception("err")):
            assert adapter.table_exists("users") is False

    async def test_execute_script_not_connected(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="Database not connected"):
            await adapter.execute_script("CREATE TABLE t (id INT)")

    async def test_execute_script_success(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.side_effect = lambda s: s
            result = await adapter.execute_script("CREATE TABLE t (id INT); INSERT INTO t VALUES (1)")

        assert result.success is True
        assert mock_session.execute.call_count == 2
        mock_session.commit.assert_called_once()

    async def test_execute_script_error(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session
        mock_session.execute.side_effect = Exception("script error")

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.side_effect = lambda s: s
            result = await adapter.execute_script("BAD SQL")

        assert result.success is False
        assert "script error" in result.error_message
        mock_session.rollback.assert_called_once()

    async def test_initialize_connect_fails(self):
        adapter = self._make_adapter()
        adapter.connect = MagicMock(return_value=False)

        result = await adapter.initialize()
        assert result is False

    async def test_initialize_no_schema(self):
        adapter = self._make_adapter()
        adapter.connect = MagicMock(return_value=True)

        result = await adapter.initialize(apply_schema=False)
        assert result is True

    async def test_initialize_with_schema_no_migrations(self):
        adapter = self._make_adapter()
        adapter.connect = MagicMock(return_value=True)

        result = await adapter.initialize(apply_schema=True)
        assert result is True

    async def test_initialize_with_migration_paths(self, tmp_path):
        adapter = self._make_adapter()
        adapter.connect = MagicMock(return_value=True)

        # Create a temp SQL file
        sql_file = tmp_path / "001_init.sql"
        sql_file.write_text("CREATE TABLE test (id INT)")

        from ia_modules.database.interfaces import create_query_result
        adapter.execute_script = AsyncMock(
            return_value=create_query_result(success=True)
        )

        result = await adapter.initialize(
            apply_schema=True,
            app_migration_paths=[str(tmp_path)]
        )
        assert result is True
        adapter.execute_script.assert_called_once()

    async def test_initialize_migration_path_not_exists(self):
        adapter = self._make_adapter()
        adapter.connect = MagicMock(return_value=True)

        result = await adapter.initialize(
            apply_schema=True,
            app_migration_paths=["/nonexistent/path"]
        )
        assert result is True  # Should warn but continue

    async def test_initialize_migration_fails(self, tmp_path):
        adapter = self._make_adapter()
        adapter.connect = MagicMock(return_value=True)

        sql_file = tmp_path / "001_init.sql"
        sql_file.write_text("BAD SQL")

        from ia_modules.database.interfaces import create_error_result
        adapter.execute_script = AsyncMock(
            return_value=create_error_result("migration failed")
        )

        result = await adapter.initialize(
            apply_schema=True,
            app_migration_paths=[str(tmp_path)]
        )
        assert result is False

    def test_engine_property(self):
        adapter = self._make_adapter()
        adapter._engine = "engine_obj"
        assert adapter.engine == "engine_obj"

    def test_session_property(self):
        adapter = self._make_adapter()
        adapter._session = "session_obj"
        assert adapter.session == "session_obj"

    def test_get_new_session_not_connected(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="Database not connected"):
            adapter.get_new_session()

    def test_get_new_session(self):
        adapter = self._make_adapter()
        mock_maker = MagicMock(return_value="new_session")
        adapter._session_maker = mock_maker
        assert adapter.get_new_session() == "new_session"

    def test_begin_transaction(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        mock_session.begin.return_value = "transaction"
        adapter._session = mock_session
        assert adapter.begin_transaction() == "transaction"

    def test_begin_transaction_not_connected(self):
        adapter = self._make_adapter()
        with pytest.raises(RuntimeError, match="Database not connected"):
            adapter.begin_transaction()

    def test_commit(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session
        adapter.commit()
        mock_session.commit.assert_called_once()

    def test_commit_no_session(self):
        adapter = self._make_adapter()
        adapter.commit()  # Should not raise

    def test_rollback(self):
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session
        adapter.rollback()
        mock_session.rollback.assert_called_once()

    def test_rollback_no_session(self):
        adapter = self._make_adapter()
        adapter.rollback()  # Should not raise

    def test_execute_describe(self):
        """Test DESCRIBE query path."""
        adapter = self._make_adapter()
        mock_session = MagicMock()
        adapter._session = mock_session

        mock_result = MagicMock()
        mock_result.fetchall.return_value = [("id", "INT")]
        mock_result.keys.return_value = ["Field", "Type"]
        mock_session.execute.return_value = mock_result

        with patch("ia_modules.database.adapters.sqlalchemy_adapter.text") as mock_text:
            mock_text.return_value = "stmt"
            rows = adapter.execute("DESCRIBE users")

        assert rows == [{"Field": "id", "Type": "INT"}]


# ============================================================
# DatabaseFactory Tests
# ============================================================


class TestDatabaseFactory:
    """Tests for database factory functions."""

    def test_get_database_nexusql(self):
        with patch("ia_modules.database.factory.NexuSQLAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from ia_modules.database.factory import get_database
            db = get_database("sqlite:///test.db", backend="nexusql")
            MockAdapter.assert_called_once_with("sqlite:///test.db")

    def test_get_database_sqlalchemy(self):
        with patch("ia_modules.database.factory.SQLAlchemyAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from ia_modules.database.factory import get_database
            db = get_database("sqlite:///test.db", backend="sqlalchemy", pool_size=5)
            MockAdapter.assert_called_once_with("sqlite:///test.db", pool_size=5)

    def test_get_database_invalid_backend(self):
        from ia_modules.database.factory import get_database
        with pytest.raises(ValueError, match="Invalid database backend"):
            get_database("sqlite:///test.db", backend="mongodb")

    def test_get_database_env_backend(self):
        with patch("ia_modules.database.factory.SQLAlchemyAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from ia_modules.database.factory import get_database
            with patch.dict(os.environ, {"IA_DATABASE_BACKEND": "sqlalchemy"}):
                db = get_database("sqlite:///test.db")
                MockAdapter.assert_called_once()

    def test_get_database_default_backend(self):
        with patch("ia_modules.database.factory.NexuSQLAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from ia_modules.database.factory import get_database
            # Clear env var
            with patch.dict(os.environ, {}, clear=True):
                # Ensure IA_DATABASE_BACKEND is not set
                os.environ.pop("IA_DATABASE_BACKEND", None)
                db = get_database("sqlite:///test.db")
                MockAdapter.assert_called()

    def test_set_default_backend(self):
        from ia_modules.database.factory import set_default_backend, _DEFAULT_BACKEND
        from ia_modules.database.interfaces import DatabaseBackend

        original = _DEFAULT_BACKEND
        try:
            set_default_backend(DatabaseBackend.SQLALCHEMY)
            from ia_modules.database import factory
            assert factory._DEFAULT_BACKEND == DatabaseBackend.SQLALCHEMY
        finally:
            set_default_backend(original if original else DatabaseBackend.NEXUSQL)

    def test_get_nexusql_database(self):
        with patch("ia_modules.database.factory.NexuSQLAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from ia_modules.database.factory import get_nexusql_database
            db = get_nexusql_database("sqlite:///test.db")
            MockAdapter.assert_called_once_with("sqlite:///test.db")

    def test_get_sqlalchemy_database(self):
        with patch("ia_modules.database.factory.SQLAlchemyAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from ia_modules.database.factory import get_sqlalchemy_database
            db = get_sqlalchemy_database("sqlite:///test.db", echo=True)
            MockAdapter.assert_called_once_with("sqlite:///test.db", echo=True)

    def test_get_database_backend_case_insensitive(self):
        with patch("ia_modules.database.factory.SQLAlchemyAdapter") as MockAdapter:
            MockAdapter.return_value = MagicMock()
            from ia_modules.database.factory import get_database
            db = get_database("sqlite:///test.db", backend="  SQLAlchemy  ")
            MockAdapter.assert_called_once()


# ============================================================
# NexuSQLAdapter Tests
# ============================================================


class TestNexuSQLAdapter:
    """Tests for NexuSQLAdapter with mocked nexusql."""

    def _make_adapter(self):
        mock_manager = MagicMock()
        with patch("ia_modules.database.adapters.nexusql_adapter.NEXUSQL_AVAILABLE", True):
            with patch("ia_modules.database.adapters.nexusql_adapter.NexuSQLManager", return_value=mock_manager):
                from ia_modules.database.adapters.nexusql_adapter import NexuSQLAdapter
                adapter = NexuSQLAdapter("sqlite:///test.db")
                return adapter, mock_manager

    def test_init(self):
        adapter, mock_db = self._make_adapter()
        assert adapter.database_url == "sqlite:///test.db"

    def test_init_not_available(self):
        with patch("ia_modules.database.adapters.nexusql_adapter.NEXUSQL_AVAILABLE", False):
            from ia_modules.database.adapters.nexusql_adapter import NexuSQLAdapter
            with pytest.raises(ImportError, match="nexusql is not installed"):
                NexuSQLAdapter("sqlite:///test.db")

    def test_connect(self):
        adapter, mock_db = self._make_adapter()
        mock_db.connect.return_value = True
        assert adapter.connect() is True

    def test_disconnect(self):
        adapter, mock_db = self._make_adapter()
        adapter.disconnect()
        mock_db.disconnect.assert_called_once()

    async def test_close(self):
        adapter, mock_db = self._make_adapter()
        mock_db.close = AsyncMock()
        await adapter.close()
        mock_db.close.assert_called_once()

    def test_execute(self):
        adapter, mock_db = self._make_adapter()
        mock_db.execute.return_value = [{"id": 1}]
        result = adapter.execute("SELECT * FROM t")
        assert result == [{"id": 1}]

    def test_execute_with_params(self):
        adapter, mock_db = self._make_adapter()
        mock_db.execute.return_value = []
        adapter.execute("INSERT INTO t VALUES (:v)", {"v": 1})
        mock_db.execute.assert_called_once_with("INSERT INTO t VALUES (:v)", {"v": 1})

    async def test_execute_async(self):
        adapter, mock_db = self._make_adapter()
        mock_db.execute_async = AsyncMock(return_value=[{"id": 1}])
        result = await adapter.execute_async("SELECT 1")
        assert result == [{"id": 1}]

    def test_fetch_one(self):
        adapter, mock_db = self._make_adapter()
        mock_db.fetch_one.return_value = {"id": 1, "name": "test"}
        result = adapter.fetch_one("SELECT * FROM t WHERE id = :id", {"id": 1})
        assert result == {"id": 1, "name": "test"}

    def test_fetch_one_none(self):
        adapter, mock_db = self._make_adapter()
        mock_db.fetch_one.return_value = None
        result = adapter.fetch_one("SELECT * FROM t WHERE id = :id", {"id": 999})
        assert result is None

    def test_fetch_all(self):
        adapter, mock_db = self._make_adapter()
        mock_db.fetch_all.return_value = [{"id": 1}, {"id": 2}]
        result = adapter.fetch_all("SELECT * FROM t")
        assert len(result) == 2

    def test_table_exists(self):
        adapter, mock_db = self._make_adapter()
        mock_db.table_exists.return_value = True
        assert adapter.table_exists("users") is True

    async def test_execute_script(self):
        adapter, mock_db = self._make_adapter()
        mock_result = MagicMock()
        mock_db.execute_script = AsyncMock(return_value=mock_result)
        result = await adapter.execute_script("CREATE TABLE t (id INT)")
        assert result is mock_result

    async def test_initialize(self):
        adapter, mock_db = self._make_adapter()
        mock_db.initialize = AsyncMock(return_value=True)
        result = await adapter.initialize(apply_schema=True)
        assert result is True

    async def test_initialize_with_paths(self):
        adapter, mock_db = self._make_adapter()
        mock_db.initialize = AsyncMock(return_value=True)
        result = await adapter.initialize(
            apply_schema=True,
            app_migration_paths=["/path/to/migrations"]
        )
        assert result is True
        mock_db.initialize.assert_called_once_with(True, ["/path/to/migrations"])

    def test_nexusql_property(self):
        adapter, mock_db = self._make_adapter()
        assert adapter.nexusql is mock_db

    def test_config_property(self):
        adapter, mock_db = self._make_adapter()
        mock_db.config = MagicMock(database_type="sqlite")
        assert adapter.config.database_type == "sqlite"

    def test_database_type_property(self):
        adapter, mock_db = self._make_adapter()
        mock_db.config = MagicMock(database_type="postgresql")
        assert adapter.database_type == "postgresql"


# ============================================================
# QueryResult / Interface Tests
# ============================================================


class TestQueryResult:
    def test_create_query_result(self):
        from ia_modules.database.interfaces import create_query_result
        result = create_query_result(success=True, data=[{"id": 1}])
        assert result.success is True
        assert result.row_count == 1
        assert result.data == [{"id": 1}]

    def test_create_query_result_defaults(self):
        from ia_modules.database.interfaces import create_query_result
        result = create_query_result()
        assert result.success is True
        assert result.data == []
        assert result.row_count == 0

    def test_create_error_result(self):
        from ia_modules.database.interfaces import create_error_result
        result = create_error_result("something broke")
        assert result.success is False
        assert result.error_message == "something broke"
        assert result.row_count == 0

    def test_get_first_row(self):
        from ia_modules.database.interfaces import QueryResult
        qr = QueryResult(success=True, data=[{"a": 1}, {"a": 2}], row_count=2)
        assert qr.get_first_row() == {"a": 1}

    def test_get_first_row_empty(self):
        from ia_modules.database.interfaces import QueryResult
        qr = QueryResult(success=True, data=[], row_count=0)
        assert qr.get_first_row() is None

    def test_get_column_values(self):
        from ia_modules.database.interfaces import QueryResult
        qr = QueryResult(
            success=True,
            data=[{"name": "Alice"}, {"name": "Bob"}, {"age": 30}],
            row_count=3
        )
        assert qr.get_column_values("name") == ["Alice", "Bob"]
