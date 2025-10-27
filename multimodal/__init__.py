"""
Multi-Modal Support for IA Modules.

Enables AI patterns to work with multiple modalities:
- Text
- Images
- Audio
- Video
"""

from .processor import (
    MultiModalProcessor,
    MultiModalConfig,
    MultiModalInput,
    MultiModalOutput,
    ModalityType
)
from .image_processor import ImageProcessor
from .audio_processor import AudioProcessor
from .video_processor import VideoProcessor
from .fusion import ModalityFusion

__all__ = [
    'MultiModalProcessor',
    'MultiModalConfig',
    'MultiModalInput',
    'MultiModalOutput',
    'ModalityType',
    'ImageProcessor',
    'AudioProcessor',
    'VideoProcessor',
    'ModalityFusion',
]
