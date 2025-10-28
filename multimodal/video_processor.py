"""Video processing by frame extraction and analysis."""

from typing import Union, Optional, List
import logging

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Process videos by extracting and analyzing frames."""

    def __init__(self, fps: int = 1):
        """
        Initialize video processor.

        Args:
            fps: Frames per second to extract
        """
        self.fps = fps

    async def process(
        self,
        video: Union[bytes, str],
        prompt: Optional[str],
        image_processor
    ) -> str:
        """
        Process video by analyzing frames.

        Args:
            video: Video bytes or file path
            prompt: Optional prompt for frame analysis
            image_processor: ImageProcessor instance

        Returns:
            Video analysis

        Raises:
            ValueError: If no frames could be extracted
            Exception: If video processing fails
        """
        # Extract frames
        frames = await self.extract_frames(video)

        if not frames:
            raise ValueError(
                "No frames could be extracted from video. "
                "Check that the video file is valid and OpenCV is installed."
            )

        # Analyze each frame
        frame_analyses = []
        for i, frame in enumerate(frames):
            analysis = await image_processor.process(
                frame,
                prompt or f"Describe what's happening in this frame"
            )
            frame_analyses.append(f"Frame {i+1}: {analysis}")

        # Combine analyses
        combined = "\n".join(frame_analyses)
        return f"Video Analysis ({len(frames)} frames):\n{combined}"

    async def extract_frames(
        self,
        video: Union[bytes, str],
        max_frames: int = 10
    ) -> List[bytes]:
        """
        Extract frames from video.

        Args:
            video: Video bytes or file path
            max_frames: Maximum number of frames to extract

        Returns:
            List of frame images as bytes

        Raises:
            ImportError: If OpenCV is not installed
            ValueError: If video cannot be opened
        """
        try:
            import cv2
            import numpy as np
            import tempfile
        except ImportError as e:
            raise ImportError(
                "OpenCV required for video processing. "
                "Install with: pip install opencv-python"
            ) from e

        # Handle bytes input
        if isinstance(video, bytes):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(video)
                video_path = f.name
        else:
            video_path = video

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps = cap.get(cv2.CAP_PROP_FPS)

            # Calculate frame interval
            interval = max(1, int(video_fps / self.fps))
            frames_to_extract = min(max_frames, total_frames // interval)

            frames = []
            frame_count = 0

            while len(frames) < frames_to_extract:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % interval == 0:
                    # Convert frame to JPEG bytes
                    _, buffer = cv2.imencode('.jpg', frame)
                    frames.append(buffer.tobytes())

                frame_count += 1

            return frames
        finally:
            cap.release()

    async def get_video_info(self, video: Union[bytes, str]) -> dict:
        """
        Get video metadata.

        Args:
            video: Video bytes or file path

        Returns:
            Dictionary with video properties (fps, frame_count, width, height, duration)

        Raises:
            ImportError: If OpenCV is not installed
            ValueError: If video cannot be opened
        """
        try:
            import cv2
            import tempfile
        except ImportError as e:
            raise ImportError(
                "OpenCV required for video processing. "
                "Install with: pip install opencv-python"
            ) from e

        if isinstance(video, bytes):
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
                f.write(video)
                video_path = f.name
        else:
            video_path = video

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            info = {
                "fps": fps,
                "frame_count": frame_count,
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "duration": frame_count / fps if fps > 0 else 0
            }

            return info
        finally:
            cap.release()
