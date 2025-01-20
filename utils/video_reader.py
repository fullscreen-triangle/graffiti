import cv2
from typing import Generator, Tuple, Optional
import numpy as np


class VideoReader:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.resolution = (0, 0)

    def __enter__(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.resolution = (
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()

    def read_frames(self, start_frame: int = 0, end_frame: Optional[int] = None) -> Generator[
        Tuple[int, np.ndarray], None, None]:
        """
        Read frames from the video

        Args:
            start_frame: Frame to start reading from
            end_frame: Frame to stop reading at (exclusive)

        Yields:
            Tuple of (frame_index, frame_data)
        """
        if not self.cap:
            raise RuntimeError("VideoReader not initialized. Use with context manager.")

        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_idx = start_frame
        end_frame = end_frame or self.frame_count

        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            yield frame_idx, frame
            frame_idx += 1

    def get_frame(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get a specific frame by index"""
        if not self.cap:
            raise RuntimeError("VideoReader not initialized. Use with context manager.")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()

        return frame if ret else None
