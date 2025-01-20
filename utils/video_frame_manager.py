import cv2
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple
import numpy as np
import logging
import json


class VideoFrameManager:
    def __init__(self, storage_path: str, target_resolution: tuple, compression_level: int):
        self.storage_path = Path(storage_path)
        self.target_resolution = target_resolution
        self.compression_level = compression_level
        self.logger = logging.getLogger(__name__)

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Validate parameters
        if not (0 <= compression_level <= 100):
            raise ValueError("Compression level must be between 0 and 100")

        if not all(x > 0 for x in target_resolution):
            raise ValueError("Resolution dimensions must be positive")

    def process_video(self, video_path: str, sequence_name: str, frame_step: int = 1) -> Dict:
        """
        Process video file and save frames to storage

        Args:
            video_path: Path to video file
            sequence_name: Name for the sequence of frames
            frame_step: Process every nth frame

        Returns:
            Dictionary containing video metadata
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        try:
            metadata = {
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'fps': cap.get(cv2.CAP_PROP_FPS),
                'original_resolution': (
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                'target_resolution': self.target_resolution,
                'frame_step': frame_step,
                'compression_level': self.compression_level
            }

            sequence_dir = self.storage_path / sequence_name
            sequence_dir.mkdir(parents=True, exist_ok=True)

            # Save metadata
            with open(sequence_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)

            frame_idx = 0
            saved_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    if self._is_valid_frame(frame):
                        processed_frame = self._process_frame(frame)
                        frame_path = sequence_dir / f"frame_{frame_idx:06d}.jpg"

                        success = cv2.imwrite(
                            str(frame_path),
                            processed_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, self.compression_level]
                        )

                        if not success:
                            self.logger.warning(f"Failed to save frame {frame_idx}")
                        else:
                            saved_frames += 1
                    else:
                        self.logger.warning(f"Invalid frame detected at index {frame_idx}")

                frame_idx += 1

            metadata['saved_frames'] = saved_frames

            # Update metadata with final count
            with open(sequence_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=4)

            self.logger.info(f"Processed {frame_idx} frames, saved {saved_frames} frames")
            return metadata

        finally:
            cap.release()

    def get_frames(self, sequence_name: str) -> Generator[np.ndarray, None, None]:
        """
        Retrieve frames for a given sequence

        Args:
            sequence_name: Name of the frame sequence

        Yields:
            numpy.ndarray: Frame image data
        """
        sequence_dir = self.storage_path / sequence_name
        if not sequence_dir.exists():
            raise ValueError(f"Sequence directory not found: {sequence_name}")

        for frame_path in sorted(sequence_dir.glob("frame_*.jpg")):
            frame = cv2.imread(str(frame_path))
            if frame is not None:
                yield frame
            else:
                self.logger.warning(f"Failed to read frame: {frame_path}")

    def get_frame_at_index(self, sequence_name: str, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get specific frame by index

        Args:
            sequence_name: Name of the frame sequence
            frame_idx: Index of the frame to retrieve

        Returns:
            numpy.ndarray or None: Frame image data if found
        """
        frame_path = self.storage_path / sequence_name / f"frame_{frame_idx:06d}.jpg"
        if frame_path.exists():
            return cv2.imread(str(frame_path))
        return None

    def get_sequence_metadata(self, sequence_name: str) -> Dict:
        """Get metadata for a sequence"""
        metadata_path = self.storage_path / sequence_name / 'metadata.json'
        if not metadata_path.exists():
            raise ValueError(f"Metadata not found for sequence: {sequence_name}")

        with open(metadata_path, 'r') as f:
            return json.load(f)

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame according to target parameters"""
        if frame.shape[:2][::-1] != self.target_resolution:
            frame = cv2.resize(frame, self.target_resolution)
        return frame

    def _is_valid_frame(self, frame: np.ndarray) -> bool:
        """Check if frame is valid and usable"""
        if frame is None or frame.size == 0:
            return False

        # Check for completely black or white frames
        if np.mean(frame) < 1 or np.mean(frame) > 254:
            return False

        # Check for corrupted dimensions
        if frame.shape[0] == 0 or frame.shape[1] == 0:
            return False

        return True

    def clear_sequence(self, sequence_name: str):
        """Delete all frames and metadata for a sequence"""
        sequence_dir = self.storage_path / sequence_name
        if sequence_dir.exists():
            for file in sequence_dir.glob("*"):
                file.unlink()
            sequence_dir.rmdir()
            self.logger.info(f"Cleared sequence: {sequence_name}")
