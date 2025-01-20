from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2


@dataclass
class GapInfo:
    start_frame: int
    end_frame: int
    known_metrics: Dict[str, np.ndarray]
    surrounding_frames: Dict[str, np.ndarray]
    camera_trajectory: Optional[np.ndarray]


class VideoReconstructor:
    def __init__(self, config: dict):
        self.config = config
        self.metrics_interpolator = MetricsInterpolator()

    def reconstruct_gaps(
            self,
            video_segments: List[Dict],
            gaps: List[GapInfo]
    ) -> Tuple[np.ndarray, Dict]:
        reconstructed_frames = []
        reconstruction_metadata = {}

        for gap in gaps:
            # Interpolate metrics through gap
            interpolated_metrics = self.metrics_interpolator.interpolate(
                gap.known_metrics,
                gap.start_frame,
                gap.end_frame
            )

            # Generate intermediate frames
            frames = self._generate_intermediate_frames(
                gap.surrounding_frames['prev'],
                gap.surrounding_frames['next'],
                gap.end_frame - gap.start_frame
            )

            reconstructed_frames.extend(frames)

            # Calculate confidence scores
            reconstruction_metadata[f"gap_{gap.start_frame}_{gap.end_frame}"] = {
                'confidence': self._calculate_reconstruction_confidence(interpolated_metrics),
                'frame_count': len(frames)
            }

        return np.array(reconstructed_frames), reconstruction_metadata

    def _generate_intermediate_frames(
            self,
            frame1: np.ndarray,
            frame2: np.ndarray,
            num_frames: int
    ) -> List[np.ndarray]:
        """Generate intermediate frames using linear interpolation"""
        frames = []
        for i in range(num_frames):
            alpha = i / (num_frames + 1)
            interpolated = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            frames.append(interpolated)
        return frames

    def _calculate_reconstruction_confidence(
            self,
            interpolated_metrics: Dict[str, np.ndarray]
    ) -> float:
        """Calculate confidence score for reconstruction"""
        # Simple confidence based on metrics stability
        confidence = 1.0
        for metric_values in interpolated_metrics.values():
            confidence *= (1 - np.std(metric_values) / np.mean(metric_values))
        return max(0.0, min(1.0, confidence))


class MetricsInterpolator:
    def interpolate(
            self,
            known_metrics: Dict[str, np.ndarray],
            start_frame: int,
            end_frame: int
    ) -> Dict[str, np.ndarray]:
        interpolated = {}
        num_frames = end_frame - start_frame

        for metric_name, values in known_metrics.items():
            # Linear interpolation between known values
            start_val = values[max(0, start_frame - 1)]
            end_val = values[min(len(values) - 1, end_frame + 1)]

            interpolated[metric_name] = np.linspace(
                start_val,
                end_val,
                num_frames,
                endpoint=False
            )

        return interpolated
