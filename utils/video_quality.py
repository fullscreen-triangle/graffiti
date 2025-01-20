import numpy as np
import cv2
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path



class VideoQualityAnalyzer:
    """Analyzes video quality metrics frame by frame"""

    def __init__(self, config: dict):
        self.config = config
        self.metrics_history = {}
        self.output_dir = Path(config['output']['plots_directory'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Add quality thresholds from config
        self.quality_thresholds = config['video_processing']['quality_check']

    def compute_frame_metrics(self, frame: np.ndarray) -> Dict[str, float]:
        """Compute quality metrics for a single frame"""
        metrics = {}

        # Convert to grayscale for some calculations
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Brightness
        metrics['brightness'] = np.mean(gray)

        # Contrast
        metrics['contrast'] = np.std(gray)

        # Blur detection (Laplacian variance)
        metrics['blur'] = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Noise estimation
        metrics['noise'] = self._estimate_noise(frame)

        # Color statistics
        metrics['color_range'] = np.max(frame) - np.min(frame)
        metrics['saturation'] = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1])

        return metrics

    def _estimate_noise(self, frame: np.ndarray) -> float:
        """Estimate image noise using filter-based method"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = np.std(gray - mean)
        return noise

    def update_metrics_history(self, frame_idx: int, metrics: Dict[str, float]):
        """Update metrics history with new frame data"""
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            self.metrics_history[metric_name].append((frame_idx, value))

    def plot_metrics(self, sequence_name: str):
        """Generate plots for all tracked metrics"""
        plt.style.use('seaborn')

        metrics_count = len(self.metrics_history)
        fig, axes = plt.subplots(metrics_count, 1, figsize=(12, 4 * metrics_count))

        if metrics_count == 1:
            axes = [axes]

        for idx, (metric_name, values) in enumerate(self.metrics_history.items()):
            frame_indices, metric_values = zip(*values)

            axes[idx].plot(frame_indices, metric_values, linewidth=1)
            axes[idx].set_title(f'{metric_name.capitalize()} Over Time')
            axes[idx].set_xlabel('Frame Index')
            axes[idx].set_ylabel(metric_name.capitalize())

            # Add threshold lines if defined in config
            if metric_name in self.quality_thresholds:
                threshold = self.quality_thresholds[metric_name]
                axes[idx].axhline(y=threshold, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(self.output_dir / f"{sequence_name}_quality_metrics.png", dpi=300)
        plt.close()
