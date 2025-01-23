import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from core.scene_detector import SceneDetector, SceneChange
from stabilography.movement_detector import SpeedEstimator
from utils.lane_detector import LaneDetector
from utils.video_frame_manager import VideoFrameManager
from utils.video_quality import VideoQualityAnalyzer
from utils.video_reader import VideoReader
from utils.video_reconstructor import VideoReconstructor, GapInfo


class BasePipeline:
    """Base pipeline class that handles common video processing functionality"""
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Common video processing components
        self.frame_manager = VideoFrameManager(
            storage_path=config['output']['storage_path'],
            target_resolution=(
                config['video_processing']['resize_width'],
                config['video_processing']['resize_height']
            ),
            compression_level=config['output']['compression_quality']
        )
        self.quality_analyzer = VideoQualityAnalyzer(config)

    def _process_video_metadata(self, video_path: str) -> Dict:
        """Common method to process video metadata"""
        video_reader = VideoReader(video_path)
        metadata = video_reader.get_metadata()
        video_reader.release()
        return {
            'frame_count': metadata.total_frames,
            'fps': metadata.fps,
            'resolution': (metadata.width, metadata.height)
        }

    def _get_frame_generator(self, video_path: str):
        """Common method to generate frames"""
        return VideoReader(video_path).read_frames()


class SceneAnalysisPipeline(BasePipeline):
    """Pipeline for basic scene analysis"""
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Scene-specific components
        self.scene_detector = SceneDetector(config)
        self.reconstructor = VideoReconstructor(config)
        self.lane_detector = LaneDetector(config)
        self.speed_estimator = SpeedEstimator(
            fps=config['video_processing']['fps'],
            track_length=config['tracking']['track_length'],
            stability_threshold=config['tracking']['stability_threshold']
        )
        
        self.track_history = []

    def analyze_video(self, video_path: str) -> Dict:
        """Main analysis method that other pipelines can extend"""
        self.logger.info(f"Starting scene analysis of {video_path}")
        
        # Get basic video info
        metadata = self._process_video_metadata(video_path)
        
        # Initialize analysis results
        scene_changes = []
        quality_metrics = []
        gaps_to_reconstruct = []
        lane_detections = []
        speed_measurements = []
        prev_frame = None

        self.speed_estimator.calibrate(self.frame_manager.width)

        # Process frames
        for frame_idx, frame in enumerate(self._get_frame_generator(video_path)):
            # Basic frame processing that other pipelines might need
            frame_result = self._process_frame(frame, frame_idx, prev_frame)
            
            # Scene-specific processing
            scene_result = self._process_scene(frame, frame_idx, prev_frame)
            
            # Combine results
            quality_metrics.append(frame_result['quality_metrics'])
            if scene_result.get('scene_change'):
                scene_changes.append(scene_result['scene_change'])
            
            lane_info = self.lane_detector.detect(frame)
            if lane_info:
                lane_detections.append({
                    'frame_idx': frame_idx,
                    'left_lane': lane_info.left_lane.tolist() if lane_info.left_lane is not None else None,
                    'right_lane': lane_info.right_lane.tolist() if lane_info.right_lane is not None else None,
                    'lane_width': lane_info.lane_width,
                    'curvature': lane_info.curvature
                })

            if self._needs_reconstruction(frame_result['quality_metrics']):
                gaps_to_reconstruct.append(self._create_gap_info(
                    frame_idx, frame, quality_metrics
                ))

            if lane_info and lane_info.center_position:
                self.track_history.append({
                    'frame_idx': frame_idx,
                    'center': lane_info.center_position
                })

                if len(self.track_history) > 10:
                    self.track_history = self.track_history[-10:]

                speed = self.speed_estimator.estimate_speed(
                    self.track_history, frame_idx
                )

                speed_measurements.append({
                    'frame_idx': frame_idx,
                    'speed': speed
                })

            prev_frame = frame.copy()

        metadata = self._process_video_metadata(video_path)

        reconstructed_segments = {}
        if gaps_to_reconstruct:
            video_segments = self._get_video_segments(Path(video_path).stem, scene_changes)
            reconstructed_frames, reconstruction_metadata = self.reconstructor.reconstruct_gaps(
                video_segments=video_segments,
                gaps=gaps_to_reconstruct
            )
            reconstructed_segments = {
                'frames': reconstructed_frames,
                'metadata': reconstruction_metadata
            }

        scene_changes_dict = [{
            'frame_idx': sc.frame_idx,
            'type': sc.change_type.value,
            'confidence': sc.confidence,
            'metrics': sc.metrics
        } for sc in scene_changes]

        analysis_results = {
            'video_metadata': metadata,
            'scene_changes': scene_changes_dict,
            'quality_metrics': quality_metrics,
            'reconstructed_segments': reconstructed_segments,
            'lane_detections': lane_detections,
            'speed_measurements': speed_measurements,
            'analysis_parameters': {
                'hist_threshold': self.config['scene_detection']['hist_threshold'],
                'quality_thresholds': self.config['video_processing']['quality_check'],
                'stability_threshold': self.config['tracking']['stability_threshold']
            }
        }

        self.logger.info("Analysis completed successfully")
        return analysis_results

    def _process_frame(self, frame: np.ndarray, frame_idx: int, prev_frame: Optional[np.ndarray]) -> Dict:
        """Basic frame processing that other pipelines can use"""
        self.frame_manager.save_frame(Path(video_path).stem, frame_idx, frame)
        frame_metrics = self.quality_analyzer.compute_frame_metrics(frame)
        self.quality_analyzer.update_metrics_history(frame_idx, frame_metrics)
        
        return {
            'frame_idx': frame_idx,
            'quality_metrics': frame_metrics,
        }

    def _process_scene(self, frame: np.ndarray, frame_idx: int, prev_frame: Optional[np.ndarray]) -> Dict:
        """Scene-specific processing"""
        result = {}
        
        if prev_frame is not None:
            scene_change = self.scene_detector.detect_scene_change(prev_frame, frame)
            if scene_change:
                result['scene_change'] = scene_change
        
        return result

    def _needs_reconstruction(self, metrics: Dict[str, float]) -> bool:
        thresholds = self.config['video_processing']['quality_check']
        return (
            metrics['brightness'] < thresholds['min_brightness'] or
            metrics['brightness'] > thresholds['max_brightness'] or
            metrics['contrast'] < thresholds['min_contrast'] or
            metrics['blur'] < thresholds['blur_threshold']
        )

    def _create_gap_info(
            self,
            frame_idx: int,
            current_frame: np.ndarray,
            quality_metrics: List[Dict]
    ) -> GapInfo:
        return GapInfo(
            start_frame=frame_idx,
            end_frame=frame_idx + 1,
            known_metrics={
                metric: np.array([values[metric] for values in quality_metrics])
                for metric in quality_metrics[0].keys()
            },
            surrounding_frames={
                'prev': current_frame,
                'next': None
            },
            camera_trajectory=None
        )

    def _get_video_segments(self, sequence_name: str, scene_changes: List[SceneChange]) -> List[Dict]:
        segments = []
        frames = list(self.frame_manager.get_frames(sequence_name))

        if not frames:
            return segments

        current_segment = {
            'start_frame': 0,
            'frames': [],
            'quality_scores': [],
            'scene_type': None
        }

        scene_change_frames = [sc.frame_idx for sc in scene_changes]

        for idx, frame in enumerate(frames):
            metrics = self.quality_analyzer.compute_frame_metrics(frame)
            quality_score = np.mean([
                metrics['brightness'],
                metrics['contrast'],
                metrics['blur']
            ])

            if idx in scene_change_frames:
                scene_info = next(sc for sc in scene_changes if sc.frame_idx == idx)
                current_segment['end_frame'] = idx - 1
                current_segment['avg_quality'] = np.mean(current_segment['quality_scores'])
                segments.append(current_segment)

                current_segment = {
                    'start_frame': idx,
                    'frames': [],
                    'quality_scores': [],
                    'scene_type': scene_info.change_type.value
                }

            current_segment['frames'].append(frame)
            current_segment['quality_scores'].append(quality_score)

        if current_segment['frames']:
            current_segment['end_frame'] = len(frames) - 1
            current_segment['avg_quality'] = np.mean(current_segment['quality_scores'])
            segments.append(current_segment)

        return segments

# Add this at the end of scene_analysis_pipeline.py

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(base_dir, "../public/chisora_joyce_round_03.mp4")

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create output directory
    output_dir = os.path.join(base_dir, "../output")
    os.makedirs(output_dir, exist_ok=True)

    # Complete configuration with all required parameters
    config = {
        'output': {
            'storage_path': os.path.join(output_dir, 'frames'),
            'compression_quality': 95,
            'plots_directory': os.path.join(output_dir, 'plots')  # Added this
        },
        'video_processing': {
            'resize_width': 1280,
            'resize_height': 720,
            'fps': 30,
            'quality_check': {
                'min_brightness': 0.2,
                'max_brightness': 0.8,
                'min_contrast': 0.4,
                'blur_threshold': 100
            }
        },
        'scene_detection': {
            'hist_threshold': 0.5
        },
        'tracking': {
            'track_length': 50,
            'stability_threshold': 0.1
        }
    }

    # Create necessary directories
    os.makedirs(config['output']['storage_path'], exist_ok=True)
    os.makedirs(config['output']['plots_directory'], exist_ok=True)

    # Initialize pipeline
    pipeline = SceneAnalysisPipeline(config)

    # Process video
    try:
        results = pipeline.analyze_video(input_file)

        # Save results
        output_path = os.path.join(output_dir, f"{Path(input_file).stem}_analysis.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"Analysis completed. Results saved to {output_path}")

    except Exception as e:
        logger.error(f"Error during video analysis: {e}")

