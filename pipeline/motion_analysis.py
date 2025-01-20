import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from ..core.pose_detector import PoseDetector
from ..core.movement_tracker import MovementTracker
from ..motion.motion_classifier import ActionClassifier
from ..motion.motion_classifier import MotionMetricsCalculator
from ..motion.motion_classifier import PhaseAnalyzer
from ..motion.motion_classifier import PatternMatcher
from ..motion.motion_classifier import SequenceAnalyzer
from ..motion.motion_classifier import SymmetryAnalyzer
from ..motion.motion_classifier import TempoAnalyzer
from ..motion.motion_classifier import TrajectoryAnalyzer


@dataclass
class MotionSegment:
    start_frame: int
    end_frame: int
    action_type: str
    confidence: float
    keypoints: List[np.ndarray]
    metrics: Dict


class MotionAnalysisPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.pose_detector = PoseDetector(
            model_path=config['models']['pose_detection'],
            confidence_threshold=config['detection']['confidence_threshold']
        )

        self.movement_tracker = MovementTracker(
            tracking_threshold=config['tracking']['movement_threshold'],
            window_size=config['tracking']['window_size']
        )

        self.action_classifier = ActionClassifier(
            model_path=config['models']['action_classification'],
            class_mapping=config['classification']['action_classes']
        )

        fps = config['processing'].get('fps', 30)
        self.metrics_calculator = MotionMetricsCalculator(fps=fps)

        self.phase_analyzer = PhaseAnalyzer(
            window_size=config['analysis']['phase_window'],
            overlap=config['analysis']['phase_overlap']
        )

        self.pattern_matcher = PatternMatcher(
            template_path=config['models']['pattern_templates'],
            similarity_threshold=config['analysis']['pattern_similarity']
        )

        self.sequence_analyzer = SequenceAnalyzer(
            min_sequence_length=config['analysis']['min_sequence_length']
        )

        self.symmetry_analyzer = SymmetryAnalyzer(
            reference_points=config['analysis']['symmetry_points']
        )

        self.tempo_analyzer = TempoAnalyzer(fps=fps)

        self.trajectory_analyzer = TrajectoryAnalyzer(
            smoothing_window=config['analysis']['trajectory_smoothing']
        )

        self.max_workers = config['processing']['max_workers']
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

        self.current_segments: List[MotionSegment] = []

    def analyze_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        pose_data = self.pose_detector.detect(frame)
        if pose_data is None:
            return self._create_empty_result(frame_idx)

        movement_data = self.movement_tracker.track(pose_data)
        if not movement_data['is_moving']:
            return self._create_static_result(frame_idx, pose_data)

        futures = []
        futures.append(self.executor.submit(
            self.action_classifier.classify, pose_data
        ))
        futures.append(self.executor.submit(
            self.metrics_calculator.compute_metrics, pose_data
        ))
        futures.append(self.executor.submit(
            self.trajectory_analyzer.analyze, pose_data
        ))

        action_result = futures[0].result()
        metrics = futures[1].result()
        trajectory = futures[2].result()

        self._update_segments(frame_idx, pose_data, action_result, metrics)

        phase_info = self.phase_analyzer.analyze(pose_data)
        pattern_matches = self.pattern_matcher.find_matches(pose_data)
        sequence_info = self.sequence_analyzer.analyze(self.current_segments)
        symmetry_metrics = self.symmetry_analyzer.compute_symmetry(pose_data)
        tempo_info = self.tempo_analyzer.analyze(pose_data)

        return {
            'frame_idx': frame_idx,
            'pose_data': pose_data.tolist(),
            'movement': movement_data,
            'action': action_result,
            'metrics': metrics,
            'phase': phase_info,
            'patterns': pattern_matches,
            'sequence': sequence_info,
            'symmetry': symmetry_metrics,
            'tempo': tempo_info,
            'trajectory': trajectory
        }

    def analyze_video(self, video_path: str) -> Dict:
        self.logger.info(f"Starting motion analysis of {video_path}")

        import cv2
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        all_results = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_results = self.analyze_frame(frame, frame_idx)
            all_results.append(frame_results)
            frame_idx += 1

        cap.release()

        final_segments = self._finalize_segments()

        analysis_summary = {
            'video_info': {
                'path': video_path,
                'frames': frame_count,
                'fps': fps
            },
            'frame_results': all_results,
            'motion_segments': [self._segment_to_dict(seg) for seg in final_segments],
            'analysis_parameters': self._get_analysis_parameters()
        }

        self.logger.info("Motion analysis completed successfully")
        return analysis_summary

    def _create_empty_result(self, frame_idx: int) -> Dict:
        return {
            'frame_idx': frame_idx,
            'pose_data': None,
            'movement': {'is_moving': False},
            'action': None,
            'metrics': {},
            'phase': None,
            'patterns': [],
            'sequence': None,
            'symmetry': None,
            'tempo': None,
            'trajectory': None
        }

    def _create_static_result(self, frame_idx: int, pose_data: np.ndarray) -> Dict:
        return {
            'frame_idx': frame_idx,
            'pose_data': pose_data.tolist(),
            'movement': {'is_moving': False},
            'action': None,
            'metrics': self.metrics_calculator.compute_metrics(pose_data),
            'phase': None,
            'patterns': [],
            'sequence': None,
            'symmetry': self.symmetry_analyzer.compute_symmetry(pose_data),
            'tempo': None,
            'trajectory': None
        }

    def _update_segments(self, frame_idx: int, pose_data: np.ndarray,
                         action_result: Dict, metrics: Dict) -> None:
        if not self.current_segments or (
                action_result['action'] != self.current_segments[-1].action_type and
                action_result['confidence'] > self.config['detection']['action_threshold']
        ):
            if self.current_segments:
                self.current_segments[-1].end_frame = frame_idx - 1

            self.current_segments.append(MotionSegment(
                start_frame=frame_idx,
                end_frame=frame_idx,
                action_type=action_result['action'],
                confidence=action_result['confidence'],
                keypoints=[pose_data],
                metrics={'initial': metrics}
            ))
        else:
            current_segment = self.current_segments[-1]
            current_segment.end_frame = frame_idx
            current_segment.keypoints.append(pose_data)
            current_segment.metrics[f'frame_{frame_idx}'] = metrics

    def _finalize_segments(self) -> List[MotionSegment]:
        for segment in self.current_segments:
            segment.metrics['summary'] = self._compute_segment_summary(segment)
        return self.current_segments

    def _compute_segment_summary(self, segment: MotionSegment) -> Dict:
        return {
            'duration': segment.end_frame - segment.start_frame + 1,
            'average_metrics': {
                key: np.mean([metrics[key] for metrics in segment.metrics.values() if key in metrics])
                for key in segment.metrics['initial'].keys()
            }
        }

    def _segment_to_dict(self, segment: MotionSegment) -> Dict:
        return {
            'start_frame': segment.start_frame,
            'end_frame': segment.end_frame,
            'action_type': segment.action_type,
            'confidence': segment.confidence,
            'metrics': segment.metrics
        }

    def _get_analysis_parameters(self) -> Dict:
        return {
            'confidence_threshold': self.config['detection']['confidence_threshold'],
            'movement_threshold': self.config['tracking']['movement_threshold'],
            'action_threshold': self.config['detection']['action_threshold'],
            'phase_window': self.config['analysis']['phase_window'],
            'pattern_similarity': self.config['analysis']['pattern_similarity'],
            'trajectory_smoothing': self.config['analysis']['trajectory_smoothing']
        }

