import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from core.movement_tracker import MovementTracker
from core.pose_detector import PoseDetector
from motion.motion_classifier import ActionClassifier, MotionMetricsCalculator, PhaseAnalyzer, PatternMatcher, \
    SequenceAnalyzer, SymmetryAnalyzer, TempoAnalyzer, TrajectoryAnalyzer
from pipeline.scene_analysis import SceneAnalysisPipeline


@dataclass
class MotionSegment:
    start_frame: int
    end_frame: int
    action_type: str
    confidence: float
    keypoints: List[np.ndarray]
    metrics: Dict


class MotionAnalysisPipeline(SceneAnalysisPipeline):
    """Pipeline for motion analysis that builds upon scene analysis"""
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Motion-specific components
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

    def _process_frame(self, frame: np.ndarray, frame_idx: int, prev_frame: Optional[np.ndarray]) -> Dict:
        """Override frame processing to add motion analysis"""
        # Get base scene analysis results
        base_result = super()._process_frame(frame, frame_idx, prev_frame)
        
        # Add motion analysis
        pose_data = self.pose_detector.detect(frame)
        if pose_data is None:
            return {**base_result, **self._create_empty_motion_result(frame_idx)}

        movement_data = self.movement_tracker.track(pose_data)
        if not movement_data['is_moving']:
            return {**base_result, **self._create_static_motion_result(frame_idx, pose_data)}

        # Perform parallel motion analysis
        futures = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures.append(executor.submit(self.action_classifier.classify, pose_data))
            futures.append(executor.submit(self.metrics_calculator.compute_metrics, pose_data))
            futures.append(executor.submit(self.trajectory_analyzer.analyze, pose_data))

        action_result = futures[0].result()
        metrics = futures[1].result()
        trajectory = futures[2].result()

        self._update_segments(frame_idx, pose_data, action_result, metrics)

        # Additional motion analysis
        motion_result = {
            'pose_data': pose_data.tolist(),
            'movement': movement_data,
            'action': action_result,
            'metrics': metrics,
            'phase': self.phase_analyzer.analyze(pose_data),
            'patterns': self.pattern_matcher.find_matches(pose_data),
            'sequence': self.sequence_analyzer.analyze(self.current_segments),
            'symmetry': self.symmetry_analyzer.compute_symmetry(pose_data),
            'tempo': self.tempo_analyzer.analyze(pose_data),
            'trajectory': trajectory
        }

        return {**base_result, **motion_result}

    def analyze_video(self, video_path: str) -> Dict:
        """Override video analysis to include motion-specific results"""
        # Get base scene analysis results
        base_results = super().analyze_video(video_path)
        
        # Add motion-specific results
        final_segments = self._finalize_segments()
        
        return {
            **base_results,
            'motion_segments': [self._segment_to_dict(seg) for seg in final_segments],
            'motion_parameters': self._get_motion_parameters()
        }

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

    def _get_motion_parameters(self) -> Dict:
        return {
            'confidence_threshold': self.config['detection']['confidence_threshold'],
            'movement_threshold': self.config['tracking']['movement_threshold'],
            'action_threshold': self.config['detection']['action_threshold'],
            'phase_window': self.config['analysis']['phase_window'],
            'pattern_similarity': self.config['analysis']['pattern_similarity'],
            'trajectory_smoothing': self.config['analysis']['trajectory_smoothing']
        }

