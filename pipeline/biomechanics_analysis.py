import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from biomechanics.dynamics_analyzer import DynamicsAnalyzer
from biomechanics.grf_analyzer import GRFAnalyzer
from biomechanics.kinematics_analyzer import KinematicsAnalyzer
from biomechanics.posture_converter import MannequinConverter
from biomechanics.stability_analyzer import StabilityAnalyzer
from biomechanics.stride_analyzer import StrideAnalyzer
from biomechanics.sync_analyzer import SynchronizationAnalyzer
from core.athlete_detection import AthleteDetector
from core.skeleton import SkeletonDrawer
from pipeline.motion_analysis import MotionAnalysisPipeline


@dataclass
class TrackingInfo:
    id: int
    bbox: Tuple[int, int, int, int]
    confidence: float
    history: List[Tuple[float, float]]


class AthleteTracker:
    def __init__(self, max_athletes: int, iou_threshold: float):
        self.max_athletes = max_athletes
        self.iou_threshold = iou_threshold
        self.tracked_athletes: Dict[int, TrackingInfo] = {}
        self.next_id = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        current_athletes = {}

        for detection in detections:
            matched = False
            bbox = detection['bbox']

            for track_id, track_info in self.tracked_athletes.items():
                if self._calculate_iou(bbox, track_info.bbox) > self.iou_threshold:
                    current_athletes[track_id] = TrackingInfo(
                        id=track_id,
                        bbox=bbox,
                        confidence=detection['confidence'],
                        history=track_info.history + [detection['center']]
                    )
                    matched = True
                    break

            if not matched and len(current_athletes) < self.max_athletes:
                current_athletes[self.next_id] = TrackingInfo(
                    id=self.next_id,
                    bbox=bbox,
                    confidence=detection['confidence'],
                    history=[detection['center']]
                )
                self.next_id += 1

        self.tracked_athletes = current_athletes
        return [{'id': k, **dataclasses.asdict(v)} for k, v in current_athletes.items()]

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                       bbox2: Tuple[int, int, int, int]) -> float:
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        intersection_x = max(x1, x2)
        intersection_y = max(y1, y2)
        intersection_w = min(x1 + w1, x2 + w2) - intersection_x
        intersection_h = min(y1 + h1, y2 + h2) - intersection_y

        if intersection_w <= 0 or intersection_h <= 0:
            return 0.0

        intersection_area = intersection_w * intersection_h
        union_area = w1 * h1 + w2 * h2 - intersection_area

        return intersection_area / union_area


class BiomechanicsAnalysisPipeline(MotionAnalysisPipeline):
    """Pipeline for biomechanical analysis that builds upon motion analysis"""
    def __init__(self, config: dict):
        super().__init__(config)
        
        # Biomechanics-specific components
        self.athlete_detector = AthleteDetector()
        self.athlete_tracker = AthleteTracker(
            max_athletes=config['tracking']['max_athletes'],
            iou_threshold=config['tracking']['iou_threshold']
        )
        self.skeleton_drawer = SkeletonDrawer()

        fps = config['processing'].get('fps', 30)
        self.kinematics_analyzer = KinematicsAnalyzer(fps=fps)
        self.dynamics_analyzer = DynamicsAnalyzer()
        self.grf_analyzer = GRFAnalyzer()
        self.stability_analyzer = StabilityAnalyzer(
            window_size=config['analysis'].get('stability_window', 30)
        )
        self.stride_analyzer = StrideAnalyzer(fps=fps)
        self.sync_analyzer = SynchronizationAnalyzer(
            window_size=config['analysis'].get('sync_window', 30)
        )
        self.mannequin_converter = MannequinConverter()

        self.max_workers = config['processing']['max_workers']
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)

    def _process_frame(self, frame: np.ndarray, frame_idx: int, prev_frame: Optional[np.ndarray]) -> Dict:
        """Override frame processing to add biomechanical analysis"""
        # Get base motion analysis results
        base_result = super()._process_frame(frame, frame_idx, prev_frame)
        
        # Add biomechanical analysis
        detections = self.human_detector.detect(frame)
        athlete_data = self.athlete_detector.identify_athletes(detections)
        tracked_athletes = self.athlete_tracker.update(athlete_data)

        biomech_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._analyze_athlete, frame, athlete, frame_idx)
                for athlete in tracked_athletes
            ]
            for future in futures:
                result = future.result()
                if result:
                    biomech_results.append(result)

        return {
            **base_result,
            'athletes': biomech_results,
            'total_detected': len(biomech_results)
        }

    def analyze_video(self, video_path: str) -> Dict:
        """Override video analysis to include biomechanics-specific results"""
        # Get base motion analysis results
        base_results = super().analyze_video(video_path)
        
        return {
            **base_results,
            'biomechanics_parameters': {
                'max_athletes': self.config['tracking']['max_athletes'],
                'confidence_threshold': self.config['detection']['confidence_threshold'],
                'iou_threshold': self.config['tracking']['iou_threshold']
            }
        }

    def _analyze_athlete(self, frame: np.ndarray, athlete_data: Dict, frame_idx: int) -> Optional[Dict]:
        skeleton = self.skeleton_drawer.extract_skeleton(frame, athlete_data['bbox'])
        if skeleton is None:
            return None

        kinematics = self.kinematics_analyzer.analyze(skeleton)
        dynamics = self.dynamics_analyzer.analyze(skeleton, kinematics)
        grf = self.grf_analyzer.estimate(dynamics)
        stability = self.stability_analyzer.analyze(skeleton, dynamics)
        stride_metrics = self.stride_analyzer.analyze(skeleton, frame_idx)
        sync_metrics = self.sync_analyzer.analyze(skeleton, frame_idx)
        mannequin = self.mannequin_converter.convert(skeleton)

        return {
            'athlete_id': athlete_data['id'],
            'bbox': athlete_data['bbox'],
            'confidence': athlete_data['confidence'],
            'skeleton': skeleton.tolist(),
            'kinematics': kinematics,
            'dynamics': dynamics,
            'grf': grf,
            'stability': stability,
            'stride': stride_metrics,
            'synchronization': sync_metrics,
            'mannequin': mannequin
        }
