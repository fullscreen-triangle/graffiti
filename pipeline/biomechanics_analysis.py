import dataclasses
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np

from ..biomechanics.dynamics_analyzer import DynamicsAnalyzer
from ..biomechanics.grf_analyzer import GRFAnalyzer
from ..biomechanics.kinematics_analyzer import KinematicsAnalyzer
from ..biomechanics.posture_converter import MannequinConverter
from ..biomechanics.stability_analyzer import StabilityAnalyzer
from ..biomechanics.stride_analyzer import StrideAnalyzer
from ..biomechanics.sync_analyzer import SynchronizationAnalyzer
from ..core.athlete_detection import AthleteDetector
from ..core.human_detector import HumanDetector
from ..core.skeleton import SkeletonDrawer


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


class BiomechanicsAnalysisPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.human_detector = HumanDetector(
            confidence_threshold=config['detection']['confidence_threshold']
        )
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

    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        detections = self.human_detector.detect(frame)
        athlete_data = self.athlete_detector.identify_athletes(detections)
        tracked_athletes = self.athlete_tracker.update(athlete_data)

        results = []
        futures = []

        for athlete in tracked_athletes:
            future = self.executor.submit(
                self._analyze_athlete,
                frame=frame,
                athlete_data=athlete,
                frame_idx=frame_idx
            )
            futures.append(future)

        for future in futures:
            result = future.result()
            if result:
                results.append(result)

        return {
            'frame_idx': frame_idx,
            'athletes': results,
            'total_detected': len(results)
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

    def analyze_video(self, video_path: str) -> Dict:
        self.logger.info(f"Starting biomechanical analysis of {video_path}")

        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        all_results = []
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_results = self.process_frame(frame, frame_idx)
            all_results.append(frame_results)
            frame_idx += 1

        cap.release()

        analysis_summary = {
            'video_info': {
                'path': video_path,
                'frames': frame_count,
                'fps': fps
            },
            'frame_results': all_results,
            'analysis_parameters': {
                'max_athletes': self.config['tracking']['max_athletes'],
                'confidence_threshold': self.config['detection']['confidence_threshold'],
                'iou_threshold': self.config['tracking']['iou_threshold']
            }
        }

        self.logger.info("Biomechanical analysis completed successfully")
        return analysis_summary
