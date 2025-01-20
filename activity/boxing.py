from dataclasses import dataclass
from datetime import time
from enum import Enum
from typing import Dict, List, Optional

import numpy as np

from base import BaseCombatAnalyzer


class PunchType(Enum):
    JAB = "jab"
    CROSS = "cross"
    LEFT_HOOK = "left_hook"
    RIGHT_HOOK = "right_hook"
    LEFT_UPPERCUT = "left_uppercut"
    RIGHT_UPPERCUT = "right_uppercut"


@dataclass
class PunchClassification:
    punch_type: PunchType
    confidence: float
    velocity: float
    power: float
    accuracy: float


class BoxingAnalyzer(BaseCombatAnalyzer):
    def __init__(self):
        # Define keypoint indices for standard pose estimation format
        self.NOSE = "0"
        self.LEFT_SHOULDER = "5"
        self.RIGHT_SHOULDER = "6"
        self.LEFT_ELBOW = "7"
        self.RIGHT_ELBOW = "8"
        self.LEFT_WRIST = "9"
        self.RIGHT_WRIST = "10"
        self.LEFT_HIP = "11"
        self.RIGHT_HIP = "12"

        # Punch classification parameters
        self.velocity_threshold = 0.5  # m/s
        self.extension_threshold = 0.8  # percentage of full extension
        self.hook_angle_threshold = 45  # degrees
        self.uppercut_angle_threshold = 60  # degrees

    def analyze_frame(self, frame: np.ndarray, poses: Dict, boxes: Dict) -> Dict:
        """Implementation of abstract method from BaseCombatAnalyzer"""
        results = {
            'poses': poses,
            'boxes': boxes,
            'strikes': [],
            'contacts': []
        }

        if poses:
            velocities = self._calculate_velocities(poses)
            strikes = self.detect_strikes(poses, velocities)
            contacts = self.detect_contacts(strikes, poses)

            results['strikes'] = strikes
            results['contacts'] = contacts

        return results

    def detect_strikes(self, poses: Dict, velocities: Dict) -> List[Dict]:
        """Implementation of abstract method from BaseCombatAnalyzer"""
        strikes = []

        for pose_id, pose_data in poses.items():
            if pose_id in velocities:
                left_velocity = velocities[pose_id].get(self.LEFT_WRIST, np.zeros(3))
                right_velocity = velocities[pose_id].get(self.RIGHT_WRIST, np.zeros(3))

                # Analyze left hand
                if np.linalg.norm(left_velocity) > self.velocity_threshold:
                    strike = self._analyze_strike(pose_data, left_velocity, "left")
                    if strike:
                        strikes.append(strike)

                # Analyze right hand
                if np.linalg.norm(right_velocity) > self.velocity_threshold:
                    strike = self._analyze_strike(pose_data, right_velocity, "right")
                    if strike:
                        strikes.append(strike)

        return strikes

    def detect_contacts(self, strikes: List[Dict], poses: Dict) -> List[Dict]:
        """Implementation of abstract method from BaseCombatAnalyzer"""
        contacts = []

        for strike in strikes:
            if 'pose_id' in strike and strike['pose_id'] in poses:
                pose = poses[strike['pose_id']]
                contact = self._detect_strike_contact(strike, pose)
                if contact:
                    contacts.append(contact)

        return contacts

    def _analyze_strike(self, pose: Dict[str, np.ndarray], velocity: np.ndarray, side: str) -> Optional[Dict]:
        """Analyze a potential strike and classify it"""
        arm_angle = self._calculate_arm_angle(pose, side)
        extension = self._calculate_arm_extension(pose, side)

        punch_class = self._classify_punch(side, arm_angle, extension, float(np.linalg.norm(velocity)))

        if punch_class.punch_type:
            return {
                'type': punch_class.punch_type.value,
                'velocity': punch_class.velocity,
                'power': punch_class.power,
                'confidence': punch_class.confidence,
                'timestamp': time.time()
            }
        return None

    def _calculate_velocities(self, poses: Dict) -> Dict:
        """Calculate velocities for all tracked points"""
        velocities = {}
        # Implementation of velocity calculation
        return velocities

    def _evaluate_combo_pattern(self, strike_types: List[str]) -> float:
        """Boxing-specific implementation of combo pattern evaluation"""
        # Common effective boxing combinations
        effective_combos = {
            ('jab', 'cross'): 0.9,
            ('jab', 'cross', 'left_hook'): 0.95,
            ('jab', 'cross', 'left_hook', 'right_hook'): 0.85,
            ('left_hook', 'right_hook'): 0.8,
        }

        combo_tuple = tuple(strike_types)
        return effective_combos.get(combo_tuple, 0.5)

