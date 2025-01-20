from typing import Dict, List, Tuple
import numpy as np
from enum import Enum

from activity.base import BaseCombatAnalyzer


class FencingAction(Enum):
    ATTACK = "attack"
    PARRY = "parry"
    RIPOSTE = "riposte"
    FLECHE = "fleche"
    LUNGE = "lunge"
    BEAT = "beat"


class WeaponType(Enum):
    FOIL = "foil"
    EPEE = "epee"
    SABRE = "sabre"


class FencingAnalyzer(BaseCombatAnalyzer):
    def __init__(self,
                 weapon_type: WeaponType = WeaponType.FOIL,
                 fps: int = 30):
        self.weapon_type = weapon_type
        self.fps = fps
        self.previous_poses = {}
        self.action_history = []

        # Weapon-specific parameters
        self.weapon_params = {
            WeaponType.FOIL: {
                'mass': 0.5,
                'target_areas': ['torso'],
                'valid_actions': [FencingAction.ATTACK, FencingAction.PARRY, FencingAction.RIPOSTE]
            },
            WeaponType.EPEE: {
                'mass': 0.77,
                'target_areas': ['whole_body'],
                'valid_actions': [FencingAction.ATTACK, FencingAction.PARRY, FencingAction.RIPOSTE]
            },
            WeaponType.SABRE: {
                'mass': 0.5,
                'target_areas': ['above_waist'],
                'valid_actions': [FencingAction.ATTACK, FencingAction.PARRY, FencingAction.RIPOSTE, FencingAction.BEAT]
            }
        }

    def analyze_frame(self,
                      frame: np.ndarray,
                      poses: Dict,
                      boxes: Dict) -> Dict:
        """
        Analyze fencing-specific actions and interactions.
        """
        results = {
            'actions': [],
            'right_of_way': None,
            'valid_hits': [],
            'distance': 0.0,
            'tempo': 0.0
        }

        try:
            # Calculate velocities and accelerations
            velocities = self._calculate_velocities(poses)

            # Detect actions
            actions = self.detect_actions(poses, velocities)
            results['actions'] = actions

            # Determine right of way
            results['right_of_way'] = self._determine_right_of_way(actions)

            # Detect valid hits
            valid_hits = self._detect_valid_hits(actions, poses)
            results['valid_hits'] = valid_hits

            # Calculate fencing-specific metrics
            results['distance'] = self._calculate_fencing_distance(boxes)
            results['tempo'] = self._calculate_tempo(actions)

            return results

        except Exception as e:
            self.logger.error(f"Error in fencing analysis: {str(e)}")
            return None

    def detect_actions(self,
                       poses: Dict,
                       velocities: Dict) -> List[Dict]:
        """
        Detect fencing-specific actions.
        """
        actions = []

        for fencer_id, pose in poses.items():
            # Detect lunge
            if self._is_lunge(pose, velocities[fencer_id]):
                actions.append({
                    'fencer_id': fencer_id,
                    'action': FencingAction.LUNGE,
                    'confidence': 0.9
                })

            # Detect fleche
            elif self._is_fleche(pose, velocities[fencer_id]):
                actions.append({
                    'fencer_id': fencer_id,
                    'action': FencingAction.FLECHE,
                    'confidence': 0.85
                })

            # Detect parry
            elif self._is_parry(pose, velocities[fencer_id]):
                actions.append({
                    'fencer_id': fencer_id,
                    'action': FencingAction.PARRY,
                    'confidence': 0.8
                })

        return actions

    def _is_lunge(self, pose: Dict, velocity: Dict) -> bool:
        """
        Detect a lunge based on characteristic movement patterns.
        """
        # Check for extended front leg
        front_knee = np.array(pose['right_knee'])
        front_ankle = np.array(pose['right_ankle'])
        front_leg_angle = np.arctan2(front_knee[1] - front_ankle[1],
                                     front_knee[0] - front_ankle[0])

        # Check for bent back leg
        back_knee = np.array(pose['left_knee'])
        back_ankle = np.array(pose['left_ankle'])
        back_leg_angle = np.arctan2(back_knee[1] - back_ankle[1],
                                    back_knee[0] - back_ankle[0])

        # Check arm extension
        shoulder = np.array(pose['right_shoulder'])
        weapon_hand = np.array(pose['right_wrist'])
        arm_extension = np.linalg.norm(weapon_hand - shoulder)

        return (front_leg_angle < -0.7 and
                back_leg_angle > 0.3 and
                arm_extension > 100)  # pixels

    def _determine_right_of_way(self, actions: List[Dict]) -> int:
        """
        Determine which fencer has right of way based on actions.
        """
        # Initialize priority
        priority_fencer = None

        # Sort actions by timestamp
        sorted_actions = sorted(actions, key=lambda x: x['timestamp'])

        for action in sorted_actions:
            if action['action'] == FencingAction.ATTACK:
                if priority_fencer is None:
                    priority_fencer = action['fencer_id']
            elif action['action'] == FencingAction.PARRY:
                if action['fencer_id'] != priority_fencer:
                    priority_fencer = action['fencer_id']

        return priority_fencer

    def _detect_valid_hits(self,
                           actions: List[Dict],
                           poses: Dict) -> List[Dict]:
        """
        Detect valid hits based on weapon type and target areas.
        """
        valid_hits = []
        target_areas = self.weapon_params[self.weapon_type]['target_areas']

        for action in actions:
            if action['action'] in [FencingAction.ATTACK, FencingAction.RIPOSTE]:
                weapon_tip = self._get_weapon_tip_position(
                    poses[action['fencer_id']]
                )

                for target_id, target_pose in poses.items():
                    if target_id == action['fencer_id']:
                        continue

                    if self._is_valid_hit(weapon_tip, target_pose, target_areas):
                        valid_hits.append({
                            'attacker_id': action['fencer_id'],
                            'defender_id': target_id,
                            'action': action['action'],
                            'position': weapon_tip,
                            'timestamp': action['timestamp']
                        })

        return valid_hits

    def _is_valid_hit(self,
                      weapon_tip: np.ndarray,
                      target_pose: Dict,
                      target_areas: List[str]) -> bool:
        """
        Check if hit is valid based on weapon type and target area.
        """
        if 'whole_body' in target_areas:
            return True

        if 'torso' in target_areas:
            torso_polygon = self._get_torso_polygon(target_pose)
            return self._point_in_polygon(weapon_tip, torso_polygon)

        if 'above_waist' in target_areas:
            waist_y = target_pose['pelvis'][1]
            return weapon_tip[1] < waist_y

        return False

    def _calculate_tempo(self, actions: List[Dict]) -> float:
        """
        Calculate the tempo of the bout based on action frequency and timing.
        """
        if len(actions) < 2:
            return 0.0

        intervals = []
        for i in range(1, len(actions)):
            interval = actions[i]['timestamp'] - actions[i - 1]['timestamp']
            intervals.append(interval)

        return 1.0 / np.mean(intervals)  # actions per second

    def _evaluate_combo_pattern(self, actions: List[FencingAction]) -> float:
        """
        Evaluate fencing-specific combinations.
        """
        # Common effective patterns
        good_patterns = [
            (FencingAction.PARRY, FencingAction.RIPOSTE),
            (FencingAction.BEAT, FencingAction.ATTACK),
            (FencingAction.ATTACK, FencingAction.FLECHE)
        ]

        for i in range(len(actions) - 1):
            pair = (actions[i], actions[i + 1])
            if pair in good_patterns:
                return 1.0

        return 0.5
