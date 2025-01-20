from datetime import time
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2
from dataclasses import dataclass
import logging
from enum import Enum


class StrikeType(Enum):
    JAB = "jab"
    CROSS = "cross"
    HOOK = "hook"
    UPPERCUT = "uppercut"
    KICK = "kick"
    KNEE = "knee"
    ELBOW = "elbow"
    UNKNOWN = "unknown"


@dataclass
class Contact:
    timestamp: float
    striker_id: int
    receiver_id: int
    strike_type: StrikeType
    impact_point: Tuple[float, float]
    impact_velocity: float
    impact_force_estimate: float
    confidence: float


@dataclass
class CombatMetrics:
    contacts: List[Contact]
    fighter_positions: Dict[int, List[Tuple[float, float]]]
    engagement_distance: float
    fighter_stances: Dict[int, str]
    exchange_intensity: float


class CombatAnalyzer:
    def __init__(self,
                 fps: int = 30,
                 contact_threshold: float = 0.85,
                 min_impact_velocity: float = 5.0):
        """
        Initialize combat sports analyzer.

        Args:
            fps: Frames per second of video
            contact_threshold: Confidence threshold for contact detection
            min_impact_velocity: Minimum velocity to consider as strike (m/s)
        """
        self.fps = fps
        self.contact_threshold = contact_threshold
        self.min_impact_velocity = min_impact_velocity
        self.frame_interval = 1.0 / fps
        self.logger = logging.getLogger(__name__)

        # Store previous frame data
        self.previous_poses = {}
        self.contact_history = []

    def analyze_frame(self,
                      frame: np.ndarray,
                      fighter_poses: Dict[int, Dict],
                      fighter_boxes: Dict[int, Tuple[int, int, int, int]]) -> CombatMetrics:
        """
        Analyze combat sports frame.

        Args:
            frame: Current video frame
            fighter_poses: Dictionary mapping fighter IDs to pose keypoints
            fighter_boxes: Dictionary mapping fighter IDs to bounding boxes

        Returns:
            CombatMetrics containing analysis results
        """
        try:
            current_contacts = []
            fighter_positions = {}

            # Calculate positions and velocities
            for fighter_id, pose in fighter_poses.items():
                positions = self._extract_strike_positions(pose)
                fighter_positions[fighter_id] = positions

                if fighter_id in self.previous_poses:
                    velocities = self._calculate_velocities(
                        self.previous_poses[fighter_id],
                        positions
                    )

                    # Detect potential strikes
                    strikes = self._detect_strikes(velocities, positions)

                    # Check for contacts with other fighters
                    for other_id, other_pose in fighter_poses.items():
                        if other_id != fighter_id:
                            contacts = self._detect_contacts(
                                strikes,
                                positions,
                                self._extract_strike_positions(other_pose)
                            )
                            current_contacts.extend(contacts)

            # Update pose history
            self.previous_poses = fighter_poses

            # Calculate engagement distance
            engagement_distance = self._calculate_engagement_distance(fighter_boxes)

            # Determine fighter stances
            fighter_stances = {
                fighter_id: self._determine_stance(pose)
                for fighter_id, pose in fighter_poses.items()
            }

            # Calculate exchange intensity
            exchange_intensity = self._calculate_exchange_intensity(current_contacts)

            # Update contact history
            self.contact_history.extend(current_contacts)

            return CombatMetrics(
                contacts=current_contacts,
                fighter_positions=fighter_positions,
                engagement_distance=engagement_distance,
                fighter_stances=fighter_stances,
                exchange_intensity=exchange_intensity
            )

        except Exception as e:
            self.logger.error(f"Error in combat analysis: {str(e)}")
            return None

    def _extract_strike_positions(self, pose: Dict) -> Dict[str, Tuple[float, float]]:
        """Extract relevant joint positions for strike detection"""
        return {
            'left_hand': pose['left_wrist'],
            'right_hand': pose['right_wrist'],
            'left_elbow': pose['left_elbow'],
            'right_elbow': pose['right_elbow'],
            'left_foot': pose['left_ankle'],
            'right_foot': pose['right_ankle'],
            'left_knee': pose['left_knee'],
            'right_knee': pose['right_knee'],
            'head': pose['nose'],
            'torso': pose['neck']
        }

    def _calculate_velocities(self,
                              prev_positions: Dict[str, Tuple[float, float]],
                              curr_positions: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """Calculate velocities of strike-relevant joints"""
        velocities = {}
        for joint in prev_positions.keys():
            prev_pos = np.array(prev_positions[joint])
            curr_pos = np.array(curr_positions[joint])
            velocity = np.linalg.norm(curr_pos - prev_pos) / self.frame_interval
            velocities[joint] = velocity
        return velocities

    def _detect_strikes(self,
                        velocities: Dict[str, float],
                        positions: Dict[str, Tuple[float, float]]) -> List[Dict]:
        """Detect potential strikes based on joint velocities"""
        strikes = []

        # Check for different strike types
        if velocities['right_hand'] > self.min_impact_velocity:
            if positions['right_hand'][0] > positions['torso'][0]:  # Cross
                strikes.append({
                    'type': StrikeType.CROSS,
                    'position': positions['right_hand'],
                    'velocity': velocities['right_hand']
                })
            else:  # Hook or Uppercut
                vertical_displacement = positions['right_hand'][1] - positions['right_elbow'][1]
                if vertical_displacement > 0:
                    strikes.append({
                        'type': StrikeType.UPPERCUT,
                        'position': positions['right_hand'],
                        'velocity': velocities['right_hand']
                    })
                else:
                    strikes.append({
                        'type': StrikeType.HOOK,
                        'position': positions['right_hand'],
                        'velocity': velocities['right_hand']
                    })

        if velocities['left_hand'] > self.min_impact_velocity:
            if positions['left_hand'][0] < positions['torso'][0]:  # Jab
                strikes.append({
                    'type': StrikeType.JAB,
                    'position': positions['left_hand'],
                    'velocity': velocities['left_hand']
                })

        # Detect kicks
        for leg in ['left_foot', 'right_foot']:
            if velocities[leg] > self.min_impact_velocity:
                strikes.append({
                    'type': StrikeType.KICK,
                    'position': positions[leg],
                    'velocity': velocities[leg]
                })

        return strikes

    def _detect_contacts(self,
                         strikes: List[Dict],
                         striker_positions: Dict[str, Tuple[float, float]],
                         receiver_positions: Dict[str, Tuple[float, float]]) -> List[Contact]:
        """Detect contacts between strikes and receiver"""
        contacts = []

        for strike in strikes:
            strike_pos = np.array(strike['position'])

            # Check distance to various target areas
            for target in ['head', 'torso']:
                target_pos = np.array(receiver_positions[target])
                distance = np.linalg.norm(strike_pos - target_pos)

                # If strike is close enough to target
                if distance < 30:  # threshold in pixels
                    impact_force = self._estimate_impact_force(
                        strike['velocity'],
                        distance
                    )

                    confidence = 1.0 - (distance / 30)

                    if confidence > self.contact_threshold:
                        contacts.append(Contact(
                            timestamp= time.time(),
                            striker_id=-1,  # needs to be set by caller
                            receiver_id=-1,  # needs to be set by caller
                            strike_type=strike['type'],
                            impact_point=(float(target_pos[0]), float(target_pos[1])),
                            impact_velocity=strike['velocity'],
                            impact_force_estimate=impact_force,
                            confidence=confidence
                        ))

        return contacts

    def _calculate_engagement_distance(self,
                                       fighter_boxes: Dict[int, Tuple[int, int, int, int]]) -> float:
        """Calculate distance between fighters"""
        if len(fighter_boxes) < 2:
            return float('inf')

        centers = []
        for box in fighter_boxes.values():
            x, y, w, h = box
            centers.append(np.array([x + w / 2, y + h / 2]))

        return np.linalg.norm(centers[0] - centers[1])

    def _determine_stance(self, pose: Dict) -> str:
        """Determine fighter's stance (orthodox/southpaw)"""
        left_shoulder = np.array(pose['left_shoulder'])
        right_shoulder = np.array(pose['right_shoulder'])

        # Simple stance detection based on shoulder position
        if left_shoulder[0] < right_shoulder[0]:
            return "orthodox"
        return "southpaw"

    def _calculate_exchange_intensity(self, contacts: List[Contact]) -> float:
        """Calculate intensity of current exchange"""
        if not contacts:
            return 0.0

        return np.mean([c.impact_force_estimate for c in contacts])

    def _estimate_impact_force(self, velocity: float, distance: float) -> float:
        """Estimate impact force based on velocity and distance"""
        # Simple force estimation model
        # Could be improved with actual physics-based calculations
        return velocity * (1.0 - distance / 30)

    def visualize_frame(self,
                        frame: np.ndarray,
                        metrics: CombatMetrics) -> np.ndarray:
        """
        Visualize combat analysis results.

        Args:
            frame: Original video frame
            metrics: Combat analysis metrics

        Returns:
            Frame with visualizations
        """
        output = frame.copy()

        # Draw contacts
        for contact in metrics.contacts:
            x, y = map(int, contact.impact_point)

            # Draw impact point
            color = (0, 0, 255) if contact.strike_type in [StrikeType.CROSS, StrikeType.HOOK] else (255, 0, 0)
            cv2.circle(output, (x, y), 5, color, -1)

            # Draw strike type and force
            cv2.putText(output,
                        f"{contact.strike_type.value}: {contact.impact_force_estimate:.1f}N",
                        (x + 10, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2)

        # Draw fighter positions and stances
        for fighter_id, positions in metrics.fighter_positions.items():
            stance = metrics.fighter_stances[fighter_id]
            center = positions['torso']
            x, y = map(int, center)

            cv2.putText(output,
                        f"Fighter {fighter_id} ({stance})",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2)

        # Draw engagement distance
        cv2.putText(output,
                    f"Distance: {metrics.engagement_distance:.1f}px",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        # Draw exchange intensity
        cv2.putText(output,
                    f"Intensity: {metrics.exchange_intensity:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2)

        return output
