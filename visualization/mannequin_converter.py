

import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Union
import numpy as np


class MannequinConverter:
    def __init__(self, config_path: Optional[str] = None):
        # Default joint mapping for mannequin.js
        self.joint_mapping = {
            'hip_l': {'mannequin_name': 'l_leg', 'axis_order': 'xyz'},
            'hip_r': {'mannequin_name': 'r_leg', 'axis_order': 'xyz'},
            'knee_l': {'mannequin_name': 'l_knee', 'axis_order': 'zxy'},
            'knee_r': {'mannequin_name': 'r_knee', 'axis_order': 'zxy'},
            'ankle_l': {'mannequin_name': 'l_ankle', 'axis_order': 'xyz'},
            'ankle_r': {'mannequin_name': 'r_ankle', 'axis_order': 'xyz'},
            'shoulder_l': {'mannequin_name': 'l_arm', 'axis_order': 'xyz'},
            'shoulder_r': {'mannequin_name': 'r_arm', 'axis_order': 'xyz'},
            'elbow_l': {'mannequin_name': 'l_elbow', 'axis_order': 'zxy'},
            'elbow_r': {'mannequin_name': 'r_elbow', 'axis_order': 'zxy'},
            'wrist_l': {'mannequin_name': 'l_wrist', 'axis_order': 'xyz'},
            'wrist_r': {'mannequin_name': 'r_wrist', 'axis_order': 'xyz'},
            'neck': {'mannequin_name': 'neck', 'axis_order': 'xyz'},
            'head': {'mannequin_name': 'head', 'axis_order': 'xyz'}
        }

        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """Load custom joint mapping configuration"""
        with open(config_path, 'r') as f:
            config = json.load(f)
            if 'joint_mapping' in config:
                self.joint_mapping.update(config['joint_mapping'])

    def convert(self, analysis_results: Dict) -> Dict:
        """
        Convert analysis results to mannequin.js format

        Args:
            analysis_results: Dictionary containing analysis results from the pipeline

        Returns:
            Dictionary formatted for mannequin.js visualization
        """
        converted_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0',
                'analysis_type': analysis_results.get('analysis_type', 'general'),
                'source': analysis_results.get('source', ''),
                'parameters': analysis_results.get('parameters', {})
            },
            'frames': []
        }

        for frame in analysis_results.get('frames', []):
            frame_data = self._convert_frame(frame)
            if frame_data:
                converted_data['frames'].append(frame_data)

        return converted_data

    def _convert_frame(self, frame: Dict) -> Dict:
        """Convert individual frame data"""
        frame_data = {
            'frame_number': frame.get('frame_number', 0),
            'timestamp': frame.get('timestamp', 0.0),
            'athletes': []
        }

        for athlete in frame.get('athletes', []):
            athlete_data = self._convert_athlete(athlete)
            if athlete_data:
                frame_data['athletes'].append(athlete_data)

        return frame_data

    def _convert_athlete(self, athlete: Dict) -> Optional[Dict]:
        """Convert individual athlete data"""
        if not athlete.get('skeleton'):
            return None

        try:
            mannequin_data = self._create_mannequin_data(athlete['skeleton'])
            return {
                'track_id': athlete.get('track_id', 0),
                'mannequin': mannequin_data,
                'biomechanics': athlete.get('biomechanics', {}),
                'motion_analysis': athlete.get('motion_analysis', {}),
                'activity_analysis': athlete.get('activity_analysis', {})
            }
        except Exception as e:
            print(f"Error converting athlete data: {str(e)}")
            return None

    def _create_mannequin_data(self, skeleton: Dict) -> Dict:
        """Create mannequin.js compatible joint data"""
        mannequin_data = []

        # Base position and rotation
        mannequin_data.append(skeleton.get('position', [0, 0, 0]))
        mannequin_data.append(skeleton.get('rotation', [0, -90, 0]))

        # Convert joint angles
        joints = skeleton.get('joints', {})
        angles = skeleton.get('angles', {})

        for joint_name, mapping in self.joint_mapping.items():
            angle_data = self._process_joint_angle(
                joint_name,
                angles.get(joint_name, {}),
                mapping['axis_order']
            )
            mannequin_data.append(angle_data)

        return {
            "version": 7,
            "data": mannequin_data
        }

    def _process_joint_angle(self,
                             joint_name: str,
                             angle_data: Union[Dict, float, int],
                             axis_order: str) -> List[float]:
        """Process joint angle data based on joint type and axis order"""
        if isinstance(angle_data, (float, int)):
            # Handle legacy single-angle format
            if 'knee' in joint_name or 'elbow' in joint_name:
                return [0, 0, float(angle_data)]
            return [float(angle_data), 0, 0]

        elif isinstance(angle_data, dict):
            # Handle full 3D rotation data
            angles = [0.0, 0.0, 0.0]
            for i, axis in enumerate(axis_order):
                angles[i] = float(angle_data.get(axis, 0.0))
            return angles

        return [0.0, 0.0, 0.0]

    def export_to_file(self,
                       converted_data: Dict,
                       output_path: str,
                       pretty_print: bool = True) -> None:
        """Export converted data to JSON file"""
        with open(output_path, 'w') as f:
            if pretty_print:
                json.dump(converted_data, f, indent=2)
            else:
                json.dump(converted_data, f)
