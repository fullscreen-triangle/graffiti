import numpy as np
from typing import Dict, Optional


class MannequinConverter:
    def __init__(self):
        self.default_pose = {
            "version": 7,
            "data": [
                [0, 0, 0],  # 0: figure position
                [0, 0, 0],  # 1: figure rotation
                [0, 0, 0],  # 2: neck
                [0, 0, 0],  # 3: head
                [0, 0, 0],  # 4: left arm upper
                [0],  # 5: left arm lower
                [0, 0, 0],  # 6: left hand
                [0, 0, 0],  # 7: right arm upper
                [0],  # 8: right arm lower
                [0, 0, 0],  # 9: right hand
                [0, 0, 0],  # 10: left leg upper
                [0],  # 11: left leg lower
                [0, 0, 0],  # 12: left foot
                [0, 0, 0],  # 13: right leg upper
                [0],  # 14: right leg lower
                [0, 0, 0]  # 15: right foot
            ]
        }

    def convert_angles(self, kinematics_results: Dict) -> Dict:
        """Convert kinematics angles to mannequin.js format"""
        if not kinematics_results:
            return self.default_pose

        mannequin_pose = self.default_pose.copy()
        angles = kinematics_results['joint_angles']

        # Convert radians to degrees
        def rad2deg(rad):
            return float(np.rad2deg(rad))

        # Left arm
        mannequin_pose["data"][4] = [
            rad2deg(angles['shoulder']['bend']),  # bend
            rad2deg(angles['shoulder']['turn']),  # turn
            rad2deg(angles['shoulder']['tilt'])  # tilt
        ]
        mannequin_pose["data"][5] = [rad2deg(angles['elbow']['bend'])]  # elbow bend

        # Right arm (mirrored from left if not available)
        mannequin_pose["data"][7] = [
            rad2deg(angles['shoulder']['bend']),  # bend
            -rad2deg(angles['shoulder']['turn']),  # turn (mirrored)
            -rad2deg(angles['shoulder']['tilt'])  # tilt (mirrored)
        ]
        mannequin_pose["data"][8] = [rad2deg(angles['elbow']['bend'])]

        # Left leg
        mannequin_pose["data"][10] = [
            rad2deg(angles['hip']['bend']),  # bend
            rad2deg(angles['hip']['turn']),  # turn
            rad2deg(angles['hip']['tilt'])  # tilt
        ]
        mannequin_pose["data"][11] = [rad2deg(angles['knee']['bend'])]  # knee bend

        # Right leg (mirrored from left if not available)
        mannequin_pose["data"][13] = [
            rad2deg(angles['hip']['bend']),  # bend
            -rad2deg(angles['hip']['turn']),  # turn (mirrored)
            -rad2deg(angles['hip']['tilt'])  # tilt (mirrored)
        ]
        mannequin_pose["data"][14] = [rad2deg(angles['knee']['bend'])]

        return mannequin_pose

    def calculate_figure_rotation(self, kinematics_results: Dict) -> list:
        """Calculate overall figure rotation from hip orientation"""
        if not kinematics_results:
            return [0, 0, 0]

        positions = kinematics_results['positions']

        # Calculate torso orientation using hip and shoulder positions
        left_hip = positions['hip']
        left_shoulder = positions['shoulder']

        # Calculate forward direction (Z-axis)
        forward = np.array([0, 0, 1])

        # Calculate up direction (Y-axis) from hip to shoulder
        up = left_shoulder - left_hip
        up = up / np.linalg.norm(up)

        # Calculate rotation angles
        pitch = np.arctan2(up[2], up[1])  # Forward/backward tilt
        roll = np.arctan2(up[0], up[1])  # Side tilt
        yaw = 0  # We need additional reference for yaw calculation

        return [
            float(np.rad2deg(pitch)),
            float(np.rad2deg(yaw)),
            float(np.rad2deg(roll))
        ]

    def convert_to_mannequin(self, kinematics_results: Dict) -> Dict:
        """Main conversion function that combines all transformations"""
        mannequin_pose = self.convert_angles(kinematics_results)

        # Update figure rotation
        mannequin_pose["data"][1] = self.calculate_figure_rotation(kinematics_results)

        # Calculate figure position (optional, if you want to position the figure in 3D space)
        if kinematics_results and 'positions' in kinematics_results:
            hip_pos = kinematics_results['positions']['hip']
            # Convert to mannequin.js coordinate system
            mannequin_pose["data"][0] = [
                float(hip_pos[0] * 100),  # Scale to centimeters
                float(hip_pos[1] * 100),
                float(hip_pos[2] * 100)
            ]

        return mannequin_pose

