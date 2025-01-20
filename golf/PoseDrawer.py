import cv2
import numpy as np
from typing import List, Tuple, Dict


class PoseDrawer:
    def __init__(self):
        self.colors = {
            'joints': (0, 255, 0),  # Green for joints
            'bones': (255, 255, 0),  # Yellow for bones
            'angles': (0, 0, 255)  # Red for angles
        }
        self.joint_radius = 4
        self.line_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5

    def draw_pose(self, frame: np.ndarray, pose_data: Dict) -> np.ndarray:
        """Draw pose keypoints and connections on frame."""
        output = frame.copy()

        # Draw joints
        for joint in pose_data['keypoints']:
            x, y = int(joint[0]), int(joint[1])
            cv2.circle(output, (x, y), self.joint_radius,
                       self.colors['joints'], -1)

        # Draw bones (connections between joints)
        for connection in pose_data['connections']:
            start_idx, end_idx = connection
            start_point = tuple(map(int, pose_data['keypoints'][start_idx]))
            end_point = tuple(map(int, pose_data['keypoints'][end_idx]))
            cv2.line(output, start_point, end_point,
                     self.colors['bones'], self.line_thickness)

        # Draw angles if available
        if 'angles' in pose_data:
            for angle_name, angle_data in pose_data['angles'].items():
                vertex = tuple(map(int, angle_data['vertex']))
                angle_value = angle_data['angle']
                cv2.putText(output, f"{angle_name}: {angle_value:.1f}°",
                            (vertex[0] + 10, vertex[1] + 10),
                            self.font, self.font_scale,
                            self.colors['angles'], 1)

        return output