import cv2
import numpy as np
from typing import List, Tuple, Dict



class ClubPathDrawer:
    def __init__(self):
        self.colors = {
            'club_path': (255, 0, 0),  # Blue for club path
            'club_head': (0, 255, 255),  # Yellow for club head
            'swing_plane': (128, 128, 0)  # Dark yellow for swing plane
        }
        self.path_thickness = 2
        self.head_radius = 5

    def draw_club_path(self, frame: np.ndarray,
                       club_positions: List[Tuple[int, int]]) -> np.ndarray:
        """Draw club head path on frame."""
        output = frame.copy()

        # Draw historical path
        for i in range(1, len(club_positions)):
            start_point = tuple(map(int, club_positions[i - 1]))
            end_point = tuple(map(int, club_positions[i]))
            cv2.line(output, start_point, end_point,
                     self.colors['club_path'], self.path_thickness)

        # Draw current club head position
        if club_positions:
            current_pos = tuple(map(int, club_positions[-1]))
            cv2.circle(output, current_pos, self.head_radius,
                       self.colors['club_head'], -1)

        return output

    def draw_swing_plane(self, frame: np.ndarray,
                         plane_points: List[Tuple[int, int]]) -> np.ndarray:
        """Draw swing plane visualization."""
        output = frame.copy()

        # Draw swing plane outline
        if len(plane_points) > 2:
            plane_points = np.array(plane_points, np.int32)
            cv2.polylines(output, [plane_points],
                          True, self.colors['swing_plane'], 2)

            # Optional: Add transparency to plane
            overlay = output.copy()
            cv2.fillPoly(overlay, [plane_points], self.colors['swing_plane'])
            cv2.addWeighted(overlay, 0.3, output, 0.7, 0, output)

        return output

    def draw_club_metrics(self, frame: np.ndarray,
                          metrics: Dict) -> np.ndarray:
        """Draw club metrics (speed, face angle, etc.)"""
        output = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = 30

        for metric_name, value in metrics.items():
            cv2.putText(output, f"{metric_name}: {value}",
                        (10, y_offset), font, 0.5, (255, 255, 255), 1)
            y_offset += 20

        return output
