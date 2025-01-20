import numpy as np
from typing import Dict, List, Tuple
from collections import deque

class StabilityAnalyzer:
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.cop_buffer = {}
        self.com_buffer = {}

    def analyze_stability(self, athlete_id: int, positions: Dict,
                        cop: Tuple[float, float]) -> Dict:
        # Initialize buffers if needed
        if athlete_id not in self.cop_buffer:
            self.cop_buffer[athlete_id] = deque(maxlen=self.window_size)
            self.com_buffer[athlete_id] = deque(maxlen=self.window_size)

        # Update buffers
        self.cop_buffer[athlete_id].append(cop)
        com = self._estimate_com(positions)
        self.com_buffer[athlete_id].append(com)

        # Calculate stability metrics
        stability_metrics = {
            'base_of_support': self._calculate_base_of_support(positions),
            'stability_margin': self._calculate_stability_margin(com, positions),
            'cop_velocity': self._calculate_cop_velocity(athlete_id),
            'postural_sway': self._calculate_postural_sway(athlete_id)
        }

        return stability_metrics

    def _estimate_com(self, positions: Dict) -> np.ndarray:
        # Simplified COM estimation
        return np.mean([positions['hip'], positions['knee'], positions['ankle']], axis=0)

    def _calculate_base_of_support(self, positions: Dict) -> float:
        return np.linalg.norm(positions['ankle'] - positions['toe'])

    def _calculate_stability_margin(self, com: np.ndarray, positions: Dict) -> float:
        return np.linalg.norm(com - positions['toe'])

    def _calculate_cop_velocity(self, athlete_id: int) -> float:
        if len(self.cop_buffer[athlete_id]) < 2:
            return 0.0
        return np.linalg.norm(np.diff(self.cop_buffer[athlete_id][-2:], axis=0))

    def _calculate_postural_sway(self, athlete_id: int) -> float:
        if len(self.com_buffer[athlete_id]) < 2:
            return 0.0
        return np.std([np.linalg.norm(com) for com in self.com_buffer[athlete_id]])
