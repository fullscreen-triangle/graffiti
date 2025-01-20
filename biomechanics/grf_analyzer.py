import numpy as np
from typing import Dict, Tuple


class GRFAnalyzer:
    def __init__(self):
        self.g = 9.81
        self.body_mass = 70  # Default mass in kg

    def estimate_grf(self, positions: Dict, accelerations: Dict) -> Dict:
        # Vertical GRF estimation
        vertical_grf = self._estimate_vertical_grf(accelerations)

        # Horizontal GRF estimation
        horizontal_grf = self._estimate_horizontal_grf(accelerations)

        # Impact forces
        impact_force = self._estimate_impact_force(positions)

        return {
            'vertical_grf': vertical_grf,
            'horizontal_grf': horizontal_grf,
            'impact_force': impact_force
        }

    def _estimate_vertical_grf(self, accelerations: Dict) -> float:
        # Simple estimation based on acceleration
        com_acc = accelerations.get('com', np.zeros(2))
        return self.body_mass * (self.g + com_acc[1])

    def _estimate_horizontal_grf(self, accelerations: Dict) -> float:
        com_acc = accelerations.get('com', np.zeros(2))
        return self.body_mass * com_acc[0]

    def _estimate_impact_force(self, positions: Dict) -> float:
        # Simplified impact force estimation
        return 2.5 * self.body_mass * self.g
