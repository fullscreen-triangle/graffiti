from dataclasses import dataclass
import numpy as np
from typing import Dict, Tuple


@dataclass
class Segment:
    mass: float
    length: float
    inertia: float


class DynamicsAnalyzer:
    def __init__(self):
        self.g = 9.81
        self.segments = {
            'thigh': Segment(mass=7.0, length=0.4, inertia=0.1),
            'shank': Segment(mass=3.5, length=0.4, inertia=0.05),
            'foot': Segment(mass=1.0, length=0.2, inertia=0.01)
        }

    def calculate_dynamics(self, positions: Dict, velocities: Dict,
                           accelerations: Dict) -> Dict:
        forces = {}
        moments = {}

        GRF = np.array([0, 2.5 * 70 * self.g])

        for segment in ['foot', 'shank', 'thigh']:
            seg = self.segments[segment]

            if segment == 'foot':
                Fd = GRF
                Md = 0
            else:
                Fd = -forces[prev_segment]['proximal']
                Md = -moments[prev_segment]['proximal']

            Fp, Mp = self._inverse_dynamics(
                positions=positions,
                accelerations=accelerations,
                segment=segment,
                seg_data=seg,
                Fd=Fd,
                Md=Md
            )

            forces[segment] = {'distal': Fd, 'proximal': Fp}
            moments[segment] = {'distal': Md, 'proximal': Mp}
            prev_segment = segment

        return {'forces': forces, 'moments': moments}

    def _inverse_dynamics(self, positions: Dict, accelerations: Dict,
                          segment: str, seg_data: Segment, Fd: np.ndarray,
                          Md: float) -> Tuple[np.ndarray, float]:
        # Inverse dynamics calculation
        com_acc = accelerations.get(segment, np.zeros(2))
        Fp = seg_data.mass * com_acc - Fd - np.array([0, -self.g * seg_data.mass])
        Mp = seg_data.inertia * 0  # Simplified moment calculation

        return Fp, Mp
