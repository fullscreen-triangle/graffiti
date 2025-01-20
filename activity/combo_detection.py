from datetime import time
from typing import Optional, Dict

from activity.base import CombatSport, CombatPatternRecognition


class ComboDetectionSystem:
    def __init__(self, sport_type: CombatSport):
        self.sport_type = sport_type
        self.pattern_recognition = CombatPatternRecognition()
        self.combo_buffer = []
        self.max_combo_interval = 1.0  # seconds

    def update(self, new_move: Dict) -> Optional[CombatPatternRecognition]:
        """
        Update combo detection with new move.
        """
        current_time = time.time()

        # Clean old moves from buffer
        self.combo_buffer = [
            move for move in self.combo_buffer
            if current_time - move['timestamp'] <= self.max_combo_interval
        ]

        # Add new move
        self.combo_buffer.append(new_move)

        # Check for combo
        if len(self.combo_buffer) >= 2:
            return self._check_for_combo()

        return None

    def _check_for_combo(self) -> Optional[CombatPatternRecognition]:
        """
        Check if current buffer contains a valid combo.
        """
        sequences = self.pattern_recognition.detect_combinations(
            self.combo_buffer,
            self.sport_type
        )

        if sequences:
            # Return highest scoring sequence
            return max(sequences, key=lambda x: x.effectiveness)

        return None
