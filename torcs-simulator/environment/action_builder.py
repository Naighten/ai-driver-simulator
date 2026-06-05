"""Builds a CarControl message from a 3-dimensional action array.

The action array from the agent contains:
  - steer: [-1, 1]
  - accel: [0, 1]
  - brake: [0, 1]

Gear is handled automatically via RPM heuristic.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from carControl import CarControl
from utils.gear import recommend_gear


class ActionBuilder:
    """Converts agent action (np.ndarray) into a CarControl with auto gear.

    Inputs:
        smoothing_alpha (float): exponential moving average factor (1.0 = no smoothing).

    Outputs:
        build(agent_action, rpm, speed_x, prev_control) -> CarControl
    """

    def __init__(self, smoothing_alpha: float = 1.0):
        self.smoothing_alpha = smoothing_alpha
        self._smooth_steer = 0.0
        self._smooth_accel = 0.0
        self._smooth_brake = 0.0

    def build(
        self,
        agent_action: np.ndarray,
        rpm: float,
        speed_x: float,
        current_gear: int,
        prev_steer: float = 0.0,
        prev_accel: float = 0.0,
        prev_brake: float = 0.0,
    ) -> CarControl:
        """Build a CarControl from raw agent action and sensor feedback.

        Args:
            agent_action: array of shape (3,) with [steer, accel, brake].
            rpm: current engine RPM for gear recommendation.
            speed_x: longitudinal speed in km/h.
            current_gear: current gear (1-6).
            prev_steer, prev_accel, prev_brake: previous frame values for smoothing.

        Returns:
            CarControl instance ready to send to TORCS.
        """
        steer = float(np.clip(agent_action[0], -1.0, 1.0))
        accel = float(np.clip(agent_action[1], 0.0, 1.0))
        brake = float(np.clip(agent_action[2], 0.0, 1.0))

        if self.smoothing_alpha < 1.0:
            steer = self.smoothing_alpha * steer + (1.0 - self.smoothing_alpha) * prev_steer
            accel = self.smoothing_alpha * accel + (1.0 - self.smoothing_alpha) * prev_accel
            brake = self.smoothing_alpha * brake + (1.0 - self.smoothing_alpha) * prev_brake

        self._smooth_steer = steer
        self._smooth_accel = accel
        self._smooth_brake = brake

        gear = recommend_gear(rpm, current_gear, speed_x)

        return CarControl(accel=accel, brake=brake, gear=gear, steer=steer, meta=0)
