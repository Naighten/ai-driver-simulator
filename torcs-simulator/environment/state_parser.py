import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from carState import CarState


def car_state_to_array(state: CarState, prev_action: np.ndarray = None) -> np.ndarray:
    """Convert a CarState to a 32-dimensional normalized float32 observation.

    Observation layout (32 floats), each scaled to ~[-1, 1]:
        [0]      angle / pi
        [1]      speedX / 200
        [2]      speedY / 50
        [3]      speedZ / 200
        [4]      trackPos (raw, roughly [-1, 1])
        [5..23]  track_0 .. track_18 / 200  (19 range sensors, 0-200m)
        [24..27] wheelSpinVel_0 .. wheelSpinVel_3 / 100
        [28]     rpm / 10000
        [29]     prev_steer
        [30]     prev_accel
        [31]     prev_brake

    Args:
        state: parsed CarState from SCR protocol.
        prev_action: (3,) array of [prev_steer, prev_accel, prev_brake].

    Returns:
        np.ndarray of shape (32,) with dtype float32.
        Missing sensor values default to 0.0.
    """
    track = getattr(state, "track", None) or [0.0] * 19
    wheel = getattr(state, "wheelSpinVel", None) or [0.0] * 4

    obs = np.zeros(32, dtype=np.float32)
    angle = state.angle if state.angle is not None else 0.0
    obs[0] = angle / np.pi
    obs[1] = (state.speedX if state.speedX is not None else 0.0) / 200.0
    obs[2] = (state.speedY if state.speedY is not None else 0.0) / 50.0
    obs[3] = (state.speedZ if state.speedZ is not None else 0.0) / 200.0
    obs[4] = state.trackPos if state.trackPos is not None else 0.0

    for i in range(min(len(track), 19)):
        t = track[i] if track[i] is not None else 0.0
        obs[5 + i] = t / 200.0

    for i in range(min(len(wheel), 4)):
        w = wheel[i] if wheel[i] is not None else 0.0
        obs[24 + i] = w / 100.0

    obs[28] = (state.rpm if state.rpm is not None else 0.0) / 10000.0

    if prev_action is not None and len(prev_action) >= 3:
        obs[29] = float(prev_action[0])  # prev_steer
        obs[30] = float(prev_action[1])  # prev_accel
        obs[31] = float(prev_action[2])  # prev_brake

    return obs
