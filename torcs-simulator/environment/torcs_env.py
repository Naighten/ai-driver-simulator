"""Gymnasium environment wrapper for TORCS via SCR protocol.

Uses the SCR-prefix init protocol: sends SCR(init <19 angles>) and waits
for ***identified*** response before starting the control loop.
"""

import time
import socket
import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.scr_client import SCRClient
from environment.state_parser import car_state_to_array
from environment.action_builder import ActionBuilder
from environment.reward import compute_reward
from utils.normalizer import RunningMeanStd

logger = logging.getLogger(__name__)


class TorcsEnv(gym.Env):
    """Gymnasium environment for TORCS.

    Observation: Box(-inf, inf, (32,), float32)
    Action: Box(-1, 1, (3,), float32) -> [steer, accel, brake]
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 3001,
        timeout: float = 1.0,
        max_steps: int = 10000,
        grace_steps: int = 100,
        offtrack_limit: float = 1.5,
        offtrack_patience: int = 5,
        stuck_speed_threshold: float = 2.0,
        stuck_steps: int = 100,
        smoothing_alpha: float = 1.0,
        action_repeat: int = 4,
        min_accel: float = 0.2,
        use_normalizer: bool = False,
        reward_weights: Optional[Dict[str, float]] = None,
        identify_attempts: int = 120,
        identify_delay: float = 0.02,
        reset_recv_retries: int = 20,
        reset_recv_delay: float = 0.05,
    ):
        super().__init__()

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_steps = max_steps
        self.grace_steps = grace_steps
        self.offtrack_limit = offtrack_limit
        self.offtrack_patience = offtrack_patience
        self.stuck_speed_threshold = stuck_speed_threshold
        self.stuck_steps = stuck_steps
        self.action_repeat = action_repeat
        self.min_accel = min_accel
        self.use_normalizer = use_normalizer
        self.reward_weights = reward_weights or {}
        self.identify_attempts = identify_attempts
        self.identify_delay = identify_delay
        self.reset_recv_retries = reset_recv_retries
        self.reset_recv_delay = reset_recv_delay

        self.client: Optional[SCRClient] = None
        self.action_builder = ActionBuilder(smoothing_alpha=smoothing_alpha)
        self.normalizer = RunningMeanStd(shape=(32,)) if use_normalizer else None

        self._step_count = 0
        self._stuck_count = 0
        self._offtrack_steps = 0
        self._prev_damage = 0.0
        self._prev_dist_raced = 0.0
        self._prev_steer = 0.0
        self._prev_accel = 0.0
        self._prev_brake = 0.0
        self._prev_speed_x = 0.0
        self._prev_rpm = 0.0
        self._prev_gear = 1
        self._prev_action = np.zeros(3, dtype=np.float32)

    def _connect(self) -> None:
        """Establish connection: identify with TORCS via SCR(init ...)."""
        if self.client is None:
            self.client = SCRClient(
                host=self.host, port=self.port, timeout=self.timeout,
            )
        if not self.client.identify(
            attempts=self.identify_attempts, delay=self.identify_delay
        ):
            raise RuntimeError(
                f"Failed to identify with TORCS on {self.host}:{self.port}"
                f" after {self.identify_attempts} attempts"
            )
        logger.info(f"Identified with TORCS on {self.host}:{self.port}")

    def _recv_initial_state(self) -> None:
        """Receive first state after identification, with retries."""
        for attempt in range(1, self.reset_recv_retries + 1):
            try:
                self.client.recv_state()
                if self.client.state.getSpeedX() is not None:
                    return
            except (socket.timeout, OSError):
                pass
            time.sleep(self.reset_recv_delay)

        raise RuntimeError(
            f"Failed to receive initial state from "
            f"{self.host}:{self.port} after {self.reset_recv_retries} retries"
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset: connect, identify, receive initial state."""
        super().reset(seed=seed)

        if self.client is not None:
            self.client.reset()
            time.sleep(0.2)
            if not self.client.identify(
                attempts=self.identify_attempts, delay=self.identify_delay
            ):
                self.close()
                time.sleep(1.0)
                self._connect()
        else:
            self._connect()
        self._recv_initial_state()

        state = self.client.state
        obs = car_state_to_array(state)
        if self.use_normalizer and self.normalizer is not None:
            self.normalizer.update(obs)
            obs = self.normalizer.normalize(obs)

        self._step_count = 0
        self._stuck_count = 0
        self._offtrack_steps = 0
        self._prev_damage = state.damage if state.damage is not None else 0.0
        self._prev_dist_raced = state.distRaced if state.distRaced is not None else 0.0
        self._prev_steer = 0.0
        self._prev_accel = 0.0
        self._prev_brake = 0.0
        self._prev_speed_x = state.speedX if state.speedX is not None else 0.0
        self._prev_rpm = state.rpm if state.rpm is not None else 0.0
        self._prev_gear = state.gear if state.gear is not None else 1
        self._prev_action[:] = 0.0

        info = {
            "dist_raced": state.distRaced,
            "cur_lap_time": state.curLapTime,
            "last_lap_time": state.lastLapTime,
            "damage": state.damage,
        }
        logger.info(f"Reset OK, speed={state.speedX}")
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step: repeat action, accumulate reward, check termination."""
        self._step_count += 1

        steer = float(np.clip(action[0], -1.0, 1.0))
        accel = float(np.clip(action[1], self.min_accel, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))
        if brake > 0.5:
            accel = 0.0

        total_reward = 0.0
        terminated = False
        final_obs = None
        final_info = {}

        for _ in range(self.action_repeat):
            control = self.action_builder.build(
                agent_action=np.array([steer, accel, brake]),
                rpm=self._prev_rpm,
                speed_x=self._prev_speed_x,
                current_gear=self._prev_gear,
                prev_steer=self._prev_steer,
                prev_accel=self._prev_accel,
                prev_brake=self._prev_brake,
            )
            self.client.send_action(control)

            try:
                self.client.recv_state()
            except (socket.timeout, OSError) as e:
                logger.warning(f"recv failed step {self._step_count}: {e}")
                return np.zeros(32, dtype=np.float32), -1.0, True, False, {"reason": str(e)}

            state = self.client.state
            speed_x = state.speedX if state.speedX is not None else 0.0
            track_pos = state.trackPos if state.trackPos is not None else 0.0
            angle = state.angle if state.angle is not None else 0.0
            damage = state.damage if state.damage is not None else 0.0
            dist_raced = state.distRaced if state.distRaced is not None else 0.0
            track = getattr(state, "track", None) or [0.0] * 19

            reward = compute_reward(
                speed_x=speed_x, angle=angle, track_pos=track_pos,
                damage=damage, prev_damage=self._prev_damage,
                dist_raced=dist_raced, prev_dist_raced=self._prev_dist_raced,
                track=track, current_steer=steer, prev_steer=self._prev_steer,
                step_count=self._step_count, grace_steps=self.grace_steps,
                weights=self.reward_weights,
            )
            total_reward += reward

            self._prev_damage = damage
            self._prev_dist_raced = dist_raced
            self._prev_speed_x = speed_x
            self._prev_rpm = state.rpm if state.rpm is not None else 0.0
            self._prev_gear = state.gear if state.gear is not None else 1

            final_obs = car_state_to_array(state, self._prev_action)
            final_info = {
                "dist_raced": dist_raced, "cur_lap_time": state.curLapTime,
                "last_lap_time": state.lastLapTime, "damage": damage,
                "speed_x": speed_x, "track_pos": track_pos, "angle": angle,
            }

            # Check offtrack with patience (only after grace period)
            if self._step_count > self.grace_steps and abs(track_pos) > self.offtrack_limit:
                self._offtrack_steps += 1
                if self._offtrack_steps >= self.offtrack_patience:
                    terminated = True
            else:
                self._offtrack_steps = 0

            # Check stuck (only after grace period)
            if self._step_count > self.grace_steps and speed_x < self.stuck_speed_threshold:
                self._stuck_count += 1
                if self._stuck_count >= self.stuck_steps:
                    terminated = True
            else:
                self._stuck_count = 0

            if terminated:
                break

        self._prev_steer = steer
        self._prev_accel = accel
        self._prev_brake = brake
        self._prev_action[:] = [steer, accel, brake]

        if self.use_normalizer and self.normalizer is not None and final_obs is not None:
            self.normalizer.update(final_obs)
            final_obs = self.normalizer.normalize(final_obs)

        truncated = self._step_count >= self.max_steps
        return final_obs, total_reward, terminated, truncated, final_info

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None

    def get_normalizer_state(self) -> Optional[dict]:
        return self.normalizer.get_state() if self.normalizer is not None else None

    def set_normalizer_state(self, state: dict) -> None:
        if self.normalizer is not None:
            self.normalizer.set_state(state)
