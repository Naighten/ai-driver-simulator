"""Instance manager for TORCS environment pool.

Creates, validates, and manages a pool of TorcsEnv instances
mapped to UDP ports. Used by both SAC workers and NEAT evaluators.
"""

import logging
from typing import Dict, List, Optional
import socket

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from environment.torcs_env import TorcsEnv

logger = logging.getLogger(__name__)


class InstanceManager:
    """Manages a pool of TorcsEnv instances across multiple UDP ports.

    Inputs:
        ports: list of UDP port numbers for TORCS instances.
        host: TORCS hostname.
        timeout: UDP socket timeout.
        max_steps: max agent steps per episode.
        grace_steps: steps before offtrack/stuck checks activate.
        offtrack_limit: |trackPos| threshold for offtrack detection.
        offtrack_patience: consecutive offtrack steps before termination.
        stuck_speed_threshold: speed threshold for stuck detection.
        stuck_steps: consecutive stuck steps before reset.
        smoothing_alpha: action smoothing factor.
        action_repeat: number of physics steps per agent step.
        min_accel: minimum acceleration floor (0-1).
        reward_weights: reward function weights dict.
        use_normalizer: whether to enable observation normalization.

    Outputs:
        get_env(port) -> TorcsEnv: get env for a specific port.
        acquire_free() -> (port, TorcsEnv): acquire any available env.
        release(port): return env to pool.
        check_ports(): validate port availability.
        close_all(): shutdown all envs.
    """

    def __init__(
        self,
        ports: List[int],
        host: str = "localhost",
        timeout: float = 1.0,
        max_steps: int = 50000,
        grace_steps: int = 100,
        offtrack_limit: float = 1.5,
        offtrack_patience: int = 5,
        stuck_speed_threshold: float = 2.0,
        stuck_steps: int = 100,
        smoothing_alpha: float = 1.0,
        action_repeat: int = 4,
        min_accel: float = 0.2,
        reward_weights: Optional[Dict] = None,
        use_normalizer: bool = False,
    ):
        self.host = host
        self.timeout = timeout
        self.max_steps = max_steps
        self.grace_steps = grace_steps
        self.offtrack_limit = offtrack_limit
        self.offtrack_patience = offtrack_patience
        self.stuck_speed_threshold = stuck_speed_threshold
        self.stuck_steps = stuck_steps
        self.smoothing_alpha = smoothing_alpha
        self.action_repeat = action_repeat
        self.min_accel = min_accel
        self.reward_weights = reward_weights or {}
        self.use_normalizer = use_normalizer

        self._envs: Dict[int, TorcsEnv] = {}
        self._in_use: Dict[int, bool] = {}
        self._lock = None  # Optional threading.Lock for thread safety

        for port in ports:
            env = self._create_env(port)
            self._envs[port] = env
            self._in_use[port] = False

        logger.info(
            f"InstanceManager initialized with {len(self._envs)} envs on ports {ports}"
        )

    def _create_env(self, port: int) -> TorcsEnv:
        """Create a single TorcsEnv instance.

        Note: connection is established on first reset() call.
        """
        env = TorcsEnv(
            host=self.host,
            port=port,
            timeout=self.timeout,
            max_steps=self.max_steps,
            grace_steps=self.grace_steps,
            offtrack_limit=self.offtrack_limit,
            offtrack_patience=self.offtrack_patience,
            stuck_speed_threshold=self.stuck_speed_threshold,
            stuck_steps=self.stuck_steps,
            smoothing_alpha=self.smoothing_alpha,
            action_repeat=self.action_repeat,
            min_accel=self.min_accel,
            use_normalizer=self.use_normalizer,
            reward_weights=self.reward_weights,
        )
        return env

    def check_ports(self) -> List[int]:
        """Check which ports have a running TORCS instance.

        Tries a UDP connect + send of a dummy message.
        Returns list of ports that responded.
        """
        available = []
        for port, env in self._envs.items():
            try:
                test_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                test_sock.settimeout(2.0)
                test_sock.connect((self.host, port))
                test_sock.send(b"(meta 1)\n")
                try:
                    test_sock.recv(1024)
                    available.append(port)
                except socket.timeout:
                    logger.warning(f"Port {port}: no response (timeout)")
                test_sock.close()
            except Exception as e:
                logger.warning(f"Port {port}: connection failed - {e}")

        if len(available) < len(self._envs):
            logger.warning(
                f"Only {len(available)}/{len(self._envs)} TORCS instances available"
            )

        return available

    def get_env(self, port: int) -> Optional[TorcsEnv]:
        """Get the TorcsEnv for a specific port."""
        return self._envs.get(port)

    def acquire_free(self) -> Optional[tuple]:
        """Acquire an available (not in use) environment.

        Returns:
            Tuple of (port, TorcsEnv), or None if all are busy.
        """
        for port, in_use in self._in_use.items():
            if not in_use:
                self._in_use[port] = True
                return port, self._envs[port]
        logger.warning("No free environment available")
        return None

    def release(self, port: int) -> None:
        """Release an environment back to the pool."""
        if port in self._in_use:
            self._in_use[port] = False

    def close_all(self) -> None:
        """Close all environment instances."""
        for port, env in self._envs.items():
            try:
                env.close()
            except Exception as e:
                logger.error(f"Error closing env on port {port}: {e}")
        self._envs.clear()
        self._in_use.clear()
        logger.info("All environments closed")
