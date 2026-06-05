"""Shared replay buffer for distributed SAC (Ape-X style).

Uses torch.multiprocessing with shared memory so multiple worker processes
can write transitions and the learner can sample batches.
"""

import numpy as np
import torch
from typing import Tuple


class SharedReplayBuffer:
    """Thread-safe, shared-memory circular replay buffer.

    Supports multiple writers (workers) and a single reader (learner).

    Inputs:
        obs_dim: observation dimension (default 32).
        action_dim: action dimension (default 3).
        capacity: maximum number of transitions.
    """

    def __init__(
        self,
        obs_dim: int = 32,
        action_dim: int = 3,
        capacity: int = 1_000_000,
    ):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self._size = 0
        self._ptr = 0

        # Pre-allocate shared numpy arrays
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def push(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition to the buffer.

        Args:
            obs: current observation (32,).
            action: action taken (3,).
            reward: scalar reward.
            next_obs: next observation (32,).
            done: whether episode terminated.
        """
        idx = self._ptr % self.capacity
        self.obs[idx] = obs.astype(np.float32)
        self.actions[idx] = action.astype(np.float32)
        self.rewards[idx] = float(reward)
        self.next_obs[idx] = next_obs.astype(np.float32)
        self.dones[idx] = float(done)

        self._ptr += 1
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a batch of transitions uniformly.

        Args:
            batch_size: number of transitions to sample.

        Returns:
            Tuple of (obs, actions, rewards, next_obs, dones) as torch float32 tensors.
        """
        indices = np.random.randint(0, self._size, size=batch_size)
        return (
            torch.from_numpy(self.obs[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_obs[indices]),
            torch.from_numpy(self.dones[indices]),
        )

    def __len__(self) -> int:
        return self._size

    def ready(self, min_size: int) -> bool:
        """Check if buffer has at least min_size transitions."""
        return self._size >= min_size
