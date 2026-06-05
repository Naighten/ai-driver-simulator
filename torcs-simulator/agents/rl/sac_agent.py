"""Soft Actor-Critic (SAC) agent with automatic entropy tuning.

Implements the learner side of distributed Ape-X style training.
The learner samples from a shared replay buffer and updates network parameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict
from copy import deepcopy

from agents.rl.actor import Actor
from agents.rl.critic import Critic
from agents.rl.shared_buffer import SharedReplayBuffer


class SACAgent:
    """Soft Actor-Critic learner.

    Inputs:
        obs_dim (int): observation dimension (default 32).
        action_dim (int): action dimension (default 3).
        hidden_dim (int): hidden layer size (default 256).
        n_hidden (int): number of hidden layers (default 2).
        lr (float): learning rate (default 3e-4).
        gamma (float): discount factor (default 0.99).
        tau (float): soft target update coefficient (default 0.005).
        alpha (float or str): entropy coefficient, 'auto' for automatic tuning.
        policy_freq (int): delay steps between policy updates (default 2).
        batch_size (int): batch size (default 256).
        buffer_capacity (int): replay buffer capacity (default 1e6).
        device (str): 'cpu' or 'cuda'.

    Outputs:
        update(buffer) -> metrics dict.
        get_actor_state() -> actor state_dict for worker sync.
        save/load methods.
    """

    def __init__(
        self,
        obs_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: str = "auto",
        policy_freq: int = 2,
        batch_size: int = 256,
        buffer_capacity: int = 1_000_000,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.policy_freq = policy_freq
        self.batch_size = batch_size
        self.total_steps = 0

        # Networks
        self.actor = Actor(obs_dim, action_dim, hidden_dim, n_hidden).to(self.device)
        self.critic = Critic(obs_dim, action_dim, hidden_dim, n_hidden).to(self.device)
        self.target_critic = deepcopy(self.critic)

        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.alpha_is_auto = isinstance(alpha, str) and alpha.lower() == "auto"
        if self.alpha_is_auto:
            target_entropy = -float(action_dim)
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
            self.target_entropy = target_entropy
        else:
            self.alpha = float(alpha)
            self.log_alpha = torch.zeros(1)
            self.target_entropy = None

        # Buffer (can be replaced by shared buffer externally)
        self.buffer = SharedReplayBuffer(obs_dim, action_dim, buffer_capacity)

    def set_shared_buffer(self, buffer: SharedReplayBuffer) -> None:
        """Replace internal buffer with a shared one for distributed training."""
        self.buffer = buffer

    def update(self) -> Dict[str, float]:
        """Perform one SAC update step using a batch from the buffer.

        Returns:
            dict of metric names -> scalar values.
        """
        batch = self.buffer.sample(self.batch_size)
        obs, actions, rewards, next_obs, dones = [b.to(self.device) for b in batch]

        with torch.no_grad():
            next_actions, next_log_probs, _, _ = self.actor(next_obs)
            q1_target, q2_target = self.target_critic(next_obs, next_actions)
            q_target = torch.min(q1_target, q2_target)
            q_target = q_target - self.alpha * next_log_probs
            q_backup = rewards + self.gamma * (1 - dones) * q_target

        # Critic loss (MSE on both Q-networks)
        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(q1, q_backup) + nn.functional.mse_loss(q2, q_backup)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        metrics = {}

        # Delayed policy update
        if self.total_steps % self.policy_freq == 0:
            # Actor loss
            new_actions, log_probs, _, _ = self.actor(obs)
            q1_new, q2_new = self.critic(obs, new_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha * log_probs - q_new).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Alpha loss (autotune)
            if self.alpha_is_auto:
                alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha = self.log_alpha.exp().item()
                metrics["alpha_loss"] = alpha_loss.item()

            # Soft update target networks
            self._soft_update_target()

            metrics["actor_loss"] = actor_loss.item()

        metrics["critic_loss"] = critic_loss.item()
        metrics["q_value"] = q1.mean().item()
        metrics["alpha"] = self.alpha

        self.total_steps += 1
        return metrics

    def _soft_update_target(self) -> None:
        """Polyak averaging for target critic."""
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def get_actor_state(self) -> dict:
        """Get actor state dict for worker synchronization."""
        return {k: v.cpu().clone() for k, v in self.actor.state_dict().items()}

    def set_actor_weights(self, state_dict: dict) -> None:
        """Load actor weights from a state dict (for worker sync)."""
        self.actor.load_state_dict(state_dict)

    def get_checkpoint_data(self) -> dict:
        """Get all data needed for a full checkpoint save."""
        return {
            "actor_state": self.actor.state_dict(),
            "critic_state": self.critic.state_dict(),
            "target_critic_state": self.target_critic.state_dict(),
            "actor_optimizer_state": self.actor_optim.state_dict(),
            "critic_optimizer_state": self.critic_optim.state_dict(),
            "alpha": self.alpha,
            "log_alpha": self.log_alpha.clone(),
            "alpha_optimizer_state": self.alpha_optim.state_dict() if self.alpha_is_auto else None,
            "step": self.total_steps,
        }

    def load_checkpoint_data(self, data: dict) -> None:
        """Load all training state from a checkpoint dict."""
        self.actor.load_state_dict(data["actor_state"])
        self.critic.load_state_dict(data["critic_state"])
        self.target_critic.load_state_dict(data["target_critic_state"])
        self.actor_optim.load_state_dict(data["actor_optimizer_state"])
        self.critic_optim.load_state_dict(data["critic_optimizer_state"])
        self.alpha = data["alpha"]
        self.log_alpha = data["log_alpha"]
        if self.alpha_is_auto and data.get("alpha_optimizer_state") is not None:
            self.alpha_optim.load_state_dict(data["alpha_optimizer_state"])
        self.total_steps = data["step"]
