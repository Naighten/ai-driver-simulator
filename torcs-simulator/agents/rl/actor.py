"""Gaussian policy (Actor) for SAC with tanh squashing.

Architecture: MLP [256, 256] with ReLU, outputs mean + log_std.
Uses the reparameterization trick for differentiable sampling.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class Actor(nn.Module):
    """Gaussian stochastic policy with tanh output squashing.

    Inputs:
        obs_dim: observation dimension (default 32).
        action_dim: action dimension (default 3).
        hidden_dim: hidden layer size (default 256).
        n_hidden: number of hidden layers (default 2).
        action_scale: scaling factor for actions (default 1.0).
        action_bias: bias for actions (default 0.0).

    Outputs:
        forward(obs) -> (squashed_action, log_prob, mean, std)
        get_action(obs, deterministic=False) -> action
    """

    def __init__(
        self,
        obs_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 256,
        n_hidden: int = 2,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.action_bias = action_bias

        layers = []
        in_dim = obs_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: returns squashed action, log_prob, mean, std.

        Args:
            obs: observation tensor of shape (batch, obs_dim).

        Returns:
            (squashed_action, log_prob, mean, std)
        """
        h = self.backbone(obs)
        mean = self.mean_layer(h)
        log_std = self.log_std_layer(h)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Reparameterization trick
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)

        # Squash correction: log_prob -= sum(log(1 - tanh(z)^2 + epsilon))
        log_prob = normal.log_prob(z).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        action = action * self.action_scale + self.action_bias
        return action, log_prob, mean, std

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> np.ndarray:
        """Get a single action for inference.

        Args:
            obs: observation tensor of shape (obs_dim,) or (1, obs_dim).
            deterministic: if True, use mean (no sampling).

        Returns:
            action array of shape (action_dim,).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        with torch.no_grad():
            h = self.backbone(obs)
            mean = self.mean_layer(h)
            log_std = self.log_std_layer(h)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)

            if deterministic:
                z = mean
            else:
                z = mean + std * torch.randn_like(std)

            action = torch.tanh(z)
            action = action * self.action_scale + self.action_bias

        return action.squeeze(0).cpu().numpy()
