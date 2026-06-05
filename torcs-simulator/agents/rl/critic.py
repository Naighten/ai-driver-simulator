"""Twin Q-networks (Critic) for SAC with clipped double-Q learning.

Architecture: two independent MLPs [256, 256] with ReLU.
Uses the minimum of the two Q-values for target computation (TD3-style).
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Single Q-network: MLP estimating Q(s, a).

    Inputs:
        obs_dim: observation dimension (default 32).
        action_dim: action dimension (default 3).
        hidden_dim: hidden layer size (default 256).
        n_hidden: number of hidden layers (default 2).
    """

    def __init__(
        self,
        obs_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        in_dim = obs_dim + action_dim
        layers = []
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q(s, a).

        Args:
            obs: observation tensor of shape (batch, obs_dim).
            action: action tensor of shape (batch, action_dim).

        Returns:
            Q-value tensor of shape (batch, 1).
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class Critic(nn.Module):
    """Twin Q-networks for clipped double-Q learning.

    Inputs: same as QNetwork.

    Outputs:
        forward(obs, action) -> (q1, q2): both Q-values.
        q_min(obs, action) -> minimum of the two Q-values.
    """

    def __init__(
        self,
        obs_dim: int = 32,
        action_dim: int = 3,
        hidden_dim: int = 256,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, hidden_dim, n_hidden)
        self.q2 = QNetwork(obs_dim, action_dim, hidden_dim, n_hidden)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple:
        """Compute both Q-values.

        Args:
            obs: observation tensor of shape (batch, obs_dim).
            action: action tensor of shape (batch, action_dim).

        Returns:
            (q1, q2) tuple of tensors, each shape (batch, 1).
        """
        return self.q1(obs, action), self.q2(obs, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Minimum of the two Q-values (for target computation)."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)
