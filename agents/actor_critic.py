import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Tuple

class ActorCritic(nn.Module):
    """
    Combined actor–critic network.
    Given state → (action distribution, state value)
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        last_dim = input_dim
        self.dropout = nn.Dropout(0.05)
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), nn.LeakyReLU(0.01), self.dropout]
            last_dim = h
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last_dim, action_dim)
        self.value_head = nn.Linear(last_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[Categorical, torch.Tensor]:
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        dist = Categorical(logits=logits)
        return dist, value
