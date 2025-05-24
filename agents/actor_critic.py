# actor_critic.py

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Standard MLP actor-critic for flattened grid input.
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.shared = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_dim)
        self.value_head  = nn.Linear(last, 1)

    def forward(self, x: torch.Tensor):
        # x: (B, input_dim)
        h = self.shared(x)
        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)
        dist   = Categorical(logits=logits)
        return dist, value


class BoardTransformer(nn.Module):
    """
    Transformer encoder over board tokens ([CLS] + M*N cell tokens).
    """
    def __init__(self,
                 rows: int,
                 cols: int,
                 action_dim: int,
                 d_model=64,
                 nhead=4,
                 layers=3,
                 dropout=0.1):
        super().__init__()
        num_tokens = rows * cols
        # embedding for cell values {0,1,2}
        self.token_embed = nn.Embedding(3, d_model)
        # learnable positional embeddings
        self.pos_embed   = nn.Parameter(torch.randn(num_tokens, d_model))
        # classification token
        self.cls_token   = nn.Parameter(torch.randn(1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # output heads
        self.policy_head = nn.Linear(d_model, action_dim)
        self.value_head  = nn.Linear(d_model, 1)

    def forward(self, board: torch.Tensor):
        # board: (B, rows, cols)
        B, R, C = board.shape
        T = R * C
        x = board.long().view(B, T)                  # (B, T)
        x = self.token_embed(x)                      # (B, T, d_model)
        x = x + self.pos_embed.unsqueeze(0)          # add positional

        # prepend CLS token
        cls = self.cls_token.unsqueeze(0).expand(B, -1, -1)  # (B,1,d_model)
        x = torch.cat([cls, x], dim=1)                       # (B, T+1, d_model)

        # transformer expects (S, B, d)
        x = x.transpose(0,1)                                 # (T+1, B, d_model)
        h = self.transformer(x)
        cls_h = h[0]                                         # (B, d_model)

        logits = self.policy_head(cls_h)                     # (B, action_dim)
        value  = self.value_head(cls_h).squeeze(-1)          # (B,)
        return Categorical(logits=logits), value
