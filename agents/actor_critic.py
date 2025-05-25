# actor_critic.py

import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """
    Decoupled MLP actor-critic for flattened grid input.
    Shared feature extractor, with separate policy and value heads.
    Supports action masking to prevent invalid moves.
    """
    def __init__(self, input_dim: int, action_dim: int, hidden_dims: list):
        super().__init__()
        # Shared trunk
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        self.shared = nn.Sequential(*layers)

        # Policy head (independent)
        self.policy_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, action_dim)
        )

        # Value head (independent)
        self.value_head = nn.Sequential(
            nn.Linear(last_dim, last_dim),
            nn.ReLU(),
            nn.Linear(last_dim, 1)
        )

        # Normalize shared features
        self.feature_norm = nn.LayerNorm(last_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.BoolTensor = None):
        # x:     (B, input_dim)
        # valid_mask: optional (B, action_dim) boolean mask of valid actions
        h = self.shared(x)
        # normalize shared features
        h_norm = self.feature_norm(h)

        # Compute policy logits
        logits = self.policy_head(h_norm)
        # Apply action mask if provided
        if valid_mask is not None:
            # mask out invalid actions
            logits = logits.masked_fill(~valid_mask, float('-1e9'))
        dist = Categorical(logits=logits)

        # Compute value from normalized features
        value = self.value_head(h_norm).squeeze(-1)

        return dist, value


class BoardTransformer(nn.Module):
    """
    Transformer encoder over board tokens ([CLS] + M*N cell tokens).
    Supports action masking to prevent invalid moves.
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
        # Embedding for cell values {0,1,2}
        self.token_embed = nn.Embedding(3, d_model)
        # Learnable positional embeddings
        self.pos_embed   = nn.Parameter(torch.randn(num_tokens, d_model))
        # Classification token
        self.cls_token   = nn.Parameter(torch.randn(1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, action_dim)
        )
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def forward(self, board: torch.Tensor):
        # board: (B, rows, cols)
        B, R, C = board.shape
        T = R * C
        # Tokenize cell values
        x = board.long().view(B, T)
        x = self.token_embed(x)
        x = x + self.pos_embed.unsqueeze(0)

        # Prepend CLS token
        cls = self.cls_token.unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        h = self.transformer(x)
        cls_h = h[0]

        # Policy
        logits = self.policy_head(cls_h)
        # Build mask from board: valid if top cell empty
        valid_mask = (board[:, 0, :] == 0)
        logits = logits.masked_fill(~valid_mask, float('-1e9'))
        dist = Categorical(logits=logits)

        # Value
        value = self.value_head(cls_h).squeeze(-1)
        return dist, value