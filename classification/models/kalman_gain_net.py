import torch
import torch.nn as nn


class GainNet(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, per_layer=False, n_layers=1):
        super().__init__()
        self.per_layer = per_layer
        self.n_layers = n_layers
        out_dim = 1 if not per_layer else n_layers
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (1, in_dim)
        return self.net(feats)
