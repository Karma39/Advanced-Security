"""
Per-device binary PLA classifiers.

MLP-FV:    (batch, 505) → Linear(505→256→128) → 2
GRU-DGT:   (batch, 150, 150) → GRU(hidden=64, layers=2) → last hidden → 2
BiGRU-DGT: (batch, 150, 150) → BiGRU(hidden=64, layers=2) → concat(fwd,bwd) → 2
"""

import torch
import torch.nn as nn


class _MLP(nn.Module):

    def __init__(self, input_dim=505, num_classes=2, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class _GRUNet(nn.Module):

    def __init__(self, input_size=150, hidden_size=64, num_layers=2,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        _, h_n = self.gru(x)       # h_n: (num_layers, batch, hidden)
        return self.classifier(h_n[-1])


class _BiGRUNet(nn.Module):

    def __init__(self, input_size=150, hidden_size=64, num_layers=2,
                 num_classes=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.drop       = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        _, h_n = self.gru(x)             # h_n: (num_layers*2, batch, hidden)
        fwd = h_n[-2]                    # last forward  layer
        bwd = h_n[-1]                    # last backward layer
        out = torch.cat([fwd, bwd], dim=1)
        return self.classifier(self.drop(out))


def build_binary_model(name: str, dropout: float = 0.3, n_classes: int = 2) -> nn.Module:
    """name: 'mlp_fv' | 'gru_dgt' | 'bigru_dgt'. Pass n_classes=12 for multiclass."""
    name = name.lower()

    if name == "mlp_fv":
        return _MLP(input_dim=505, num_classes=n_classes, dropout=dropout)

    if name == "gru_dgt":
        return _GRUNet(input_size=150, hidden_size=64,
                       num_layers=2, num_classes=n_classes, dropout=dropout)

    if name == "bigru_dgt":
        return _BiGRUNet(input_size=150, hidden_size=64,
                         num_layers=2, num_classes=n_classes, dropout=dropout)

    raise ValueError(
        f"Unknown model name '{name}'. Choose from: mlp_fv, gru_dgt, bigru_dgt."
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
