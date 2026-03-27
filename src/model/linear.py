from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class LinearConfig:
    in_channels: int = 3
    input_length: int = 128
    dropout: float = 0.0
    num_classes: int = 2
    rpm_conditioning: bool = False
    rpm_embed_dim: int = 16


class LinearClassifier(nn.Module):
    """Very simple baseline: flatten (C, L) -> linear.

    Input: x shaped (B, C, L)
    Output: logits shaped (B, num_classes)

    Notes:
    - Works especially well when paired with `feature_mode: order_spectrum`,
      where L is the number of order bins.
    - If `rpm_conditioning` is enabled, it late-fuses an RPM embedding.
    """

    def __init__(self, cfg: LinearConfig):
        super().__init__()
        self.cfg = cfg

        c = int(cfg.in_channels)
        l = int(cfg.input_length)
        if c <= 0 or l <= 0:
            raise ValueError("in_channels and input_length must be > 0")

        self.drop = nn.Dropout(p=float(cfg.dropout))

        self.rpm_conditioning = bool(getattr(cfg, "rpm_conditioning", False))
        rpm_embed_dim = int(getattr(cfg, "rpm_embed_dim", 16))

        feat_dim = c * l
        if self.rpm_conditioning:
            if rpm_embed_dim <= 0:
                raise ValueError("rpm_embed_dim must be > 0 when rpm_conditioning is enabled")
            self.rpm_mlp = nn.Sequential(
                nn.Linear(1, rpm_embed_dim),
                nn.ReLU(),
                nn.Linear(rpm_embed_dim, rpm_embed_dim),
                nn.ReLU(),
            )
            self.fc = nn.Linear(feat_dim + rpm_embed_dim, int(cfg.num_classes))
        else:
            self.rpm_mlp = None
            self.fc = nn.Linear(feat_dim, int(cfg.num_classes))

    def forward(self, x: torch.Tensor, rpm: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, C, L)
        b = int(x.shape[0])
        x = x.reshape(b, -1)
        x = self.drop(x)

        if self.rpm_conditioning:
            if rpm is None:
                raise ValueError("rpm must be provided when rpm_conditioning is enabled")
            r = rpm
            if r.dim() == 0:
                r = r.view(1)
            r = r.to(dtype=x.dtype, device=x.device).view(-1, 1)
            r = torch.log1p(torch.clamp(r, min=0.0))
            assert self.rpm_mlp is not None
            r_emb = self.rpm_mlp(r)
            x = torch.cat([x, r_emb], dim=1)

        return self.fc(x)
