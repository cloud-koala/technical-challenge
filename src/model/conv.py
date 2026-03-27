from __future__ import annotations

from dataclasses import dataclass

from typing import Iterable, Sequence

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ConvNetConfig:
    in_channels: int = 3
    stem_channels: int = 64
    block_channels: Sequence[int] = (128, 128, 256, 256)
    kernel_sizes: Sequence[int] = (3, 5, 7)
    dropout: float = 0.2
    num_classes: int = 2
    rpm_conditioning: bool = False
    rpm_embed_dim: int = 16


class _MultiScaleConv1D(nn.Module):
    """Multi-scale 1D conv block (Inception-ish) for time-series features."""

    def __init__(
        self,
        *,
        in_ch: int,
        out_ch: int,
        kernel_sizes: Iterable[int],
        dropout: float,
    ):
        super().__init__()
        ks = [int(k) for k in kernel_sizes]
        if len(ks) == 0:
            raise ValueError("kernel_sizes must be non-empty")

        # Split channels evenly across branches.
        n_branches = len(ks)
        base = out_ch // n_branches
        rem = out_ch - base * n_branches
        branch_out = [base + (1 if i < rem else 0) for i in range(n_branches)]

        branches = []
        for k, bo in zip(ks, branch_out):
            pad = k // 2
            branches.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, bo, kernel_size=k, padding=pad, bias=False),
                    nn.BatchNorm1d(bo),
                    nn.ReLU(),
                )
            )
        self.branches = nn.ModuleList(branches)

        self.mix = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        return self.mix(y)


class _ResBlock1D(nn.Module):
    def __init__(
        self,
        *,
        in_ch: int,
        out_ch: int,
        kernel_sizes: Iterable[int],
        dropout: float,
        downsample: bool,
    ):
        super().__init__()

        stride = 2 if downsample else 1
        self.proj = None
        if stride != 1 or in_ch != out_ch:
            self.proj = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

        self.conv1 = _MultiScaleConv1D(
            in_ch=in_ch,
            out_ch=out_ch,
            kernel_sizes=kernel_sizes,
            dropout=dropout,
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, stride=stride, bias=False),
            nn.BatchNorm1d(out_ch),
        )

        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)

        if self.proj is not None:
            identity = self.proj(identity)

        out = out + identity
        out = self.act(out)
        out = self.drop(out)
        return out


class ConvNetClassifier(nn.Module):
    """Pure 1D CNN for vibration classification.

    Input: x shaped (B, C, L)
    Output: logits shaped (B, num_classes)

    Design notes:
    - Multi-scale convolutions help capture different frequency bands.
    - Residual connections stabilize training.
    - Downsampling via stride reduces compute while growing receptive field.
    - Global average pooling makes the classifier length-agnostic.
    """

    def __init__(self, cfg: ConvNetConfig):
        super().__init__()
        self.cfg = cfg

        if int(cfg.stem_channels) <= 0:
            raise ValueError("stem_channels must be > 0")

        self.stem = nn.Sequential(
            nn.Conv1d(cfg.in_channels, cfg.stem_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(cfg.stem_channels),
            nn.ReLU(),
        )

        blocks = []
        in_ch = int(cfg.stem_channels)
        for i, out_ch in enumerate(cfg.block_channels):
            out_ch_i = int(out_ch)
            blocks.append(
                _ResBlock1D(
                    in_ch=in_ch,
                    out_ch=out_ch_i,
                    kernel_sizes=cfg.kernel_sizes,
                    dropout=float(cfg.dropout),
                    downsample=(i % 2 == 1),
                )
            )
            in_ch = out_ch_i
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(p=float(cfg.dropout))

        self.rpm_conditioning = bool(getattr(cfg, "rpm_conditioning", False))
        rpm_embed_dim = int(getattr(cfg, "rpm_embed_dim", 16))
        if self.rpm_conditioning:
            if rpm_embed_dim <= 0:
                raise ValueError("rpm_embed_dim must be > 0 when rpm_conditioning is enabled")
            self.rpm_mlp = nn.Sequential(
                nn.Linear(1, rpm_embed_dim),
                nn.ReLU(),
                nn.Linear(rpm_embed_dim, rpm_embed_dim),
                nn.ReLU(),
            )
            self.classifier = nn.Linear(in_ch + rpm_embed_dim, cfg.num_classes)
        else:
            self.rpm_mlp = None
            self.classifier = nn.Linear(in_ch, cfg.num_classes)

    def forward(self, x: torch.Tensor, rpm: torch.Tensor | None = None) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
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

        return self.classifier(x)


# Backwards-compatible aliases (in case other code imports the old names).
ConvLSTMConfig = ConvNetConfig
ConvLSTMClassifier = ConvNetClassifier
