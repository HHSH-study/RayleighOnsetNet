# -*- coding: utf-8 -*-
"""Minimal RayleighOnsetNet model definition for inference.

This file contains only the model components needed to load the released
coarse/refine checkpoints and run inference. It is extracted from the final
training implementation used in the manuscript, with training-only dataset,
loss, visualization, and command-line code removed.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def soft_argmax_1d(logits: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
    """Convert 1-D logits to the expected sample index."""
    prob = F.softmax(logits / temperature, dim=-1)
    idx = torch.arange(logits.shape[-1], device=logits.device, dtype=prob.dtype)
    return (prob * idx).sum(dim=-1)


@dataclass
class CFG:
    # Data and CWT settings
    npz_path: str = "seismic_dataset.npz"
    fs: int = 100
    ud_channel: int = 1
    p_min: float = 2.0
    p_max: float = 10.0
    num_scales: int = 48

    # Coarse-stage training settings retained for configuration parity.
    epochs: int = 200
    batch_size: int = 16
    num_workers: int = 6
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 4
    use_amp: bool = True
    use_tf32: bool = True
    accum_steps: int = 1
    lr: float = 3e-4
    weight_decay: float = 2e-3
    grad_clip: float = 1.0
    patience: int = 50
    temperature: float = 0.7

    # Augmentation and target parameters retained for configuration parity.
    augment: bool = True
    shift_prob: float = 0.5
    max_shift_samples: int = 1500
    stretch_prob: float = 0.35
    stretch_range: tuple = (0.95, 1.05)
    amp_prob: float = 0.8
    amp_range: tuple = (0.7, 1.3)
    noise_prob: float = 0.6
    noise_std: float = 0.04
    mask_prob: float = 0.35
    max_mask_sec: float = 1.2
    drift_prob: float = 0.3
    drift_max: float = 0.06
    flip_prob: float = 0.10
    filt_prob: float = 0.25
    filt_kernel_sec: tuple = (0.05, 0.25)
    label_jitter_prob: float = 0.4
    label_jitter_sec: float = 0.8
    sigma_narrow_sec: float = 2.0
    sigma_wide_sec: float = 7.0
    mix_wide: float = 0.30

    # EMA and outputs
    use_ema: bool = True
    ema_decay: float = 0.999
    save_dir: str = "results_rayleigh_v3"
    primary_metric: str = "acc10"
    min_acc10: float = 0.0

    # Refine-stage settings used by the released checkpoint.
    train_refine_epochs: int = 300
    refine_batch_size: int = 40
    refine_lr: float = 3e-5
    refine_weight_decay: float = 2e-3
    refine_patience: int = 60
    refine_window_sec: float = 40.0
    refine_jitter_sec: float = 12.0
    refine_sigma_narrow_sec: float = 1.0
    refine_sigma_wide_sec: float = 3.0
    refine_mix_wide: float = 0.25
    refine_resume_ckpt: str = ""
    mc_sigma_prior_sec: float = 6.0


class CWTLayer(nn.Module):
    def __init__(self, num_scales=48, p_min=2.0, p_max=10.0, fs=100.0):
        super().__init__()
        periods = np.linspace(p_min, p_max, num_scales).astype(np.float64)
        center_freqs = 1.0 / periods

        kernel_size = int(fs * p_max * 1.5)
        if kernel_size % 2 == 0:
            kernel_size += 1

        t = np.arange(kernel_size, dtype=np.float64) / fs
        t = t - (kernel_size / 2.0) / fs

        filters = []
        for f0 in center_freqs:
            sigma = 1.0 / max(f0, 1e-6)
            gaussian = np.exp(-(t**2) / (2 * (sigma**2)))
            sine = np.cos(2 * np.pi * f0 * t)
            weight = gaussian * sine
            weight = weight / (np.sqrt(np.sum(weight**2)) + 1e-12)
            filters.append(weight)

        filt = np.stack(filters, axis=0)
        filt = torch.tensor(filt, dtype=torch.float32).unsqueeze(1)
        self.register_buffer("filters", filt)
        self.padding = kernel_size // 2

    def forward(self, x):
        cwt = F.conv1d(x, self.filters, padding=self.padding)
        cwt = torch.abs(cwt) + 1e-8
        cwt = torch.log1p(cwt)
        return cwt


class GNAct(nn.Module):
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        group_count = min(groups, ch)
        self.gn = nn.GroupNorm(group_count, ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.gn(x))


class Conv2dBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=(3, 9), stride=(2, 1), dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=stride,
            padding=(k[0] // 2, k[1] // 2),
            bias=False,
        )
        self.na = GNAct(out_ch)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.na(self.conv(x)))


class TemporalResBlock(nn.Module):
    def __init__(self, ch, k=7, dilation=1, dropout=0.15):
        super().__init__()
        pad = (k // 2) * dilation
        self.conv = nn.Conv1d(ch, ch, kernel_size=k, padding=pad, dilation=dilation, bias=False)
        self.na = GNAct(ch)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = self.drop(self.na(self.conv(x)))
        return x + y


class RayleighOnsetNet(nn.Module):
    """Dual-branch Rayleigh-wave onset picker used for both coarse and refine stages."""

    def __init__(self, cfg: CFG, feat_ch: int = 112):
        super().__init__()
        self.cfg = cfg
        self.cwt = CWTLayer(num_scales=cfg.num_scales, p_min=cfg.p_min, p_max=cfg.p_max, fs=cfg.fs)

        self.cwt_enc = nn.Sequential(
            Conv2dBlock(1, 32, k=(3, 9), stride=(2, 1), dropout=0.08),
            Conv2dBlock(32, 64, k=(3, 9), stride=(2, 1), dropout=0.08),
            Conv2dBlock(64, 96, k=(3, 9), stride=(2, 1), dropout=0.08),
        )

        self.wav_enc = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=11, padding=5, bias=False),
            GNAct(32),
            nn.Conv1d(32, 64, kernel_size=11, padding=5, bias=False),
            GNAct(64),
            nn.Dropout(0.15),
        )

        scale_count = int(cfg.num_scales)
        for _ in range(3):
            scale_count = (scale_count + 1) // 2
        fuse_in_ch = 96 * scale_count + 64

        self.fuse_1x1 = nn.Conv1d(fuse_in_ch, feat_ch, kernel_size=1, bias=False)
        self.fuse_na = GNAct(feat_ch)
        self.fuse_drop = nn.Dropout(0.15)

        dilations = [1, 2, 4, 8, 16, 32]
        self.temporal = nn.Sequential(
            *[TemporalResBlock(feat_ch, k=7, dilation=d, dropout=0.18) for d in dilations]
        )

        self.head = nn.Conv1d(feat_ch, 1, kernel_size=1)
        self._feat_ch = feat_ch

    def forward(self, x):
        cwt = self.cwt(x).unsqueeze(1)
        f2d = self.cwt_enc(cwt)
        batch_size, channels, scales, time_len = f2d.shape
        f_cwt = f2d.reshape(batch_size, channels * scales, time_len)

        f_wav = self.wav_enc(x)
        fused = torch.cat([f_cwt, f_wav], dim=1)
        fused = self.fuse_drop(self.fuse_na(self.fuse_1x1(fused)))
        fused = self.temporal(fused)
        logits = self.head(fused).squeeze(1)
        return logits


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow:
                raise KeyError(f"EMA shadow is missing parameter: {name}")
            new_value = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
            self.shadow[name] = new_value.clone()

    def apply_shadow(self, model: nn.Module) -> None:
        self.backup = {}
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.backup[name] = param.data.clone()
            param.data = self.shadow[name].clone()

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param.data = self.backup[name].clone()
        self.backup = {}
