"""Dry-run verification: model forward pass with the baseline 2D config.

Builds the U-Net from the baseline config (in_channels=2) and runs a dummy
forward pass to confirm it accepts a 2-channel input and returns a
1-channel output.

NOTE: Model(config) reads config.model.*, config.data.image_size and
config.diffusion.num_diffusion_timesteps (see models/diffusion.py).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from models.diffusion import Model


class Config:
    """Recursively wrap a dict so fields are attribute-accessible."""

    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)


CONFIG_PATH = "configs/baseline_2d_npy_fair_400k.yml"


def main():
    cfg_dict = yaml.safe_load(open(CONFIG_PATH))
    cfg = Config(cfg_dict)

    print(f"[CONFIG] {CONFIG_PATH}")
    print(f"[CONFIG] in_channels = {cfg.model.in_channels}, "
          f"out_ch = {cfg.model.out_ch}, num_heads = {cfg.model.num_heads}")
    print()

    model = Model(cfg)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,} ({n_params / 1e6:.2f}M)")

    # 2D mode: noisy FD (1ch) + LD condition (1ch) = 2 channels
    img = cfg.data.image_size
    x = torch.randn(2, cfg.model.in_channels, img, img)
    t = torch.tensor([100, 200])

    with torch.no_grad():
        out = model(x, t)

    print(f"Input shape:  {tuple(x.shape)}")
    print(f"Output shape: {tuple(out.shape)}")
    print(f"Expected:     (2, {cfg.model.out_ch}, {img}, {img})")

    expected = (2, cfg.model.out_ch, img, img)
    assert tuple(out.shape) == expected, f"Shape mismatch! Got {tuple(out.shape)}"
    print("Model forward pass: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
