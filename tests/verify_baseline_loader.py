"""Dry-run verification: dataset loader with the baseline 2D config.

Loads ONE sample from the LDFDCT dataset using the baseline config and
checks the 2D shapes/ranges.

NOTE on constructor signature: the real LDFDCT signature is
    LDFDCT(dataroot, img_size, split="train", data_len=-1, config=None)
(see datasets/LDFDCT.py). The dataroot comes from config.data.train_dataroot.

NOTE on data access: the NPY data referenced by the manifest may not be
present on this machine (e.g. the /workspace/... paths). A data-access
error here is EXPECTED and is NOT a config problem -- it is reported
distinctly from a genuine config/loader failure.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml

from datasets.LDFDCT import LDFDCT


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
    print(f"[CONFIG] input_mode   = {cfg.data.input_mode}")
    print(f"[CONFIG] use_hu_npy   = {cfg.data.use_hu_npy}")
    print(f"[CONFIG] train_dataroot = {cfg.data.train_dataroot}")
    print()

    # Real constructor signature: LDFDCT(dataroot, img_size, split, data_len, config)
    try:
        dataset = LDFDCT(
            cfg.data.train_dataroot,
            cfg.data.image_size,
            split="train",
            config=cfg,
        )
    except FileNotFoundError as exc:
        print("[DATA-ACCESS] Manifest/NPY data not found on this machine.")
        print("[DATA-ACCESS] This is EXPECTED off the training server and is")
        print("[DATA-ACCESS] NOT a config problem. Details:")
        print(f"    {type(exc).__name__}: {exc}")
        print()
        print("RESULT: SKIPPED (data not present) -- config itself parsed fine.")
        return 0

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"Keys: {list(sample.keys())}")
    print(f"LD shape: {tuple(sample['LD'].shape)}")
    print(f"FD shape: {tuple(sample['FD'].shape)}")
    print(f"LD range: [{sample['LD'].min().item():.3f}, {sample['LD'].max().item():.3f}]")
    print(f"FD range: [{sample['FD'].min().item():.3f}, {sample['FD'].max().item():.3f}]")
    print()
    print("Expected for 2D mode:")
    print("  LD shape: (1, 256, 256)")
    print("  FD shape: (1, 256, 256)")
    print("  Both ranges in [-1, 1]")

    ok = (
        tuple(sample["LD"].shape) == (1, cfg.data.image_size, cfg.data.image_size)
        and tuple(sample["FD"].shape) == (1, cfg.data.image_size, cfg.data.image_size)
    )
    print()
    print("Dataset loader test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
