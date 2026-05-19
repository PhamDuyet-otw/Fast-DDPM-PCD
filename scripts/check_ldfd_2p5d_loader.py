#!/usr/bin/env python3
"""
Check script for 2.5D LDFD loader.
Loads dataset from config and prints tensor shapes and statistics.
Supports both Schema A and Schema B manifest formats.
"""

import argparse
import sys
import yaml
from pathlib import Path
from easydict import EasyDict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets.LDFDCT import LDFDCT


def main():
    parser = argparse.ArgumentParser(description="Check 2.5D LDFD loader")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--split", default="train", help="Dataset split (train/val/test)")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to check")
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = EasyDict(config_dict)

    # Get the appropriate manifest path
    manifest_key = f"{args.split}_manifest"
    if not hasattr(config.data, manifest_key):
        print(f"Error: Config does not have '{manifest_key}'")
        sys.exit(1)

    manifest_path = getattr(config.data, manifest_key)
    print(f"Loading dataset from: {manifest_path}")
    print(f"Input mode: {getattr(config.data, 'input_mode', '2d')}")
    print(f"Split: {args.split}")

    # Create dataset
    try:
        dataset = LDFDCT(
            manifest_path,
            config.data.image_size,
            split=args.split,
            config=config
        )
    except Exception as e:
        print(f"Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print(f"\nDataset created successfully with {len(dataset)} samples")
    print()

    # Check samples
    num_samples = min(args.num_samples, len(dataset))
    for idx in range(num_samples):
        print(f"{'='*70}")
        print(f"Sample {idx}")
        print(f"{'='*70}")

        try:
            sample = dataset[idx]
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Print sample info
        print(f"case_name: {sample['case_name']}")

        if "patient_id" in sample:
            print(f"patient_id: {sample['patient_id']}")
        if "slice_idx" in sample:
            print(f"slice_idx: {sample['slice_idx']}")
        if "neighbor_slice_indices" in sample:
            print(f"neighbor_slice_indices: {sample['neighbor_slice_indices']}")

        # Print tensor information
        ld = sample["LD"]
        fd = sample["FD"]

        print(f"\nLD (condition) tensor:")
        print(f"  Shape: {ld.shape}")
        print(f"  Dtype: {ld.dtype}")
        print(f"  Min: {float(ld.min()):.6f}")
        print(f"  Max: {float(ld.max()):.6f}")
        print(f"  Mean: {float(ld.mean()):.6f}")

        print(f"\nFD (target) tensor:")
        print(f"  Shape: {fd.shape}")
        print(f"  Dtype: {fd.dtype}")
        print(f"  Min: {float(fd.min()):.6f}")
        print(f"  Max: {float(fd.max()):.6f}")
        print(f"  Mean: {float(fd.mean()):.6f}")

        # Verify expected shapes for 2.5D mode
        input_mode = getattr(config.data, "input_mode", "2d")
        if input_mode == "2.5d":
            expected_ld_shape = (3, config.data.image_size, config.data.image_size)
            expected_fd_shape = (1, config.data.image_size, config.data.image_size)

            if ld.shape != expected_ld_shape:
                print(f"\n  WARNING: Expected LD shape {expected_ld_shape}, got {ld.shape}")
            else:
                print(f"  ✓ LD shape is correct")

            if fd.shape != expected_fd_shape:
                print(f"\n  WARNING: Expected FD shape {expected_fd_shape}, got {fd.shape}")
            else:
                print(f"  ✓ FD shape is correct")

        # Verify value range
        if float(ld.min()) < -1.1 or float(ld.max()) > 1.1:
            print(f"\n  WARNING: LD values outside [-1, 1] range")
        else:
            print(f"  ✓ LD values in [-1, 1] range")

        if float(fd.min()) < -1.1 or float(fd.max()) > 1.1:
            print(f"\n  WARNING: FD values outside [-1, 1] range")
        else:
            print(f"  ✓ FD values in [-1, 1] range")

        print()

    print(f"{'='*70}")
    print("Check completed successfully!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
