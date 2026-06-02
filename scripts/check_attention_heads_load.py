"""
check_attention_heads_load.py
=============================
Kiểm tra xem đổi num_heads trong AttnBlock có làm thay đổi weight shape hay không.
Nếu checkpoint 400k load được strict=True → có thể fine-tune trực tiếp.

Usage (chạy từ repo root):
    python scripts/check_attention_heads_load.py --ckpt_path <path_to_ckpt_400000.pth>

Nếu không có checkpoint file, script sẽ chỉ kiểm tra model init + forward pass.
"""

import argparse
import sys
import os

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from argparse import Namespace


def make_config(num_heads=1):
    """Tạo config namespace giống hệt ldfd_v3_2p5d_5090_full.yml"""
    config = Namespace(
        data=Namespace(
            dataset="LDFDCT",
            image_size=256,
            channels=1,
            input_mode="2.5d",
            condition_mode="2.5d",
            num_condition_slices=3,
        ),
        model=Namespace(
            type="sg",
            in_channels=4,
            out_ch=1,
            ch=128,
            ch_mult=[1, 1, 2, 2, 4, 4],
            num_res_blocks=2,
            attn_resolutions=[16],
            dropout=0.0,
            var_type="fixedsmall",
            ema_rate=0.999,
            ema=True,
            resamp_with_conv=True,
            num_heads=num_heads,
        ),
        diffusion=Namespace(
            beta_schedule="linear",
            beta_start=0.0001,
            beta_end=0.02,
            num_diffusion_timesteps=1000,
        ),
    )
    return config


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def count_attn_blocks(model):
    """Đếm tất cả AttnBlock trong model."""
    from models.diffusion import AttnBlock
    count = 0
    for m in model.modules():
        if isinstance(m, AttnBlock):
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Check attention heads checkpoint compatibility")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to ckpt_400000.pth. If not provided, only model init/forward test is run.",
    )
    parser.add_argument(
        "--new_heads",
        type=int,
        default=4,
        help="Number of attention heads to test (default: 4)",
    )
    args = parser.parse_args()

    device = torch.device("cpu")  # test trên CPU để không cần GPU
    print("=" * 70)
    print("ATTENTION HEADS CHECKPOINT COMPATIBILITY TEST")
    print("=" * 70)

    # =========================================================
    # STEP 1: Tạo model gốc (num_heads=1)
    # =========================================================
    print("\n[STEP 1] Creating ORIGINAL model (num_heads=1)...")
    from models.diffusion import Model

    config_orig = make_config(num_heads=1)
    model_orig = Model(config_orig).to(device)
    n_params_orig = count_params(model_orig)
    n_attn_orig = count_attn_blocks(model_orig)

    print(f"  Total parameters: {n_params_orig:,}")
    print(f"  AttnBlock count:  {n_attn_orig}")
    print(f"  num_heads:        1 (original)")

    # =========================================================
    # STEP 2: Load checkpoint vào model gốc (nếu có)
    # =========================================================
    ckpt_states = None
    if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
        print(f"\n[STEP 2] Loading checkpoint: {args.ckpt_path}")
        ckpt_states = torch.load(args.ckpt_path, map_location="cpu")

        # Xác định state_dict từ checkpoint format
        if isinstance(ckpt_states, (list, tuple)):
            state_dict = ckpt_states[0]
            print(f"  Checkpoint format: list/tuple with {len(ckpt_states)} elements")
            if len(ckpt_states) >= 4:
                print(f"  Checkpoint step: {ckpt_states[3]}")
        elif isinstance(ckpt_states, dict):
            if "model" in ckpt_states:
                state_dict = ckpt_states["model"]
            else:
                state_dict = ckpt_states
        else:
            print(f"  ERROR: Unknown checkpoint format: {type(ckpt_states)}")
            return

        # Strip "module." prefix
        clean_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                clean_state_dict[k.replace("module.", "", 1)] = v
            else:
                clean_state_dict[k] = v

        try:
            model_orig.load_state_dict(clean_state_dict, strict=True)
            print("  ✅ strict=True load SUCCESS on original model (num_heads=1)")
        except RuntimeError as e:
            print(f"  ❌ strict=True load FAILED on original model: {e}")
            return
    else:
        if args.ckpt_path is not None:
            print(f"\n[STEP 2] WARNING: Checkpoint file not found: {args.ckpt_path}")
        else:
            print("\n[STEP 2] No checkpoint provided. Skipping checkpoint load test.")
        print("  Will only test model init and forward pass.")

    # =========================================================
    # STEP 3: Forward pass test cho model gốc
    # =========================================================
    print(f"\n[STEP 3] Forward pass test (original model, num_heads=1)...")
    model_orig.eval()
    with torch.no_grad():
        x_dummy = torch.randn(1, 4, 256, 256, device=device)  # in_channels=4
        t_dummy = torch.randint(0, 1000, (1,), device=device)
        out = model_orig(x_dummy, t_dummy)
        print(f"  Input shape:  {x_dummy.shape}")
        print(f"  Output shape: {out.shape}")
        assert out.shape == (1, 1, 256, 256), f"Unexpected output shape: {out.shape}"
        print("  ✅ Forward pass OK")

    # =========================================================
    # STEP 4: Tạo model mới (num_heads=N)
    # =========================================================
    new_heads = args.new_heads
    print(f"\n[STEP 4] Creating NEW model (num_heads={new_heads})...")
    config_new = make_config(num_heads=new_heads)
    model_new = Model(config_new).to(device)
    n_params_new = count_params(model_new)
    n_attn_new = count_attn_blocks(model_new)

    print(f"  Total parameters: {n_params_new:,}")
    print(f"  AttnBlock count:  {n_attn_new}")
    print(f"  num_heads:        {new_heads}")

    # So sánh params
    if n_params_orig == n_params_new:
        print(f"\n  ✅ Parameter count UNCHANGED: {n_params_orig:,} == {n_params_new:,}")
    else:
        print(f"\n  ❌ Parameter count CHANGED: {n_params_orig:,} != {n_params_new:,}")
        print(f"     Diff: {n_params_new - n_params_orig:+,}")

    # =========================================================
    # STEP 5: So sánh state_dict keys và shapes
    # =========================================================
    print(f"\n[STEP 5] Comparing state_dict keys and shapes...")
    sd_orig = model_orig.state_dict()
    sd_new = model_new.state_dict()

    keys_orig = set(sd_orig.keys())
    keys_new = set(sd_new.keys())

    missing_in_new = keys_orig - keys_new
    extra_in_new = keys_new - keys_orig

    if missing_in_new:
        print(f"  ⚠️  Keys in original but MISSING in new: {missing_in_new}")
    if extra_in_new:
        print(f"  ⚠️  Keys in new but NOT in original: {extra_in_new}")
    if not missing_in_new and not extra_in_new:
        print(f"  ✅ Keys MATCH perfectly ({len(keys_orig)} keys)")

    # Check shape mismatches
    shape_mismatches = []
    for key in keys_orig & keys_new:
        if sd_orig[key].shape != sd_new[key].shape:
            shape_mismatches.append((key, sd_orig[key].shape, sd_new[key].shape))

    if shape_mismatches:
        print(f"\n  ❌ Shape MISMATCHES found ({len(shape_mismatches)}):")
        for key, shape_o, shape_n in shape_mismatches:
            print(f"     {key}: {shape_o} -> {shape_n}")
    else:
        print(f"  ✅ All weight shapes MATCH")

    # =========================================================
    # STEP 6: Load checkpoint vào model mới (nếu có)
    # =========================================================
    if ckpt_states is not None:
        print(f"\n[STEP 6] Loading checkpoint into NEW model (num_heads={new_heads})...")

        try:
            model_new.load_state_dict(clean_state_dict, strict=True)
            print(f"  ✅ strict=True load SUCCESS!")
            print(f"  → Có thể fine-tune TRỰC TIẾP từ checkpoint 400k với num_heads={new_heads}")
        except RuntimeError as e:
            err_msg = str(e)
            print(f"  ❌ strict=True load FAILED:")

            # Parse error for details
            if "size mismatch" in err_msg:
                print(f"  → SIZE MISMATCH detected. KHÔNG thể fine-tune trực tiếp.")
                # Extract mismatched keys
                import re
                mismatches = re.findall(r'"([^"]+)".*?(\([^)]+\)).*?(\([^)]+\))', err_msg)
                for key, expected, got in mismatches:
                    print(f"     Key: {key}, checkpoint: {expected}, model: {got}")
                print(f"\n  ⛔ STOP: Không train. Size mismatch nghiêm trọng.")
            elif "Missing key" in err_msg or "Unexpected key" in err_msg:
                print(f"  → Missing/Unexpected keys. Thử strict=False...")
                try:
                    result = model_new.load_state_dict(clean_state_dict, strict=False)
                    print(f"  strict=False result:")
                    if result.missing_keys:
                        print(f"    Missing keys: {result.missing_keys}")
                    if result.unexpected_keys:
                        print(f"    Unexpected keys: {result.unexpected_keys}")
                    print(f"  → Có thể partial load, nhưng cần xem xét kỹ.")
                except RuntimeError as e2:
                    print(f"  ❌ strict=False cũng FAILED: {e2}")
            else:
                print(f"  Error: {err_msg}")
    else:
        print(f"\n[STEP 6] Skipped (no checkpoint)")

    # =========================================================
    # STEP 7: Forward pass test cho model mới
    # =========================================================
    print(f"\n[STEP 7] Forward pass test (new model, num_heads={new_heads})...")
    model_new.eval()
    with torch.no_grad():
        x_dummy = torch.randn(1, 4, 256, 256, device=device)
        t_dummy = torch.randint(0, 1000, (1,), device=device)
        out_new = model_new(x_dummy, t_dummy)
        print(f"  Input shape:  {x_dummy.shape}")
        print(f"  Output shape: {out_new.shape}")
        assert out_new.shape == (1, 1, 256, 256), f"Unexpected output shape: {out_new.shape}"
        print("  ✅ Forward pass OK")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Original num_heads:    1")
    print(f"  New num_heads:         {new_heads}")
    print(f"  Parameters (original): {n_params_orig:,}")
    print(f"  Parameters (new):      {n_params_new:,}")
    print(f"  Parameter change:      {'NONE ✅' if n_params_orig == n_params_new else 'CHANGED ❌'}")
    print(f"  Shape mismatches:      {'NONE ✅' if not shape_mismatches else f'{len(shape_mismatches)} found ❌'}")
    print(f"  Key mismatches:        {'NONE ✅' if (not missing_in_new and not extra_in_new) else 'FOUND ❌'}")

    if ckpt_states is not None:
        print(f"  Checkpoint strict load: See STEP 6 above")
    else:
        print(f"  Checkpoint load:       NOT TESTED (no checkpoint provided)")

    print(f"  Forward pass (orig):   ✅")
    print(f"  Forward pass (new):    ✅")

    can_finetune = (n_params_orig == n_params_new and not shape_mismatches
                    and not missing_in_new and not extra_in_new)
    if can_finetune:
        print(f"\n  🎯 CONCLUSION: num_heads={new_heads} KHÔNG thay đổi weight shape.")
        print(f"     → Fine-tune trực tiếp từ checkpoint 400k là KHẢ THI.")
    else:
        print(f"\n  ⛔ CONCLUSION: num_heads={new_heads} thay đổi model. Cần xem xét.")

    print("=" * 70)


if __name__ == "__main__":
    main()
