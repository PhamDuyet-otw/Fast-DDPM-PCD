# Fast-DDPM CT Denoising — Thesis Demo

## Overview

This is a Gradio-based inference demo for the thesis:
**"Low-Dose Lung CT Image Denoising Using Diffusion Models"**

The demo has **two separate tabs**, each using a **different model pipeline and checkpoint**:

| Tab | Mode | Input | Model Channels |
|-----|------|-------|----------------|
| 1 | **PNG Original Fast-DDPM** | 2D grayscale PNG | in_channels=2 |
| 2 | **HU-NPY / DICOM 2.5D** | raw-HU NPY or DICOM | in_channels=4 |

> ⚠️ **Research prototype only. Not for clinical diagnosis.**

---

## Quick Start

### Local (Windows/Linux)

```bash
# Install dependencies
pip install gradio torch torchvision pyyaml scikit-image pillow numpy

# Optional: for DICOM mode
pip install pydicom

# Run the app
cd Fast-DDPM-PCD
python demo/app_gradio.py --device cuda
```

### Google Colab

```python
# Cell 1: Install dependencies
!pip install gradio pydicom scikit-image pyyaml

# Cell 2: Clone and run
!git clone https://github.com/PhamDuyet-otw/Fast-DDPM-PCD.git
%cd Fast-DDPM-PCD
!git checkout demo/gradio-app

# Cell 3: Launch with public URL
!python demo/app_gradio.py --share --device cuda
```

---

## CLI Arguments

```
python demo/app_gradio.py \
    --share              # Create public Gradio link (for Colab)
    --port 7860          # Server port
    --device cuda        # cuda or cpu
    --png_config configs/ldfd_linear.yml        # Default PNG config
    --png_ckpt D:/B3/Thesis/Fast-DDPM/pretrained_models/ckpt_LDFDCT.pth  # Default PNG checkpoint
    --png_sample_folder D:/B3/Thesis/Fast-DDPM/Fast-DDPM/data/LD_FD_CT_test  # Default PNG sample folder
    --v25d_config configs/ldfd_v3_2p5d_5090_full.yml  # Default 2.5D config
    --v25d_ckpt path/to/2p5d_ckpt.pth           # Default 2.5D checkpoint
```

All paths are also configurable in the Gradio UI at runtime.

---

## Mode 1: PNG Original Fast-DDPM

### What it does
- Uses the original-style Fast-DDPM model (same as the original Fast-DDPM paper/repo)
- 2D input/output
- Single LD condition channel
- `in_channels = 2` (1 noisy target + 1 LD condition)
- PNG grayscale images
- No 2.5D, no HU windowing, no Dataset V3 logic

### Required paths
1. **Config**: `configs/ldfd_linear.yml`
2. **Checkpoint**: `D:/B3/Thesis/Fast-DDPM/pretrained_models/ckpt_LDFDCT.pth`
3. **Sample folder**: `D:/B3/Thesis/Fast-DDPM/Fast-DDPM/data/LD_FD_CT_test`

### How to use
1. Enter config and checkpoint paths
2. Click "Load PNG Model"
3. Enter PNG sample folder path and click "Load PNG Samples" to browse LD/FD pairs
4. Select a sample from the dropdown, OR upload images manually
5. Upload an LDCT PNG image
6. Optionally upload an FDCT PNG for metric comparison
7. Click "Run PNG Denoising"

### Sample folder structure

```
The sample folder scanner expects this naming convention:
  {patient_id}_{slice_idx}_ld.png  — low-dose
  {patient_id}_{slice_idx}_fd.png  — full-dose

Example structure:
  LD_FD_CT_test/
    C002/
      C002_0_ld.png
      C002_0_fd.png
      C002_1_ld.png
      C002_1_fd.png
      ...
    C050/
      ...
```

---

## Mode 2: HU-NPY / DICOM 2.5D

### What it does
- Uses the Dataset V3 raw-HU NPY pipeline with 2.5D conditioning
- Condition: LD[z-1], LD[z], LD[z+1] (3 channels)
- Target: FD[z] (1 channel)
- `in_channels = 4` (1 noisy target + 3 condition channels)
- Full 512×512 inference via sliding-window patches (256×256)
- HU windowing: [-1000, 400] → normalize to [-1, 1]

### Input options

**Option A: Dataset V3 NPY manifest**
1. Load a manifest CSV (e.g., `test_v3.csv`)
2. Select a sample from the dropdown
3. Run inference

**Option B: DICOM series**
1. Upload a ZIP containing DICOM CT files
2. The app converts DICOM → HU arrays → 2.5D triplets
3. Select a slice and run inference

### Required paths
1. **Config**: `configs/ldfd_v3_2p5d_5090_full.yml`
2. **Checkpoint**: path to trained 2.5D model `.pth` file
3. **Manifest CSV** (for NPY mode): Dataset V3 test manifest
4. **Data root** (optional): for resolving relative paths in manifest

---

## File Structure

```
demo/
├── app_gradio.py              # Main Gradio app
├── inference_png_original.py  # PNG mode inference logic
├── inference_2p5d_hu.py       # 2.5D HU-NPY mode inference logic
├── dicom_utils.py             # DICOM processing utilities
├── image_utils.py             # Shared image utilities
├── README_demo.md             # This file
└── __init__.py                # Package init
```

---

## Important Notes

### Two separate pipelines
- **PNG mode** and **2.5D mode** use **different models** with **different checkpoints**
- Do NOT use the 2.5D checkpoint for PNG mode
- Do NOT use the PNG checkpoint for 2.5D mode
- Each tab has its own model state — they never interfere

### Concatenation order
The model input concatenation is: `[x_t, condition]` (noisy target first, then condition).
This is handled internally by `sg_generalized_steps`.

### Metrics
- Computed only when ground truth (FDCT) is available
- PNG mode: PSNR/SSIM on [0, 1] scale, data_range=1.0
- 2.5D mode: PSNR/SSIM on [0, 1] scale (after converting from [-1, 1])
- DICOM mode: no ground truth → no metrics

### DICOM export
- Not yet implemented in the Gradio UI (planned for future)
- The `dicom_utils.py` module has export functions ready
- Any exported DICOM will have new SeriesInstanceUID/SOPInstanceUID
- Metadata clearly marks output as "DERIVED" / "SECONDARY"

---

## Dependencies

| Package | Required for | Install |
|---------|-------------|---------|
| gradio | UI | `pip install gradio` |
| torch | Model inference | `pip install torch` |
| pyyaml | Config loading | `pip install pyyaml` |
| scikit-image | PSNR/SSIM | `pip install scikit-image` |
| pillow | Image handling | `pip install pillow` |
| numpy | Array ops | `pip install numpy` |
| pydicom | DICOM mode (optional) | `pip install pydicom` |

---

## Verification Checklist

- [ ] PNG model loads with correct config (in_channels=2)
- [ ] 2.5D model loads with correct config (in_channels=4)
- [ ] PNG denoising runs on uploaded LDCT PNG
- [ ] Denoised PNG output is displayed
- [ ] PSNR/SSIM computed when FDCT PNG is provided
- [ ] 2.5D manifest loads and shows samples in dropdown
- [ ] 2.5D NPY denoising runs on selected sample
- [ ] 2.5D LD/Denoised/FD previews displayed
- [ ] DICOM ZIP upload and processing works
- [ ] Both tabs use separate model states
- [ ] Clear error messages when paths are missing
