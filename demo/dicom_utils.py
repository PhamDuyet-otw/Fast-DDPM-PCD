"""
DICOM Processing Utilities for Mode 2 (HU-NPY / DICOM 2.5D Mode).

Handles:
    - Extracting DICOM series from ZIP files
    - Converting DICOM pixel data to HU arrays
    - Sorting slices by position/instance number
    - Exporting derived DICOM series with denoised data

WARNING: This is a research prototype. NOT for clinical diagnosis.
"""

import os
import sys
import zipfile
import tempfile
import uuid
from pathlib import Path
from datetime import datetime

import numpy as np

# pydicom is optional — gracefully handle missing dependency
try:
    import pydicom
    from pydicom.uid import generate_uid
    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False


def check_pydicom():
    """Check if pydicom is available and raise helpful error if not."""
    if not HAS_PYDICOM:
        raise ImportError(
            "pydicom is required for DICOM processing.\n"
            "Install it with: pip install pydicom\n"
            "On Colab: !pip install pydicom"
        )


# ============================================================================
# DICOM extraction
# ============================================================================

def extract_dicom_zip(zip_path: str, extract_dir: str = None) -> str:
    """
    Extract a DICOM ZIP file to a temporary directory.

    Args:
        zip_path: path to ZIP file containing DICOM files.
        extract_dir: optional output directory. If None, uses a temp dir.

    Returns:
        Path to the directory containing extracted DICOM files.
    """
    if not os.path.isfile(zip_path):
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")

    if extract_dir is None:
        extract_dir = tempfile.mkdtemp(prefix="dicom_extract_")

    os.makedirs(extract_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)

    # Find the directory containing actual DICOM files
    # (ZIP might have a subdirectory)
    dicom_dir = _find_dicom_dir(extract_dir)
    print(f"[DICOM] Extracted to: {dicom_dir}")
    return dicom_dir


def _find_dicom_dir(root_dir: str) -> str:
    """
    Find the directory containing DICOM files, handling nested ZIPs.
    """
    # Check if current dir has DICOM files
    dcm_extensions = {".dcm", ".ima", ".dicom", ""}
    for f in os.listdir(root_dir):
        fpath = os.path.join(root_dir, f)
        if os.path.isfile(fpath):
            _, ext = os.path.splitext(f.lower())
            if ext in dcm_extensions:
                try:
                    pydicom.dcmread(fpath, stop_before_pixels=True)
                    return root_dir
                except Exception:
                    continue

    # Check subdirectories (one level)
    for d in os.listdir(root_dir):
        dpath = os.path.join(root_dir, d)
        if os.path.isdir(dpath):
            for f in os.listdir(dpath):
                fpath = os.path.join(dpath, f)
                if os.path.isfile(fpath):
                    try:
                        pydicom.dcmread(fpath, stop_before_pixels=True)
                        return dpath
                    except Exception:
                        continue

    return root_dir


# ============================================================================
# DICOM loading and HU conversion
# ============================================================================

def dicom_to_hu_array(ds) -> np.ndarray:
    """
    Convert a pydicom Dataset to a Hounsfield Unit array.

    HU = pixel_array * RescaleSlope + RescaleIntercept

    Args:
        ds: pydicom Dataset with pixel_array.

    Returns:
        float32 numpy array [H, W] in HU values.
    """
    pixel_array = ds.pixel_array.astype(np.float32)

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    hu = pixel_array * slope + intercept
    return hu.astype(np.float32)


def load_dicom_series(dicom_dir: str) -> list:
    """
    Load all DICOM files from a directory, sort by position, convert to HU.

    Sorting priority:
        1. ImagePositionPatient[2] (z-coordinate) if available
        2. InstanceNumber as fallback

    Args:
        dicom_dir: directory containing DICOM files.

    Returns:
        List of dicts sorted by slice position:
        [{
            "ds": pydicom Dataset,
            "hu_array": np.ndarray [H, W] float32,
            "position_z": float,
            "instance_number": int,
            "file_path": str,
        }, ...]
    """
    check_pydicom()

    if not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    slices = []
    for fname in os.listdir(dicom_dir):
        fpath = os.path.join(dicom_dir, fname)
        if not os.path.isfile(fpath):
            continue

        try:
            ds = pydicom.dcmread(fpath)
            if not hasattr(ds, "pixel_array"):
                continue
        except Exception:
            continue

        # Get position for sorting
        position_z = 0.0
        if hasattr(ds, "ImagePositionPatient"):
            try:
                position_z = float(ds.ImagePositionPatient[2])
            except (IndexError, TypeError, ValueError):
                pass

        instance_number = int(getattr(ds, "InstanceNumber", 0))

        hu_array = dicom_to_hu_array(ds)

        slices.append({
            "ds": ds,
            "hu_array": hu_array,
            "position_z": position_z,
            "instance_number": instance_number,
            "file_path": fpath,
        })

    if not slices:
        raise ValueError(f"No valid DICOM files found in: {dicom_dir}")

    # Sort by ImagePositionPatient[z], fall back to InstanceNumber
    has_position = any(s["position_z"] != 0.0 for s in slices)
    if has_position:
        slices.sort(key=lambda s: s["position_z"])
        print(f"[DICOM] Sorted {len(slices)} slices by ImagePositionPatient[z]")
    else:
        slices.sort(key=lambda s: s["instance_number"])
        print(f"[DICOM] Sorted {len(slices)} slices by InstanceNumber")

    return slices


def normalize_hu_to_minus1_1(
    hu_array: np.ndarray,
    hu_min: float = -1000,
    hu_max: float = 400,
) -> np.ndarray:
    """
    Apply HU windowing and normalize to [-1, 1].

    Same transform as Dataset V3 training:
        x = 2 * (clip(HU, hu_min, hu_max) - hu_min) / (hu_max - hu_min) - 1

    Args:
        hu_array: raw HU values [H, W].
        hu_min: lower window bound.
        hu_max: upper window bound.

    Returns:
        float32 array [H, W] in [-1, 1].
    """
    arr = np.clip(hu_array, hu_min, hu_max).astype(np.float32)
    arr = (arr - hu_min) / (hu_max - hu_min)
    arr = arr * 2.0 - 1.0
    return arr


# ============================================================================
# DICOM series to 2.5D triplets
# ============================================================================

def dicom_series_to_triplets(
    dicom_slices: list,
    hu_min: float = -1000,
    hu_max: float = 400,
) -> list:
    """
    Convert a loaded DICOM series to 2.5D triplet samples.

    Each valid center slice (with both neighbors) becomes one sample.

    Args:
        dicom_slices: sorted list from load_dicom_series().
        hu_min, hu_max: HU window parameters.

    Returns:
        List of dicts:
        [{
            "slice_idx": int (center index in the series),
            "ld_triplet": np.ndarray [3, H, W] in [-1, 1],
            "display_name": str,
        }, ...]
    """
    n = len(dicom_slices)
    if n < 3:
        raise ValueError(
            f"Need at least 3 slices for 2.5D inference, got {n}.\n"
            "Cannot build triplets from fewer than 3 DICOM slices."
        )

    # Normalize all slices
    normalized = []
    for s in dicom_slices:
        normalized.append(normalize_hu_to_minus1_1(s["hu_array"], hu_min, hu_max))

    triplets = []
    for center in range(1, n - 1):
        triplet = np.stack([
            normalized[center - 1],
            normalized[center],
            normalized[center + 1],
        ], axis=0)  # [3, H, W]

        triplets.append({
            "slice_idx": center,
            "ld_triplet": triplet,
            "display_name": f"slice_{center:04d}",
        })

    print(f"[DICOM] Built {len(triplets)} 2.5D triplets from {n} slices")
    return triplets


# ============================================================================
# Derived DICOM export
# ============================================================================

def export_derived_dicom_series(
    original_slices: list,
    denoised_arrays: dict,
    output_dir: str,
    hu_min: float = -1000,
    hu_max: float = 400,
) -> str:
    """
    Export denoised results as a new derived DICOM series.

    Does NOT overwrite original DICOM files.

    Args:
        original_slices: list from load_dicom_series() — the original DICOM data.
        denoised_arrays: dict mapping slice_idx → denoised array [H, W] in [-1, 1].
        output_dir: directory for output DICOM files.
        hu_min, hu_max: for reverse normalization.

    Returns:
        Path to the output directory containing derived DICOM files.
    """
    check_pydicom()

    os.makedirs(output_dir, exist_ok=True)

    # Generate new series UID
    new_series_uid = generate_uid()
    now_str = datetime.now().strftime("%Y%m%d%H%M%S")

    exported_count = 0
    for slice_idx, denoised_minus1_1 in denoised_arrays.items():
        if slice_idx >= len(original_slices):
            continue

        original_ds = original_slices[slice_idx]["ds"]

        # Create a copy to avoid modifying the original
        new_ds = original_ds.copy()

        # Reverse normalization: [-1, 1] → HU
        denoised_01 = (denoised_minus1_1 + 1.0) / 2.0
        denoised_hu = denoised_01 * (hu_max - hu_min) + hu_min

        # Convert back to pixel values using original rescale
        slope = float(getattr(original_ds, "RescaleSlope", 1.0))
        intercept = float(getattr(original_ds, "RescaleIntercept", 0.0))
        pixel_values = (denoised_hu - intercept) / slope
        pixel_values = np.clip(pixel_values, 0, 65535).astype(np.uint16)

        # Update pixel data
        new_ds.PixelData = pixel_values.tobytes()
        new_ds.Rows = pixel_values.shape[0]
        new_ds.Columns = pixel_values.shape[1]

        # Update UIDs and metadata for derived series
        new_ds.SeriesInstanceUID = new_series_uid
        new_ds.SOPInstanceUID = generate_uid()
        new_ds.ImageType = ["DERIVED", "SECONDARY"]
        new_ds.SeriesDescription = "AI denoised LDCT research output"
        new_ds.DerivationDescription = (
            "Denoised using Fast-DDPM thesis research prototype. "
            "NOT for clinical diagnosis."
        )
        new_ds.SeriesDate = now_str[:8]
        new_ds.SeriesTime = now_str[8:]

        # Preserve geometry metadata
        # (PixelSpacing, SliceThickness, ImageOrientation/Position are kept from original)

        # Save
        output_path = os.path.join(output_dir, f"denoised_{slice_idx:04d}.dcm")
        new_ds.save_as(output_path)
        exported_count += 1

    print(f"[DICOM] Exported {exported_count} derived DICOM files to: {output_dir}")
    return output_dir


def zip_dicom_output(dicom_dir: str, zip_path: str = None) -> str:
    """
    ZIP a directory of DICOM files for download.

    Args:
        dicom_dir: directory containing DICOM files.
        zip_path: output ZIP path. Auto-generated if None.

    Returns:
        Path to the ZIP file.
    """
    if zip_path is None:
        zip_path = dicom_dir.rstrip("/\\") + "_derived.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in sorted(os.listdir(dicom_dir)):
            fpath = os.path.join(dicom_dir, fname)
            if os.path.isfile(fpath):
                z.write(fpath, arcname=fname)

    print(f"[DICOM] Created ZIP: {zip_path}")
    return zip_path
