"""
preprocess.py - M1 demo input preprocessing.
"""
import os
import numpy as np
import torch
from PIL import Image


HU_WINDOW_MIN = -1000.0
HU_WINDOW_MAX = 400.0


def normalize_hu_to_minus_one_one(hu_array):
    arr = np.clip(hu_array, HU_WINDOW_MIN, HU_WINDOW_MAX)
    arr = (arr - HU_WINDOW_MIN) / (HU_WINDOW_MAX - HU_WINDOW_MIN)
    arr = arr * 2.0 - 1.0
    return arr.astype(np.float32)


def denormalize_to_uint8(arr_minus_one_one):
    if isinstance(arr_minus_one_one, torch.Tensor):
        arr = arr_minus_one_one.detach().cpu().numpy()
    else:
        arr = arr_minus_one_one
    arr = np.clip((arr + 1.0) / 2.0, 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def read_npy_to_hu(path):
    arr = np.load(path).astype(np.float32)
    if arr.ndim != 2:
        raise ValueError(f"NPY must be 2D, got shape {arr.shape}")
    return arr


def read_dicom_to_hu(path):
    try:
        import pydicom
    except ImportError:
        raise RuntimeError("pydicom required. pip install pydicom")
    ds = pydicom.dcmread(path)
    pixel = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixel * slope + intercept


def read_png_to_fake_hu(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr * (HU_WINDOW_MAX - HU_WINDOW_MIN) + HU_WINDOW_MIN


def load_input_as_tensor(filepath, target_size=512):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".npy":
        hu = read_npy_to_hu(filepath)
        fmt = "NPY"
        warning = None
    elif ext in (".dcm", ".dicom"):
        hu = read_dicom_to_hu(filepath)
        fmt = "DICOM"
        warning = None
    elif ext in (".png", ".jpg", ".jpeg"):
        hu = read_png_to_fake_hu(filepath)
        fmt = "PNG"
        warning = ("PNG input uses approximated HU. For best quality, use .npy or .dcm.")
    else:
        raise ValueError(f"Unsupported format: {ext}")

    original_shape = hu.shape
    hu_range = (float(hu.min()), float(hu.max()))

    if hu.shape != (target_size, target_size):
        hu_min, hu_max = hu.min(), hu.max()
        denom = (hu_max - hu_min) if hu_max > hu_min else 1.0
        hu_norm = (hu - hu_min) / denom
        img = Image.fromarray((hu_norm * 255).astype(np.uint8))
        img = img.resize((target_size, target_size), Image.BICUBIC)
        hu_resized = np.array(img, dtype=np.float32) / 255.0
        hu = hu_resized * denom + hu_min

    arr = normalize_hu_to_minus_one_one(hu)
    arr_25d = np.stack([arr, arr, arr], axis=0)
    tensor = torch.from_numpy(arr_25d).unsqueeze(0)

    info = {
        "format": fmt,
        "warning": warning,
        "original_shape": original_shape,
        "hu_range": hu_range,
        "tensor_shape": tuple(tensor.shape),
    }
    return tensor, info


def hu_to_display_image(hu_array, size=512):
    arr = np.clip(hu_array, HU_WINDOW_MIN, HU_WINDOW_MAX)
    arr = (arr - HU_WINDOW_MIN) / (HU_WINDOW_MAX - HU_WINDOW_MIN)
    arr = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(arr)
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        tensor, info = load_input_as_tensor(path)
        print(f"[TEST] Loaded: {path}")
        print(f"  Format: {info['format']}")
        print(f"  Original shape: {info['original_shape']}")
        print(f"  Original HU range: {info['hu_range']}")
        print(f"  Output tensor shape: {info['tensor_shape']}")
        print(f"  Output range: [{tensor.min():.3f}, {tensor.max():.3f}]")
        if info['warning']:
            print(f"  WARNING: {info['warning']}")
