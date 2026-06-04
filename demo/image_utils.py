"""
Shared image utility functions for the thesis demo app.

Used by both PNG Original mode and HU-NPY/DICOM 2.5D mode.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def to_uint8(arr_minus1_1: np.ndarray) -> np.ndarray:
    """Convert a [-1, 1] float array to [0, 255] uint8."""
    arr_01 = np.clip((arr_minus1_1 + 1.0) / 2.0, 0.0, 1.0)
    return (arr_01 * 255.0).round().astype(np.uint8)


def to_display_image(arr_minus1_1: np.ndarray) -> Image.Image:
    """Convert a [-1, 1] float array (H, W) to a PIL grayscale Image."""
    return Image.fromarray(to_uint8(arr_minus1_1), mode="L")


def arr_01_to_display(arr_01: np.ndarray) -> Image.Image:
    """Convert a [0, 1] float array (H, W) to a PIL grayscale Image."""
    arr_u8 = (np.clip(arr_01, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr_u8, mode="L")


def _get_font(size: int = 16):
    """Try to load a TrueType font; fall back to Pillow default."""
    for path in [
        "arial.ttf",
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def create_comparison_figure(
    images: list,
    titles: list,
    pad: int = 6,
    label_h: int = 28,
) -> Image.Image:
    """
    Create a horizontal comparison panel from multiple grayscale images.

    Args:
        images: list of np.ndarray in [-1, 1] or PIL Images
        titles: list of str labels for each panel
        pad: pixel gap between panels
        label_h: pixel height reserved for labels

    Returns:
        PIL Image with labeled panels side by side.
    """
    panels = []
    for img in images:
        if isinstance(img, np.ndarray):
            panels.append(to_uint8(img))
        elif isinstance(img, Image.Image):
            panels.append(np.array(img.convert("L")))
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    h, w = panels[0].shape[:2]
    n = len(panels)
    canvas_w = n * w + (n - 1) * pad
    canvas_h = h + label_h

    canvas = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(canvas)
    font = _get_font(16)

    for i, (title, arr) in enumerate(zip(titles, panels)):
        x_off = i * (w + pad)
        panel_img = Image.fromarray(arr, mode="L")
        canvas.paste(panel_img, (x_off, label_h))

        # Center the title above the panel
        try:
            bbox = draw.textbbox((0, 0), title, font=font)
            tw = bbox[2] - bbox[0]
        except AttributeError:
            tw = len(title) * 8
        tx = x_off + (w - tw) // 2
        draw.text((tx, 4), title, fill=255, font=font)

    return canvas
