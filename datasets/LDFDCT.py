import os
import re
import csv
import random
import glob
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image

from .sr_util import get_paths_from_images, transform_augment


def _is_windows_abs_path(p):
    return len(p) >= 3 and p[1] == ":" and p[2] in ["\\", "/"]


def _normalize_path_str(p):
    return str(p).strip().strip('"').strip("'").replace("\\", "/")


def _dose_from_filename(path):
    name = os.path.basename(path).lower()
    if "_ld" in name or name.endswith("ld.npy"):
        return "low"
    if "_fd" in name or name.endswith("fd.npy"):
        return "full"
    return ""


@lru_cache(maxsize=200000)
def _resolve_npy_path(path_str, manifest_dir, split, patient_id):
    """
    Resolve path trong manifest.

    Manifest hiện tại có path Windows:
        D:/B3/Thesis/LDCT_DATASET_V2/02_processed_npy_256/C002/C002_0_ld.npy

    Trên Colab thường nằm ở:
        /content/FastDDPM_Data/02_processed_npy_256/C002/C002_0_ld.npy
    hoặc:
        /content/FastDDPM_Data/02_processed_npy_256_split/train/...
    """

    p = _normalize_path_str(path_str)
    manifest_dir = _normalize_path_str(manifest_dir)
    split = str(split).strip().lower()
    patient_id = str(patient_id).strip()

    if p == "":
        raise ValueError("Empty npy path in manifest.")

    # 1. Nếu path gốc tồn tại thì dùng luôn
    if os.path.exists(p):
        return p

    # 2. Root ứng viên
    # manifest thường: /content/FastDDPM_Data/manifests/train.csv
    # data root sẽ là: /content/FastDDPM_Data
    manifest_parent = os.path.dirname(manifest_dir)
    root_candidates = []

    env_root = os.environ.get("LDCT_DATASET_ROOT", "")
    if env_root:
        root_candidates.append(_normalize_path_str(env_root))

    root_candidates += [
        manifest_parent,
        "/content/FastDDPM_Data",
        "/content/drive/MyDrive/Thesis_Mus",
        "/content/drive/MyDrive/Thesis",
    ]

    # remove duplicate
    root_candidates = list(dict.fromkeys([r for r in root_candidates if r]))

    basename = os.path.basename(p)
    dose = _dose_from_filename(p)

    # 3. Lấy suffix sau LDCT_DATASET_V2 nếu có
    suffixes = []

    marker = "LDCT_DATASET_V2/"
    if marker in p:
        suffixes.append(p.split(marker, 1)[1])

    # Nếu không có marker thì fallback basename/patient
    if "02_processed_npy_256/" in p:
        suffixes.append(p[p.index("02_processed_npy_256/"):])

    # Thêm biến thể split
    more_suffixes = []
    for s in suffixes:
        more_suffixes.append(s)
        more_suffixes.append(s.replace("02_processed_npy_256/", "02_processed_npy_256_split/"))
        more_suffixes.append(s.replace("02_processed_npy_256", "02_processed_npy_256_split"))

    suffixes = list(dict.fromkeys(more_suffixes))

    candidates = []

    for root in root_candidates:
        for s in suffixes:
            candidates.append(os.path.join(root, s))

        # Cấu trúc không split, theo patient
        candidates.append(os.path.join(root, "02_processed_npy_256", patient_id, basename))
        candidates.append(os.path.join(root, "02_processed_npy_256_split", patient_id, basename))

        # Cấu trúc có split/patient
        if split:
            candidates.append(os.path.join(root, "02_processed_npy_256", split, patient_id, basename))
            candidates.append(os.path.join(root, "02_processed_npy_256_split", split, patient_id, basename))

        # Cấu trúc có split/low hoặc split/full
        if split and dose:
            candidates.append(os.path.join(root, "02_processed_npy_256", split, dose, basename))
            candidates.append(os.path.join(root, "02_processed_npy_256_split", split, dose, basename))

            candidates.append(os.path.join(root, "02_processed_npy_256", split, dose, patient_id, basename))
            candidates.append(os.path.join(root, "02_processed_npy_256_split", split, dose, patient_id, basename))

            candidates.append(os.path.join(root, "02_processed_npy_256", split, patient_id, dose, basename))
            candidates.append(os.path.join(root, "02_processed_npy_256_split", split, patient_id, dose, basename))

    for c in candidates:
        c = os.path.normpath(c)
        if os.path.exists(c):
            return c

    # 4. Fallback cuối: tìm basename trong root, có cache nên chỉ tìm 1 lần/path
    for root in root_candidates:
        if os.path.exists(root):
            hits = glob.glob(os.path.join(root, "**", basename), recursive=True)
            if hits:
                # Ưu tiên file nằm trong patient_id và split
                hits_sorted = sorted(
                    hits,
                    key=lambda x: (
                        patient_id not in x,
                        split not in x.lower() if split else False,
                        len(x),
                    )
                )
                return os.path.normpath(hits_sorted[0])

    raise FileNotFoundError(
        "Cannot resolve npy path.\n"
        f"Original path: {path_str}\n"
        f"Manifest dir: {manifest_dir}\n"
        f"Split: {split}\n"
        f"Patient: {patient_id}\n"
        f"Basename: {basename}\n"
        f"Tried roots: {root_candidates}\n"
        "Nếu đang ở Colab, hãy set:\n"
        "os.environ['LDCT_DATASET_ROOT'] = '/content/FastDDPM_Data'\n"
    )


def _parse_slice_idx_from_filename(filename):
    basename = os.path.basename(filename)
    stem = os.path.splitext(basename)[0]
    if "_" in stem:
        parts = stem.split("_")
        for part in reversed(parts):
            if part.isdigit():
                return int(part.lstrip("0") or "0")
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1))
    raise ValueError(f"Cannot parse slice_idx from filename: {filename}")


def _apply_hu_window(arr, min_val, max_val, normalize):
    if normalize == "minus_one_to_one":
        arr = np.clip(arr, min_val, max_val)
        arr = (arr - min_val) / (max_val - min_val)
        arr = arr * 2.0 - 1.0
    elif normalize == "zero_one":
        arr = np.clip(arr, min_val, max_val)
        arr = (arr - min_val) / (max_val - min_val)
    else:
        arr = np.clip(arr, min_val, max_val)
    return arr.astype(np.float32)


def _load_npy_as_tensor(path, img_size, use_hu_npy=False, hu_window_min=-1000, hu_window_max=400, normalize="minus_one_to_one"):
    arr = np.load(path)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D npy image, got shape {arr.shape} at {path}")

    arr = arr.astype(np.float32)

    if use_hu_npy:
        arr = _apply_hu_window(arr, hu_window_min, hu_window_max, normalize)
    else:
        if normalize == "minus_one_to_one":
            arr = np.clip(arr, -1.0, 1.0)

    return torch.from_numpy(arr).unsqueeze(0).float()


def _crop_tensor(tensor, crop_coords):
    if crop_coords is None:
        return tensor
    x1, x2, y1, y2 = crop_coords
    return tensor[:, x1:x2, y1:y2]


def _resize_tensor(tensor, size):
    if tuple(tensor.shape[1:]) == tuple(size):
        return tensor
    return F.interpolate(tensor.unsqueeze(0), size=size, mode='bilinear', align_corners=False).squeeze(0)


class LDFDCT(Dataset):
    """
    Hỗ trợ 2 kiểu:

    1. PNG gốc của Fast-DDPM:
        data/LD_FD_CT_train

    2. Manifest CSV Dataset V2:
        train.csv / val.csv / test.csv

    Manifest Dataset V2 dùng cột:
        low_npy_path, full_npy_path

    Loader cũng hỗ trợ fallback cũ:
        low_path, full_path
    
    3. 2.5D Mode (new):
        Groups slices by patient_id, loads 3 consecutive low-dose slices
        as condition and 1 full-dose slice as target.
    """

    def __init__(self, dataroot, img_size, split="train", data_len=-1, config=None):
        self.dataroot = str(dataroot)
        self.img_size = img_size
        self.split = str(split).lower()
        self.data_len_arg = data_len
        self.config = config

        # Set defaults for 2.5D parameters
        self.input_mode = getattr(config.data, "input_mode", "2d") if config else "2d"
        self.condition_slices = getattr(config.data, "condition_slices", 1) if config else 1
        self.slice_offsets = getattr(config.data, "slice_offsets", [-1, 0, 1]) if config else [-1, 0, 1]
        self.drop_boundary_slices = getattr(config.data, "drop_boundary_slices", True) if config else True
        self.random_flip = getattr(config.data, "random_flip", False) if config else False
        self.train_crop = getattr(config.data, "train_crop", "random") if config else "random"
        self.val_crop = getattr(config.data, "val_crop", "center") if config else "center"
        self.test_crop = getattr(config.data, "test_crop", "center") if config else "center"
        self.use_hu_npy = getattr(config.data, "use_hu_npy", False) if config else False
        self.hu_window_min = getattr(config.data, "hu_window_min", -1000) if config else -1000
        self.hu_window_max = getattr(config.data, "hu_window_max", 400) if config else 400
        self.normalize = getattr(config.data, "normalize", "minus_one_to_one") if config else "minus_one_to_one"
        self.original_size = getattr(config.data, "original_size", img_size) if config else img_size

        self.use_manifest_npy = self.dataroot.lower().endswith(".csv")

        if self.use_manifest_npy:
            self.manifest_path = os.path.normpath(self.dataroot)
            self.manifest_dir = os.path.dirname(self.manifest_path)

            self.all_samples = self._read_manifest(self.manifest_path)

            if data_len is not None and data_len > 0:
                self.all_samples = self.all_samples[:data_len]

            # Handle 2.5D mode: group by patient and filter valid center slices
            if self.input_mode == "2.5d":
                self.samples, self.sample_to_neighbors = self._prepare_25d_samples(self.all_samples)
                self.data_len = len(self.samples)
                print("[LDFDCT] Using 2.5D mode")
                print(f"[LDFDCT] Original pairs: {len(self.all_samples)}, Valid 2.5D triplets: {self.data_len}")
            else:
                self.samples = self.all_samples
                self.data_len = len(self.samples)

            print("[LDFDCT] Using Dataset V2 NPY manifest loader")
            print(f"[LDFDCT] Manifest: {self.manifest_path}")
            print(f"[LDFDCT] Split: {self.split}")
            print(f"[LDFDCT] Input mode: {self.input_mode}")
            print(f"[LDFDCT] Number of samples: {self.data_len}")

        else:
            self.img_ld_path, self.img_fd_path = get_paths_from_images(self.dataroot)

            if data_len is not None and data_len > 0:
                self.img_ld_path = self.img_ld_path[:data_len]
                self.img_fd_path = self.img_fd_path[:data_len]

            self.data_len = len(self.img_ld_path)

            print("[LDFDCT] Using original PNG folder loader")
            print(f"[LDFDCT] Dataroot: {self.dataroot}")
            print(f"[LDFDCT] Number of pairs: {self.data_len}")

    def _read_manifest(self, manifest_path):
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"Manifest CSV not found: {manifest_path}")

        samples = []
        manifest_root = os.path.dirname(self.manifest_dir)

        with open(manifest_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = [fn.strip() for fn in (reader.fieldnames or [])]

            if "Patient_ID" in fieldnames and "LD_Path" in fieldnames and "FD_Path" in fieldnames:
                schema = "B"
                patient_col = "Patient_ID"
                low_col = "LD_Path"
                full_col = "FD_Path"
            elif "low_npy_path" in fieldnames and "full_npy_path" in fieldnames:
                schema = "A"
                patient_col = "patient_id"
                low_col = "low_npy_path"
                full_col = "full_npy_path"
            elif "low_path" in fieldnames and "full_path" in fieldnames:
                schema = "A"
                patient_col = "patient_id"
                low_col = "low_path"
                full_col = "full_path"
            else:
                raise ValueError(
                    f"Manifest columns are: {fieldnames}\n"
                    "Required columns should be either:\n"
                    "  low_npy_path, full_npy_path\n"
                    "or:\n"
                    "  low_path, full_path\n"
                    "or:\n"
                    "  Patient_ID, LD_Path, FD_Path"
                )

            for row in reader:
                row_split = str(row.get("split", "")).strip().lower()

                if row_split and self.split in ["train", "val", "test"]:
                    if row_split != self.split:
                        continue

                patient_id = str(row.get(patient_col, "")).strip()
                low_original = str(row.get(low_col, "")).strip()
                full_original = str(row.get(full_col, "")).strip()

                if schema == "B":
                    if not os.path.isabs(low_original):
                        low_original = os.path.normpath(
                            os.path.join(manifest_root, "01_processed_npy_512", _normalize_path_str(low_original))
                        )
                    if not os.path.isabs(full_original):
                        full_original = os.path.normpath(
                            os.path.join(manifest_root, "01_processed_npy_512", _normalize_path_str(full_original))
                        )

                low_path = low_original
                full_path = full_original

                if schema != "B" or not os.path.exists(low_path):
                    low_path = _resolve_npy_path(
                        low_original,
                        self.manifest_dir,
                        self.split,
                        patient_id,
                    )
                if schema != "B" or not os.path.exists(full_path):
                    full_path = _resolve_npy_path(
                        full_original,
                        self.manifest_dir,
                        self.split,
                        patient_id,
                    )

                if patient_id == "":
                    patient_id = Path(low_path).stem.split("_")[0]

                slice_idx_value = str(row.get("slice_idx", "")).strip()
                if slice_idx_value == "":
                    slice_idx = _parse_slice_idx_from_filename(Path(low_path).name)
                else:
                    try:
                        slice_idx = int(slice_idx_value)
                    except ValueError:
                        slice_idx = _parse_slice_idx_from_filename(Path(low_path).name)

                samples.append(
                    {
                        "low_path": low_path,
                        "full_path": full_path,
                        "patient_id": patient_id,
                        "slice_idx": slice_idx,
                    }
                )

        if len(samples) == 0:
            raise RuntimeError(
                f"No samples loaded from manifest: {manifest_path}. "
                f"Check split column / manifest path."
            )

        return samples

    def _prepare_25d_samples(self, samples):
        """
        Prepare 2.5D samples: group by patient_id, sort by slice_idx.
        Return list of center slice indices and mapping to neighbor indices.
        Drop boundary slices if drop_boundary_slices=True.
        """
        from collections import defaultdict

        # Group samples by patient_id
        patient_groups = defaultdict(list)
        for idx, sample in enumerate(samples):
            patient_id = sample["patient_id"]
            patient_groups[patient_id].append((idx, sample))

        # Sort each patient's slices by slice_idx
        for patient_id in patient_groups:
            try:
                patient_groups[patient_id].sort(
                    key=lambda x: int(x[1]["slice_idx"]) if x[1]["slice_idx"] else 0
                )
            except (ValueError, TypeError):
                # If slice_idx is not an integer, just keep original order
                pass

        # Find valid center slices
        valid_centers = []
        sample_to_neighbors = {}

        for patient_id, slices in patient_groups.items():
            num_slices = len(slices)

            for center_pos in range(num_slices):
                # Check if we can get neighbors at offsets [-1, 0, 1]
                valid = True
                neighbors = {}

                for offset in self.slice_offsets:
                    neighbor_pos = center_pos + offset
                    if neighbor_pos < 0 or neighbor_pos >= num_slices:
                        valid = False
                        break
                    neighbors[offset] = slices[neighbor_pos][0]  # Store original index

                if valid or not self.drop_boundary_slices:
                    if valid:
                        center_idx = slices[center_pos][0]
                        valid_centers.append(center_idx)
                        sample_to_neighbors[center_idx] = neighbors

        return valid_centers, sample_to_neighbors


    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.use_manifest_npy:
            if self.input_mode == "2.5d":
                return self._getitem_25d(index)
            else:
                return self._getitem_2d(index)

        # PNG loader gốc
        base_name = self.img_ld_path[index].split("/")[-1]
        case_name = base_name.split("_")[0]

        img_LD = Image.open(self.img_ld_path[index]).convert("L")
        img_FD = Image.open(self.img_fd_path[index]).convert("L")

        img_LD = img_LD.resize((self.img_size, self.img_size))
        img_FD = img_FD.resize((self.img_size, self.img_size))

        [img_LD, img_FD] = transform_augment(
            [img_LD, img_FD],
            split=self.split,
            min_max=(-1, 1),
        )

        return {
            "FD": img_FD,
            "LD": img_LD,
            "case_name": case_name,
        }

    def _getitem_2d(self, index):
        """Standard 2D loading (single slice)."""
        item = self.samples[index]

        img_LD = _load_npy_as_tensor(
            item["low_path"],
            self.img_size,
            use_hu_npy=self.use_hu_npy,
            hu_window_min=self.hu_window_min,
            hu_window_max=self.hu_window_max,
            normalize=self.normalize,
        )
        img_FD = _load_npy_as_tensor(
            item["full_path"],
            self.img_size,
            use_hu_npy=self.use_hu_npy,
            hu_window_min=self.hu_window_min,
            hu_window_max=self.hu_window_max,
            normalize=self.normalize,
        )

        crop_coords = self._get_crop_coordinates(img_LD.shape[1:])
        if crop_coords is not None:
            img_LD = _crop_tensor(img_LD, crop_coords)
            img_FD = _crop_tensor(img_FD, crop_coords)

        if tuple(img_LD.shape[1:]) != (self.img_size, self.img_size):
            img_LD = _resize_tensor(img_LD, (self.img_size, self.img_size))
            img_FD = _resize_tensor(img_FD, (self.img_size, self.img_size))

        if self.split == "train" and random.random() < 0.5:
            img_LD = torch.flip(img_LD, dims=[2])
            img_FD = torch.flip(img_FD, dims=[2])

        case_name = item["patient_id"]
        if item["slice_idx"] != "":
            case_name = f"{case_name}_{item['slice_idx']}"

        return {
            "FD": img_FD,
            "LD": img_LD,
            "case_name": case_name,
        }

    def _getitem_25d(self, index):
        """
        2.5D loading: load 3 consecutive low-dose slices as condition,
        and 1 full-dose slice as target.

        Returns:
            {
                "FD": torch.Tensor of shape [1, H, W] (target),
                "LD": torch.Tensor of shape [3, H, W] (condition),
                "case_name": str,
                "patient_id": str,
                "slice_idx": int (center slice index),
                "neighbor_slice_indices": dict {offset: slice_idx},
            }
        """
        center_idx = self.samples[index]
        neighbors = self.sample_to_neighbors[center_idx]

        center_item = self.all_samples[center_idx]
        center_patient_id = center_item["patient_id"]
        center_slice_idx = int(center_item["slice_idx"]) if center_item["slice_idx"] != "" else 0

        img_FD = _load_npy_as_tensor(
            center_item["full_path"],
            self.img_size,
            use_hu_npy=self.use_hu_npy,
            hu_window_min=self.hu_window_min,
            hu_window_max=self.hu_window_max,
            normalize=self.normalize,
        )

        ld_slices = []
        neighbor_slice_indices = {}

        for offset in sorted(self.slice_offsets):
            neighbor_item = self.all_samples[neighbors[offset]]
            ld_slice = _load_npy_as_tensor(
                neighbor_item["low_path"],
                self.img_size,
                use_hu_npy=self.use_hu_npy,
                hu_window_min=self.hu_window_min,
                hu_window_max=self.hu_window_max,
                normalize=self.normalize,
            )
            ld_slices.append(ld_slice)
            neighbor_slice_indices[offset] = int(neighbor_item["slice_idx"]) if neighbor_item["slice_idx"] != "" else 0

        img_LD = torch.cat(ld_slices, dim=0)

        crop_coords = self._get_crop_coordinates(img_LD.shape[1:])
        if crop_coords is not None:
            img_LD = _crop_tensor(img_LD, crop_coords)
            img_FD = _crop_tensor(img_FD, crop_coords)

        if tuple(img_LD.shape[1:]) != (self.img_size, self.img_size):
            img_LD = _resize_tensor(img_LD, (self.img_size, self.img_size))
            img_FD = _resize_tensor(img_FD, (self.img_size, self.img_size))

        if self.split == "train" and self.random_flip and random.random() < 0.5:
            img_LD = torch.flip(img_LD, dims=[2])
            img_FD = torch.flip(img_FD, dims=[2])

        case_name = f"{center_patient_id}_{center_slice_idx}"

        return {
            "FD": img_FD,
            "LD": img_LD,
            "case_name": case_name,
            "patient_id": center_patient_id,
            "slice_idx": center_slice_idx,
            "neighbor_slice_indices": neighbor_slice_indices,
        }

    def _get_crop_coordinates(self, input_shape):
        """
        Generate crop coordinates for an input image tensor.

        Args:
            input_shape: Tuple (H, W) of the input tensor.

        Returns:
            (x1, x2, y1, y2) or None if no cropping is needed.
        """
        if self.split == "train":
            crop_mode = self.train_crop
        elif self.split == "val":
            crop_mode = self.val_crop
        else:
            crop_mode = self.test_crop

        if crop_mode == "none" or crop_mode is None:
            return None

        height, width = input_shape
        if height == self.img_size and width == self.img_size:
            return None

        if self.img_size > height or self.img_size > width:
            return None

        if crop_mode == "center":
            x1 = (height - self.img_size) // 2
            y1 = (width - self.img_size) // 2
        elif crop_mode == "random":
            x1 = random.randint(0, height - self.img_size)
            y1 = random.randint(0, width - self.img_size)
        else:
            x1 = (height - self.img_size) // 2
            y1 = (width - self.img_size) // 2

        return x1, x1 + self.img_size, y1, y1 + self.img_size
