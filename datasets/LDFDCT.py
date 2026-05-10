import os
import csv
import random
import glob
from pathlib import Path
from functools import lru_cache

import numpy as np
import torch
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


def _load_npy_as_tensor(path, img_size):
    arr = np.load(path)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    if arr.ndim != 2:
        raise ValueError(f"Expected 2D npy image, got shape {arr.shape} at {path}")

    if arr.shape != (img_size, img_size):
        raise ValueError(
            f"Expected shape {(img_size, img_size)}, got {arr.shape} at {path}"
        )

    arr = arr.astype(np.float32)

    min_val = float(arr.min())
    max_val = float(arr.max())

    if min_val < -1.001 or max_val > 1.001:
        raise ValueError(
            f"NPY range must be [-1, 1], got [{min_val:.4f}, {max_val:.4f}] at {path}"
        )

    arr = np.clip(arr, -1.0, 1.0)
    return torch.from_numpy(arr).unsqueeze(0).float()


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
    """

    def __init__(self, dataroot, img_size, split="train", data_len=-1):
        self.dataroot = str(dataroot)
        self.img_size = img_size
        self.split = str(split).lower()
        self.data_len_arg = data_len

        self.use_manifest_npy = self.dataroot.lower().endswith(".csv")

        if self.use_manifest_npy:
            self.manifest_path = os.path.normpath(self.dataroot)
            self.manifest_dir = os.path.dirname(self.manifest_path)

            self.samples = self._read_manifest(self.manifest_path)

            if data_len is not None and data_len > 0:
                self.samples = self.samples[:data_len]

            self.data_len = len(self.samples)

            print("[LDFDCT] Using Dataset V2 NPY manifest loader")
            print(f"[LDFDCT] Manifest: {self.manifest_path}")
            print(f"[LDFDCT] Split: {self.split}")
            print(f"[LDFDCT] Number of pairs: {self.data_len}")

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

        with open(manifest_path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # Manifest mới của bạn
            if "low_npy_path" in fieldnames and "full_npy_path" in fieldnames:
                low_col = "low_npy_path"
                full_col = "full_npy_path"

            # Fallback nếu sau này đổi tên ngắn
            elif "low_path" in fieldnames and "full_path" in fieldnames:
                low_col = "low_path"
                full_col = "full_path"

            else:
                raise ValueError(
                    f"Manifest columns are: {fieldnames}\n"
                    "Required columns should be either:\n"
                    "  low_npy_path, full_npy_path\n"
                    "or:\n"
                    "  low_path, full_path"
                )

            for row in reader:
                row_split = str(row.get("split", "")).strip().lower()

                # Nếu file là train.csv/val.csv/test.csv thì có thể không cần filter.
                # Nếu dùng all_pairs_npy256_with_split.csv thì filter theo split.
                if row_split and self.split in ["train", "val", "test"]:
                    if row_split != self.split:
                        continue

                patient_id = str(row.get("patient_id", "")).strip()
                slice_idx = str(row.get("slice_idx", "")).strip()

                low_original = row[low_col]
                full_original = row[full_col]

                low_path = _resolve_npy_path(
                    low_original,
                    self.manifest_dir,
                    self.split,
                    patient_id,
                )

                full_path = _resolve_npy_path(
                    full_original,
                    self.manifest_dir,
                    self.split,
                    patient_id,
                )

                if patient_id == "":
                    patient_id = Path(low_path).stem.split("_")[0]

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

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        if self.use_manifest_npy:
            item = self.samples[index]

            img_LD = _load_npy_as_tensor(item["low_path"], self.img_size)
            img_FD = _load_npy_as_tensor(item["full_path"], self.img_size)

            # flip ngang đồng bộ LD/FD khi train
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
