"""
HistoricalDegradationDataset — paired (HR, LQ) loader where LQ is generated
on-the-fly by the project's `DegradationMachine` (vignetting, foxing, sepia,
spatial blur, digital noise, JPEG, sinc).

Differences from the previous in-place hijack of `RealESRGANDataset`:
- Class name reflects what it does, so the YAML and the paper read consistently.
- `DegradationMachine` is instantiated once per worker, not once per __getitem__.
- Per-worker seeding so degradations are reproducible and differ across workers.
- Robust crop-or-pad logic identical to v1, but with a documented gt_size default.

Returns dict {'gt': float32 RGB CHW [0,1], 'lq': same 4x smaller, 'gt_path': str}.
"""

import os
import os.path as osp
import random
import time

import cv2
import numpy as np
import torch
from torch.utils import data as data

from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.utils.registry import DATASET_REGISTRY

from basicsr.degradation_machine import DegradationMachine


@DATASET_REGISTRY.register()
class HistoricalDegradationDataset(data.Dataset):
    """Paired GT/LQ dataset where LQ is produced by DegradationMachine.

    Required opt keys:
        dataroot_gt (str)        : root with HR images.
        meta_info (str)          : text file, one relative HR path per line.
        io_backend (dict)        : {'type': 'disk'} or lmdb spec.
        gt_size (int)            : square HR crop size; LQ becomes gt_size/4.
        use_hflip (bool)         : random horizontal flip.
        use_rot (bool)           : random 90 deg rotation / transpose.

    Optional:
        scale (int)              : default 4. The DegradationMachine pipeline
                                   currently produces a fixed 4x downscale; if
                                   scale != 4 we raise so callers don't get
                                   silently miscalibrated tensors.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.scale = int(opt.get("scale", 4))
        if self.scale != 4:
            raise ValueError(
                f"HistoricalDegradationDataset only supports scale=4 (got {self.scale}); "
                "DegradationMachine.run() outputs a 4x downscale by construction."
            )

        self.gt_size = int(opt.get("gt_size", 192))
        if self.gt_size % self.scale != 0:
            raise ValueError(f"gt_size ({self.gt_size}) must be divisible by scale ({self.scale}).")

        self.io_backend_opt = opt["io_backend"]
        self.gt_folder = opt["dataroot_gt"]
        self.file_client = None

        if self.io_backend_opt["type"] == "lmdb":
            self.io_backend_opt["db_paths"] = [self.gt_folder]
            self.io_backend_opt["client_keys"] = ["gt"]
            if not self.gt_folder.endswith(".lmdb"):
                raise ValueError(f"'dataroot_gt' must end with '.lmdb', got {self.gt_folder}")
            with open(osp.join(self.gt_folder, "meta_info.txt")) as fin:
                self.paths = [line.split(".")[0] for line in fin]
        else:
            with open(opt["meta_info"]) as fin:
                rel_paths = [line.strip().split(" ")[0] for line in fin if line.strip()]
            self.paths = [os.path.join(self.gt_folder, v) for v in rel_paths]

        # Single per-worker degrader. Real seeding happens in worker_init_fn
        # (basicsr.data.__init__), so each worker gets a different stream
        # while remaining reproducible per (seed, rank, worker_id).
        self.degrader = DegradationMachine()

    def __len__(self):
        return len(self.paths)

    def _load_image(self, idx):
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop("type"), **self.io_backend_opt)

        # Robust loop that resamples on read or decode failure.
        retries_remaining = 3
        path = self.paths[idx]
        while True:
            try:
                img_bytes = self.file_client.get(path, "gt")
                img = imfrombytes(img_bytes, float32=True)
                if img is None:
                    raise ValueError("decode returned None")
                return img, path
            except (IOError, OSError, AttributeError, ValueError) as e:
                logger = get_root_logger()
                logger.warning(f"failed to load {path} ({e}); resampling")
                idx = random.randint(0, len(self.paths) - 1)
                path = self.paths[idx]
                retries_remaining -= 1
                if retries_remaining <= 0:
                    time.sleep(0.5)
                    retries_remaining = 3

    def _crop_or_pad(self, img):
        h, w = img.shape[:2]
        size = self.gt_size

        if h < size or w < size:
            pad_h = max(0, size - h)
            pad_w = max(0, size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            h, w = img.shape[:2]

        if h > size or w > size:
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
            img = img[top:top + size, left:left + size, :]
        return img

    def __getitem__(self, index):
        img_gt, gt_path = self._load_image(index)
        img_gt = augment(img_gt, self.opt.get("use_hflip", True), self.opt.get("use_rot", True))
        img_gt = self._crop_or_pad(img_gt)

        # DegradationMachine consumes uint8 BGR; img_gt here is float32 BGR in [0,1]
        img_gt_uint8 = (np.clip(img_gt, 0, 1) * 255.0).astype(np.uint8)
        img_lq_uint8 = self.degrader.run(img_gt_uint8)

        # Sanity: enforce expected LQ size (handles any rounding inside resizes).
        expected = self.gt_size // self.scale
        if img_lq_uint8.shape[0] != expected or img_lq_uint8.shape[1] != expected:
            img_lq_uint8 = cv2.resize(
                img_lq_uint8, (expected, expected), interpolation=cv2.INTER_AREA
            )

        img_lq = img_lq_uint8.astype(np.float32) / 255.0

        gt_t = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        lq_t = img2tensor([img_lq], bgr2rgb=True, float32=True)[0]

        return {"lq": lq_t, "gt": gt_t, "gt_path": gt_path}
