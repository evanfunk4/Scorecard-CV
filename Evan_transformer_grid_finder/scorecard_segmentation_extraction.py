"""Segmentation-first golf scorecard extractor.

Implements a full train/infer/batch pipeline with a 4-head segmentation model:
- table region
- vertical separators
- horizontal separators
- junction/intersection heatmap

Design notes:
- Uses PyTorch U-Net when available.
- Decoding is flexible to mild warp: line support is evaluated in a local band
  around each estimated separator, so separators do not need to be perfectly
  vertical/horizontal in raw scans.
- Final output is OCR-oriented:
  - one PNG per base matrix cell (top-left to bottom-right ordering)
  - merged-cell inference metadata for structure-aware parsing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional
import argparse
import json
import random

import cv2
import numpy as np

from scorecard_preprocessing import preprocess_scorecard, PreprocessConfig


# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_path(base: Path, p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


def _is_label_record(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    req = ("image", "table_mask", "v_mask", "h_mask")
    return all(k in data for k in req)


def _smooth_1d(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    if signal.size < window:
        return signal.astype(np.float32, copy=True)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def _projection_peaks(signal: np.ndarray, rel_thresh: float, min_gap: int) -> list[int]:
    if signal.size < 3:
        return []
    mx = float(np.max(signal))
    if mx <= 0:
        return []
    thr = float(rel_thresh) * mx
    cands: list[tuple[float, int]] = []
    for i in range(1, signal.size - 1):
        s = float(signal[i])
        if s < thr:
            continue
        if s >= float(signal[i - 1]) and s >= float(signal[i + 1]):
            cands.append((s, i))
    cands.sort(key=lambda t: t[0], reverse=True)
    out: list[int] = []
    mg = max(1, int(min_gap))
    for _, idx in cands:
        if all(abs(idx - v) >= mg for v in out):
            out.append(idx)
    return sorted(out)


def _prune_near(values: list[int], min_gap: int) -> list[int]:
    if not values:
        return []
    vals = sorted(values)
    out = [vals[0]]
    for v in vals[1:]:
        if v - out[-1] >= min_gap:
            out.append(v)
    if len(out) == 1:
        out.append(out[0] + 1)
    return out


def _bbox_iou_xyxy(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0, ix1 - ix0 + 1)
    ih = max(0, iy1 - iy0 + 1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(1, (ax1 - ax0 + 1) * (ay1 - ay0 + 1))
    bb = max(1, (bx1 - bx0 + 1) * (by1 - by0 + 1))
    return inter / float(max(1, aa + bb - inter))


def _max_true_run_ratio(bits: np.ndarray) -> float:
    if bits.size <= 0:
        return 0.0
    b = (bits.astype(np.uint8) > 0).astype(np.uint8)
    if b.size <= 0:
        return 0.0
    best = 0
    cur = 0
    for v in b.tolist():
        if int(v) != 0:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return float(best) / float(max(1, b.size))


def _derive_junction_mask(v_mask: np.ndarray, h_mask: np.ndarray, dilate_px: int = 4) -> np.ndarray:
    vb = (v_mask > 127).astype(np.uint8)
    hb = (h_mask > 127).astype(np.uint8)
    inter = (vb & hb).astype(np.uint8) * 255
    if dilate_px > 0:
        k = int(max(1, dilate_px))
        inter = cv2.dilate(inter, cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1)), iterations=1)
    return inter


def _resize_with_pad(image: np.ndarray, masks: list[np.ndarray], size: int) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = image.shape[:2]
    size = int(size)
    if size <= 16:
        raise ValueError("size must be > 16")
    scale = float(size) / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))

    img_r = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    mask_r = [cv2.resize(m, (nw, nh), interpolation=cv2.INTER_NEAREST) for m in masks]

    top = (size - nh) // 2
    left = (size - nw) // 2

    out_img = np.zeros((size, size, 3), dtype=np.uint8)
    out_img[top : top + nh, left : left + nw] = img_r

    out_masks: list[np.ndarray] = []
    for m in mask_r:
        mm = np.zeros((size, size), dtype=np.uint8)
        mm[top : top + nh, left : left + nw] = m
        out_masks.append(mm)
    return out_img, out_masks


# -----------------------------------------------------------------------------
# Label IO
# -----------------------------------------------------------------------------


def _collect_label_jsons(labels_dir: Path) -> list[Path]:
    out: list[Path] = []
    for js in sorted(Path(labels_dir).glob("*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _is_label_record(data):
            out.append(js)
    return out


def _collect_image_files(root_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    out: list[Path] = []
    for p in sorted(Path(root_dir).rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() in exts:
            out.append(p)
    return out


def _load_label_sample(json_path: Path, junction_dilate_px: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rec = json.loads(json_path.read_text(encoding="utf-8"))
    image = cv2.imread(str(_resolve_path(json_path.parent, rec["image"])), cv2.IMREAD_COLOR)
    table = cv2.imread(str(_resolve_path(json_path.parent, rec["table_mask"])), cv2.IMREAD_GRAYSCALE)
    vmask = cv2.imread(str(_resolve_path(json_path.parent, rec["v_mask"])), cv2.IMREAD_GRAYSCALE)
    hmask = cv2.imread(str(_resolve_path(json_path.parent, rec["h_mask"])), cv2.IMREAD_GRAYSCALE)
    if image is None or table is None or vmask is None or hmask is None:
        raise RuntimeError(f"Failed to load sample from {json_path}")
    if image.shape[:2] != table.shape[:2] or image.shape[:2] != vmask.shape[:2] or image.shape[:2] != hmask.shape[:2]:
        raise RuntimeError(f"Shape mismatch in {json_path}")
    jmask = _derive_junction_mask(vmask, hmask, dilate_px=junction_dilate_px)
    return image, table, vmask, hmask, jmask


# -----------------------------------------------------------------------------
# Torch model (imported lazily)
# -----------------------------------------------------------------------------


def _import_torch() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from torch.utils.data import random_split
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for scorecard_segmentation_extraction.py. "
            "Install torch/torchvision in your environment and rerun."
        ) from exc
    return torch, nn, F, Dataset, DataLoader, random_split


def _build_unet_small(nn: Any) -> Any:
    def _norm(ch: int) -> Any:
        # GroupNorm is stabler than BatchNorm for tiny batch sizes.
        for g in (16, 8, 4, 2):
            if ch % g == 0:
                return nn.GroupNorm(g, ch)
        return nn.GroupNorm(1, ch)

    class ConvBlock(nn.Module):
        def __init__(self, in_ch: int, out_ch: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                _norm(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                _norm(out_ch),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Any) -> Any:
            return self.net(x)

    class UNetSmall(nn.Module):
        def __init__(self, in_ch: int = 3, out_ch: int = 4):
            super().__init__()
            c1, c2, c3, c4, cb = 16, 32, 64, 128, 192
            self.e1 = ConvBlock(in_ch, c1)
            self.e2 = ConvBlock(c1, c2)
            self.e3 = ConvBlock(c2, c3)
            self.e4 = ConvBlock(c3, c4)
            self.pool = nn.MaxPool2d(2)

            self.b = ConvBlock(c4, cb)

            self.u4 = nn.ConvTranspose2d(cb, c4, 2, stride=2)
            self.d4 = ConvBlock(c4 + c4, c4)
            self.u3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
            self.d3 = ConvBlock(c3 + c3, c3)
            self.u2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.d2 = ConvBlock(c2 + c2, c2)
            self.u1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.d1 = ConvBlock(c1 + c1, c1)

            self.head = nn.Conv2d(c1, out_ch, 1)
            # Biases reflect rough class priors to avoid starting at 0.5
            # everywhere on sparse line/junction heads.
            import torch

            with torch.no_grad():
                if int(out_ch) == 4:
                    b = torch.tensor([1.1, -2.2, -2.2, -2.8], dtype=torch.float32)
                else:
                    b = torch.zeros((int(out_ch),), dtype=torch.float32)
                self.head.bias.copy_(b)

        @staticmethod
        def _cat(skip: Any, up: Any) -> Any:
            import torch
            dh = skip.shape[2] - up.shape[2]
            dw = skip.shape[3] - up.shape[3]
            if dh != 0 or dw != 0:
                up = nn.functional.pad(up, (0, max(0, dw), 0, max(0, dh)))
                up = up[:, :, : skip.shape[2], : skip.shape[3]]
            return torch.cat([skip, up], dim=1)

        def forward(self, x: Any) -> Any:
            e1 = self.e1(x)
            e2 = self.e2(self.pool(e1))
            e3 = self.e3(self.pool(e2))
            e4 = self.e4(self.pool(e3))
            b = self.b(self.pool(e4))

            u4 = self.u4(b)
            d4 = self.d4(self._cat(e4, u4))
            u3 = self.u3(d4)
            d3 = self.d3(self._cat(e3, u3))
            u2 = self.u2(d3)
            d2 = self.d2(self._cat(e2, u2))
            u1 = self.u1(d2)
            d1 = self.d1(self._cat(e1, u1))
            return self.head(d1)

    return UNetSmall


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    labels_dir: Path
    out_weights: Path
    epochs: int = 45
    batch_size: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    train_size: int = 768
    val_ratio: float = 0.18
    seed: int = 7
    junction_dilate_px: int = 4
    num_workers: int = 0
    pretrain_images_dir: Optional[Path] = None
    pretrain_epochs: int = 0
    pretrain_batch_size: int = 4
    pretrain_lr: float = 3e-4
    pretrain_weight_decay: float = 1e-5
    pretrain_max_images: int = 0


class _TrainDatasetBase:
    """Thin wrapper to avoid importing torch at module import time."""

    pass


def train_model(cfg: TrainConfig) -> None:
    torch, nn, F, Dataset, DataLoader, random_split = _import_torch()
    _set_seed(cfg.seed)

    js_files = _collect_label_jsons(cfg.labels_dir)
    if not js_files:
        raise RuntimeError(f"No valid label JSONs found in {cfg.labels_dir}")

    class TrainDataset(Dataset):
        def __init__(self, files: list[Path], train_size: int, is_train: bool):
            self.files = files
            self.train_size = int(train_size)
            self.is_train = bool(is_train)

        def __len__(self) -> int:
            return len(self.files)

        def _augment(self, image: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
            if not self.is_train:
                return image, masks

            # Stronger geometric perturbation for small-data and scan warp.
            if random.random() < 0.60:
                h, w = image.shape[:2]
                angle = random.uniform(-3.8, 3.8)
                scale = random.uniform(0.965, 1.035)
                mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
                mat[0, 2] += random.uniform(-0.02, 0.02) * w
                mat[1, 2] += random.uniform(-0.02, 0.02) * h
                image = cv2.warpAffine(
                    image,
                    mat,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                masks = [
                    cv2.warpAffine(m, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for m in masks
                ]

            # Perspective warp to mimic page bend / camera misalignment.
            if random.random() < 0.50:
                h, w = image.shape[:2]
                src = np.array(
                    [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                    dtype=np.float32,
                )
                j = 0.035 * min(h, w)
                dst = src + np.random.uniform(-j, j, size=(4, 2)).astype(np.float32)
                dst[:, 0] = np.clip(dst[:, 0], 0.0, w - 1.0)
                dst[:, 1] = np.clip(dst[:, 1], 0.0, h - 1.0)
                pm = cv2.getPerspectiveTransform(src, dst)
                image = cv2.warpPerspective(
                    image,
                    pm,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                masks = [
                    cv2.warpPerspective(m, pm, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for m in masks
                ]

            # Mild photometric perturbation.
            if random.random() < 0.55:
                alpha = random.uniform(0.90, 1.13)
                beta = random.uniform(-18.0, 18.0)
                image = np.clip(image.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

            if random.random() < 0.25:
                k = random.choice([3, 5])
                image = cv2.GaussianBlur(image, (k, k), 0)

            if random.random() < 0.30:
                noise = np.random.normal(0.0, 5.0, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

            return image, masks

        def __getitem__(self, idx: int) -> Any:
            js = self.files[idx]
            image, table, vmask, hmask, jmask = _load_label_sample(js, junction_dilate_px=cfg.junction_dilate_px)
            masks = [table, vmask, hmask, jmask]
            image, masks = self._augment(image, masks)
            image, masks = _resize_with_pad(image, masks, size=self.train_size)

            x = image.astype(np.float32) / 255.0
            y = np.stack([(m > 127).astype(np.float32) for m in masks], axis=2)

            x_t = torch.from_numpy(np.transpose(x, (2, 0, 1))).float()
            y_t = torch.from_numpy(np.transpose(y, (2, 0, 1))).float()
            return x_t, y_t

    class PretrainDataset(Dataset):
        def __init__(self, files: list[Path], train_size: int):
            self.files = files
            self.train_size = int(train_size)

        def __len__(self) -> int:
            return len(self.files)

        @staticmethod
        def _shared_geom(image: np.ndarray) -> np.ndarray:
            h, w = image.shape[:2]
            out = image

            if random.random() < 0.75:
                angle = random.uniform(-8.5, 8.5)
                scale = random.uniform(0.90, 1.10)
                mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
                mat[0, 2] += random.uniform(-0.04, 0.04) * w
                mat[1, 2] += random.uniform(-0.04, 0.04) * h
                out = cv2.warpAffine(
                    out,
                    mat,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

            if random.random() < 0.70:
                src = np.array(
                    [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
                    dtype=np.float32,
                )
                j = 0.06 * min(h, w)
                dst = src + np.random.uniform(-j, j, size=(4, 2)).astype(np.float32)
                dst[:, 0] = np.clip(dst[:, 0], 0.0, w - 1.0)
                dst[:, 1] = np.clip(dst[:, 1], 0.0, h - 1.0)
                pm = cv2.getPerspectiveTransform(src, dst)
                out = cv2.warpPerspective(
                    out,
                    pm,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )

            if random.random() < 0.25:
                sx = random.uniform(0.93, 1.07)
                sy = random.uniform(0.93, 1.07)
                mat = np.array([[sx, 0.0, (1.0 - sx) * 0.5 * w], [0.0, sy, (1.0 - sy) * 0.5 * h]], dtype=np.float32)
                out = cv2.warpAffine(
                    out,
                    mat,
                    (w, h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            return out

        @staticmethod
        def _motion_blur(image: np.ndarray, k: int) -> np.ndarray:
            k = max(3, int(k) | 1)
            ker = np.zeros((k, k), dtype=np.float32)
            mode = random.choice(["h", "v", "d1", "d2"])
            if mode == "h":
                ker[k // 2, :] = 1.0
            elif mode == "v":
                ker[:, k // 2] = 1.0
            elif mode == "d1":
                np.fill_diagonal(ker, 1.0)
            else:
                np.fill_diagonal(np.fliplr(ker), 1.0)
            ker /= float(np.sum(ker))
            return cv2.filter2D(image, -1, ker, borderType=cv2.BORDER_REPLICATE)

        @staticmethod
        def _jpeg_roundtrip(image: np.ndarray, quality: int) -> np.ndarray:
            ok, enc = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
            if not ok:
                return image
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            if dec is None or dec.shape != image.shape:
                return image
            return dec

        @staticmethod
        def _shadow_overlay(image: np.ndarray) -> np.ndarray:
            h, w = image.shape[:2]
            cx = random.randint(0, max(0, w - 1))
            cy = random.randint(0, max(0, h - 1))
            ax = max(10, int(random.uniform(0.16, 0.62) * w))
            ay = max(10, int(random.uniform(0.16, 0.62) * h))
            ang = random.uniform(0.0, 180.0)
            ov = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(ov, (cx, cy), (ax, ay), ang, 0, 360, 255, -1, cv2.LINE_AA)
            ov = cv2.GaussianBlur(ov, (0, 0), sigmaX=max(6.0, 0.10 * min(h, w)))
            alpha = random.uniform(0.16, 0.48)
            dark = 1.0 - alpha * (ov.astype(np.float32) / 255.0)
            out = np.clip(image.astype(np.float32) * dark[:, :, None], 0.0, 255.0).astype(np.uint8)
            return out

        @staticmethod
        def _degrade_input(image: np.ndarray) -> np.ndarray:
            out = image
            h, w = out.shape[:2]

            if random.random() < 0.90:
                alpha = random.uniform(0.62, 1.38)
                beta = random.uniform(-52.0, 52.0)
                out = np.clip(out.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)

            if random.random() < 0.55:
                gamma = random.uniform(0.65, 1.65)
                lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
                out = cv2.LUT(out, lut.astype(np.uint8))

            if random.random() < 0.52:
                gains = np.array(
                    [random.uniform(0.80, 1.22), random.uniform(0.80, 1.22), random.uniform(0.80, 1.22)],
                    dtype=np.float32,
                )
                out = np.clip(out.astype(np.float32) * gains.reshape(1, 1, 3), 0.0, 255.0).astype(np.uint8)

            if random.random() < 0.45:
                k = random.choice([3, 5, 7])
                out = cv2.GaussianBlur(out, (k, k), sigmaX=random.uniform(0.8, 2.8))
            if random.random() < 0.30:
                out = PretrainDataset._motion_blur(out, random.choice([7, 9, 11, 13]))

            if random.random() < 0.70:
                sigma = random.uniform(4.0, 18.0)
                noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
                out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

            if random.random() < 0.55:
                out = PretrainDataset._jpeg_roundtrip(out, quality=random.randint(24, 82))

            if random.random() < 0.55:
                out = PretrainDataset._shadow_overlay(out)

            if random.random() < 0.40:
                n_rect = random.randint(1, 3)
                for _ in range(n_rect):
                    rw = max(6, int(random.uniform(0.04, 0.20) * w))
                    rh = max(6, int(random.uniform(0.03, 0.14) * h))
                    x0 = random.randint(0, max(0, w - rw))
                    y0 = random.randint(0, max(0, h - rh))
                    col = int(random.uniform(0.0, 255.0))
                    cv2.rectangle(out, (x0, y0), (x0 + rw, y0 + rh), (col, col, col), -1, cv2.LINE_AA)

            if random.random() < 0.35:
                n_streak = random.randint(2, 12)
                for _ in range(n_streak):
                    if random.random() < 0.60:
                        yy = random.randint(0, max(0, h - 1))
                        cv2.line(
                            out,
                            (0, yy),
                            (max(0, w - 1), yy),
                            (random.randint(0, 255),) * 3,
                            random.choice([1, 1, 2]),
                            cv2.LINE_AA,
                        )
                    else:
                        xx = random.randint(0, max(0, w - 1))
                        cv2.line(
                            out,
                            (xx, 0),
                            (xx, max(0, h - 1)),
                            (random.randint(0, 255),) * 3,
                            random.choice([1, 1, 2]),
                            cv2.LINE_AA,
                        )
            return out

        def __getitem__(self, idx: int) -> Any:
            image = None
            p = self.files[idx]
            max_probe = min(len(self.files), 8)
            for k in range(max_probe):
                pp = self.files[(idx + k) % len(self.files)]
                image = cv2.imread(str(pp), cv2.IMREAD_COLOR)
                if image is not None:
                    p = pp
                    break
            if image is None:
                raise RuntimeError(f"Failed to load pretraining image: {p}")

            image = self._shared_geom(image)
            target = image.copy()
            inp = self._degrade_input(image.copy())

            inp, _ = _resize_with_pad(inp, [], size=self.train_size)
            target, _ = _resize_with_pad(target, [], size=self.train_size)

            x = inp.astype(np.float32) / 255.0
            y = target.astype(np.float32) / 255.0
            x_t = torch.from_numpy(np.transpose(x, (2, 0, 1))).float()
            y_t = torch.from_numpy(np.transpose(y, (2, 0, 1))).float()
            return x_t, y_t

    n_total = len(js_files)
    n_val = max(1, int(round(cfg.val_ratio * n_total)))
    n_train = max(1, n_total - n_val)
    if n_train + n_val > n_total:
        n_val = n_total - n_train

    all_ds = TrainDataset(js_files, cfg.train_size, is_train=True)
    train_ds, val_ds = random_split(
        all_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    # Swap val split to no augmentation.
    val_files = [js_files[i] for i in val_ds.indices]
    val_ds = TrainDataset(val_files, cfg.train_size, is_train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=True,
        num_workers=max(0, int(cfg.num_workers)),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, int(cfg.batch_size)),
        shuffle=False,
        num_workers=max(0, int(cfg.num_workers)),
        pin_memory=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    UNetSmall = _build_unet_small(nn)
    model = UNetSmall(in_ch=3, out_ch=4).to(device)

    # Optional self-supervised pretraining on unlabeled scorecards.
    if cfg.pretrain_images_dir is not None and int(cfg.pretrain_epochs) > 0:
        pre_dir = Path(cfg.pretrain_images_dir)
        if pre_dir.exists():
            pre_files = _collect_image_files(pre_dir)
            if pre_files:
                rnd = random.Random(int(cfg.seed))
                rnd.shuffle(pre_files)
                if int(cfg.pretrain_max_images) > 0:
                    pre_files = pre_files[: int(cfg.pretrain_max_images)]

                pre_bs = int(cfg.pretrain_batch_size) if int(cfg.pretrain_batch_size) > 0 else max(1, int(cfg.batch_size))
                pre_ds = PretrainDataset(pre_files, train_size=int(cfg.train_size))
                pre_loader = DataLoader(
                    pre_ds,
                    batch_size=max(1, pre_bs),
                    shuffle=True,
                    num_workers=max(0, int(cfg.num_workers)),
                    pin_memory=False,
                )

                pre_model = UNetSmall(in_ch=3, out_ch=3).to(device)
                pre_opt = torch.optim.AdamW(
                    pre_model.parameters(),
                    lr=float(cfg.pretrain_lr),
                    weight_decay=float(cfg.pretrain_weight_decay),
                )
                pre_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    pre_opt, T_max=max(2, int(cfg.pretrain_epochs))
                )
                print(
                    f"pretrain: images={len(pre_files)} epochs={int(cfg.pretrain_epochs)} "
                    f"bs={pre_bs} lr={float(cfg.pretrain_lr):.3e}"
                )
                for ep in range(1, int(cfg.pretrain_epochs) + 1):
                    pre_model.train()
                    ep_loss = 0.0
                    ep_n = 0
                    for xb, yb in pre_loader:
                        xb = xb.to(device)
                        yb = yb.to(device)
                        pre_opt.zero_grad(set_to_none=True)
                        logits = pre_model(xb)
                        pred = torch.sigmoid(logits)

                        l_rec = F.smooth_l1_loss(pred, yb)
                        pgx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
                        pgy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
                        tgx = yb[:, :, :, 1:] - yb[:, :, :, :-1]
                        tgy = yb[:, :, 1:, :] - yb[:, :, :-1, :]
                        l_grad = F.smooth_l1_loss(pgx, tgx) + F.smooth_l1_loss(pgy, tgy)
                        loss = 0.75 * l_rec + 0.25 * l_grad

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(pre_model.parameters(), max_norm=2.0)
                        pre_opt.step()

                        ep_loss += float(loss.item()) * int(xb.shape[0])
                        ep_n += int(xb.shape[0])

                    pre_sched.step()
                    ep_mean = ep_loss / float(max(1, ep_n))
                    print(f"pretrain epoch {ep:03d}/{cfg.pretrain_epochs} loss={ep_mean:.5f}")

                # Transfer all non-head weights from reconstruction model.
                dst_sd = model.state_dict()
                src_sd = pre_model.state_dict()
                copied = 0
                for k, v in src_sd.items():
                    if k.startswith("head."):
                        continue
                    if k in dst_sd and dst_sd[k].shape == v.shape:
                        dst_sd[k] = v
                        copied += 1
                model.load_state_dict(dst_sd, strict=True)
                print(f"pretrain transfer: copied {copied} tensors into segmentation model")
            else:
                print(f"pretrain skipped: no images found in {pre_dir}")
        else:
            print(f"pretrain skipped: directory not found: {pre_dir}")

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(4, int(cfg.epochs)))

    ch_weights = torch.tensor([1.0, 2.0, 2.0, 3.0], dtype=torch.float32, device=device).view(1, 4, 1, 1)
    pos_w = torch.tensor([1.0, 7.0, 7.0, 10.0], dtype=torch.float32, device=device).view(1, 4, 1, 1)
    neg_w = torch.tensor([1.0, 1.4, 1.4, 1.6], dtype=torch.float32, device=device).view(1, 4, 1, 1)

    def loss_fn(logits: Any, target: Any) -> Any:
        bce_raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        px_w = target * pos_w + (1.0 - target) * neg_w
        bce = (bce_raw * px_w * ch_weights).mean()

        prob = torch.sigmoid(logits)
        inter = (prob * target).sum(dim=(2, 3))
        denom = prob.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + 1e-6
        dice = 1.0 - (2.0 * inter + 1e-6) / denom
        dice = (dice * ch_weights.view(1, 4)).mean()
        return 0.55 * bce + 0.45 * dice

    best_val = 1e18
    _ensure_dir(cfg.out_weights.parent)

    for ep in range(1, int(cfg.epochs) + 1):
        model.train()
        tr_loss = 0.0
        tr_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            opt.step()
            tr_loss += float(loss.item()) * int(xb.shape[0])
            tr_n += int(xb.shape[0])

        model.eval()
        va_loss = 0.0
        va_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                va_loss += float(loss.item()) * int(xb.shape[0])
                va_n += int(xb.shape[0])

        tr = tr_loss / float(max(1, tr_n))
        va = va_loss / float(max(1, va_n))
        sched.step()
        lr_now = float(opt.param_groups[0]["lr"])
        print(f"epoch {ep:03d}/{cfg.epochs}  train={tr:.5f}  val={va:.5f}  lr={lr_now:.3e}")

        if va < best_val:
            best_val = va
            payload = {
                "state_dict": model.state_dict(),
                "model": "unet_small",
                "out_channels": 4,
                "train_size": int(cfg.train_size),
                "channels": ["table", "v", "h", "junction"],
                "seed": int(cfg.seed),
            }
            torch.save(payload, str(cfg.out_weights))

    print(f"Saved best weights to {cfg.out_weights}")


# -----------------------------------------------------------------------------
# Inference decoding
# -----------------------------------------------------------------------------


@dataclass
class InferConfig:
    table_thresh: float = 0.48
    line_thresh: float = 0.46
    junction_thresh: float = 0.38
    max_tables: int = 2
    min_table_area_ratio: float = 0.010
    max_component_fill_ratio: float = 0.96
    min_prob_contrast: float = 0.050
    min_line_prob_contrast: float = 0.045
    min_box_line_energy: float = 0.006
    min_grid_score: float = 1.5

    min_rows: int = 4
    max_rows: int = 26
    min_cols: int = 4
    max_cols: int = 34
    min_keep_cols: int = 9

    min_gap_px: int = 8
    peak_rel_thresh: float = 0.12
    min_line_cov: float = 0.24

    line_band_px: int = 4
    flexible_band_px: int = 10
    auto_tune_decode: bool = True
    axis_dp_enable: bool = True
    axis_dp_max_candidates: int = 72
    axis_dp_node_cost: float = 0.38
    axis_dp_gap_smooth_w: float = 0.42
    axis_dp_big_gap_w: float = 0.22
    axis_dp_soft_h_weight: float = 0.28
    axis_dp_soft_v_weight: float = 0.34
    axis_dp_count_penalty: float = 0.05
    prior_gap_fill_enable: bool = False
    border_snap_window_ratio: float = 0.10
    border_snap_min_cov: float = 0.16
    outer_line_trim_min_cov: float = 0.30
    color_edge_enable: bool = True
    color_edge_h_weight: float = 0.10
    color_edge_v_weight: float = 0.05
    weak_hline_prune_thresh: float = 0.25
    gap_fill_ratio_h: float = 1.55
    h_color_peak_min: float = 0.34
    top_hline_guard_ratio: float = 0.07
    top_cluster_short_gap_ratio: float = 0.72
    top_hline_cleanup_enable: bool = False
    global_color_line_inject_topk: int = 8
    global_color_line_min: float = 0.34
    color_line_min_cov: float = 0.68
    min_hline_junction_hit: float = 0.50
    presence_min_run_ratio: float = 0.46
    merge_presence_min_cov: float = 0.30
    conservative_table_bbox: bool = False
    conservative_max_width_ratio: float = 0.88
    conservative_max_height_ratio: float = 0.96
    conservative_prefer_compact_bonus: float = 0.22
    legacy_decode: bool = False

    cell_inset_px: int = 2
    min_cell_w: int = 10
    min_cell_h: int = 10


@dataclass
class DecodedGrid:
    table_id: int
    bbox: tuple[int, int, int, int]  # x0,y0,x1,y1
    x_lines: list[int]
    y_lines: list[int]
    v_presence: np.ndarray
    h_presence: np.ndarray
    v_polylines: Optional[list[list[list[int]]]] = None
    h_polylines: Optional[list[list[list[int]]]] = None


class GridDecoder:
    def __init__(self, cfg: Optional[InferConfig] = None):
        self.cfg = cfg or InferConfig()

    @staticmethod
    def _color_transition_maps(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Use Lab to capture luminance + chroma transitions that often align to row bands.
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab.shape[:2]
        if h < 2 or w < 2:
            z = np.zeros((h, w), dtype=np.float32)
            return z, z

        dx = np.abs(np.diff(lab, axis=1)).mean(axis=2)  # shape h, w-1
        dy = np.abs(np.diff(lab, axis=0)).mean(axis=2)  # shape h-1, w

        v_map = np.zeros((h, w), dtype=np.float32)
        h_map = np.zeros((h, w), dtype=np.float32)
        v_map[:, 1:] = dx
        h_map[1:, :] = dy

        v_map = cv2.GaussianBlur(v_map, (5, 5), 0)
        h_map = cv2.GaussianBlur(h_map, (5, 5), 0)

        def _norm(m: np.ndarray) -> np.ndarray:
            if m.size == 0:
                return m.astype(np.float32)
            lo = float(np.percentile(m, 10.0))
            hi = float(np.percentile(m, 99.0))
            if hi <= lo + 1e-6:
                return np.zeros_like(m, dtype=np.float32)
            out = (m - lo) / (hi - lo)
            return np.clip(out, 0.0, 1.0).astype(np.float32)

        return _norm(v_map), _norm(h_map)

    @staticmethod
    def _image_vertical_line_evidence_map(image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        if h < 2 or w < 2:
            return np.zeros((h, w), dtype=np.float32)

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 0)

        # Binary text/line foreground.
        bw = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            31,
            12,
        )
        # Vertical-gradient edges catch thin/low-contrast printed rules.
        gx = np.abs(cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3))
        gthr = float(np.percentile(gx, 86.0))
        gmask = ((gx >= gthr).astype(np.uint8) * 255) if np.isfinite(gthr) and gthr > 0 else np.zeros_like(bw)
        raw = cv2.bitwise_or(bw, gmask)

        # Keep structures that are vertically persistent, then bridge small gaps.
        k_open = max(8, int(round(0.020 * h)))
        k_close = max(k_open + 2, int(round(0.034 * h)))
        v_open = cv2.morphologyEx(raw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_open)), iterations=1)
        v_close = cv2.morphologyEx(v_open, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_close)), iterations=1)

        # Remove short/text-like connected components so footer/header glyphs do
        # not masquerade as true vertical grid lines.
        comp = (v_close > 0).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(comp, connectivity=8)
        keep = np.zeros_like(comp, dtype=np.uint8)
        min_h = max(12, int(round(0.065 * h)))
        max_w_soft = max(5, int(round(0.045 * w)))
        for i in range(1, n):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])
            if ch < min_h:
                continue
            if cw > max_w_soft and ch < int(round(2.6 * min_h)):
                continue
            if area < max(18, int(round(0.34 * ch))):
                continue
            keep[y : y + ch, x : x + cw][lab[y : y + ch, x : x + cw] == i] = 255

        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        return (keep.astype(np.float32) / 255.0).astype(np.float32)

    @staticmethod
    def _image_horizontal_line_evidence_map(image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        rot = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        v_rot = GridDecoder._image_vertical_line_evidence_map(rot)
        if v_rot.size == 0:
            h, w = image_bgr.shape[:2]
            return np.zeros((h, w), dtype=np.float32)
        h_map = cv2.rotate(v_rot, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return h_map.astype(np.float32)

    @staticmethod
    def _image_canny_axis_maps(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h, w = image_bgr.shape[:2]
        if h < 2 or w < 2:
            z = np.zeros((h, w), dtype=np.float32)
            return z, z

        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        den = cv2.bilateralFilter(gray, d=5, sigmaColor=35, sigmaSpace=35)
        edge = cv2.Canny(den, 40, 130)

        kx_long = max(15, int(round(0.030 * w)))
        ky_long = max(15, int(round(0.030 * h)))
        kx_small = max(9, int(round(0.012 * w)))
        ky_small = max(9, int(round(0.012 * h)))
        h_kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (kx_long, 1))
        v_kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky_long))
        h_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (kx_small, 1))
        v_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky_small))

        h_long = cv2.morphologyEx(edge, cv2.MORPH_OPEN, h_kernel_long, iterations=1)
        v_long = cv2.morphologyEx(edge, cv2.MORPH_OPEN, v_kernel_long, iterations=1)
        h_small = cv2.morphologyEx(edge, cv2.MORPH_OPEN, h_kernel_small, iterations=1)
        v_small = cv2.morphologyEx(edge, cv2.MORPH_OPEN, v_kernel_small, iterations=1)

        h_map = cv2.bitwise_or(h_long, h_small)
        v_map = cv2.bitwise_or(v_long, v_small)
        h_map = cv2.morphologyEx(h_map, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        v_map = cv2.morphologyEx(v_map, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        h_map = cv2.dilate(h_map, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        v_map = cv2.dilate(v_map, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        hf = (h_map.astype(np.float32) / 255.0).astype(np.float32)
        vf = (v_map.astype(np.float32) / 255.0).astype(np.float32)
        return vf, hf

    @staticmethod
    def _smooth_int_sequence(vals: list[int], radius: int = 1) -> list[int]:
        if not vals:
            return []
        arr = np.asarray(vals, dtype=np.float32)
        if radius <= 0 or arr.size < 3:
            return [int(round(v)) for v in arr.tolist()]
        k = int(2 * radius + 1)
        ker = np.ones((k,), dtype=np.float32) / float(k)
        pad = np.pad(arr, (radius, radius), mode="edge")
        sm = np.convolve(pad, ker, mode="valid")
        return [int(round(v)) for v in sm.tolist()]

    def _fit_vertical_polylines(
        self,
        x_lines: list[int],
        y_lines: list[int],
        v_prob: np.ndarray,
        v_line_prob: Optional[np.ndarray] = None,
    ) -> list[list[list[int]]]:
        if len(x_lines) < 2 or len(y_lines) < 2:
            return [[[int(x), int(y_lines[0])], [int(x), int(y_lines[-1])]] for x in x_lines]
        h, w = v_prob.shape[:2]
        vmap = v_prob
        if v_line_prob is not None and v_line_prob.shape[:2] == v_prob.shape[:2]:
            vmap = np.maximum(vmap, 0.90 * v_line_prob)

        yk = [int(np.clip(y, 0, h - 1)) for y in y_lines]
        step_med = float(np.median(np.diff(np.array(yk, dtype=np.float32)))) if len(yk) >= 3 else float(max(8, h // 20))
        yband = max(2, int(round(0.18 * max(4.0, step_med))))
        xband = max(1, int(round(0.18 * max(6.0, float(np.median(np.diff(np.array(sorted(x_lines), dtype=np.float32))) if len(x_lines) >= 3 else 24.0)))))
        xsearch = max(3, int(round(0.55 * xband)))

        out: list[list[list[int]]] = []
        for xb in x_lines:
            base_x = int(np.clip(xb, 0, w - 1))
            xs: list[int] = []
            prev_x = base_x
            for yy in yk:
                lo = max(0, min(base_x - 2 * xsearch, prev_x - xsearch))
                hi = min(w - 1, max(base_x + 2 * xsearch, prev_x + xsearch))
                best_x = prev_x
                best_s = -1e18
                y0 = max(0, yy - yband)
                y1 = min(h - 1, yy + yband)
                for xx in range(lo, hi + 1):
                    x0 = max(0, xx - xband)
                    x1 = min(w - 1, xx + xband)
                    patch = vmap[y0 : y1 + 1, x0 : x1 + 1]
                    if patch.size == 0:
                        continue
                    s = 0.72 * float(np.max(patch)) + 0.28 * float(np.mean(patch))
                    s -= 0.0045 * abs(xx - prev_x)
                    s -= 0.0018 * abs(xx - base_x)
                    if s > best_s:
                        best_s = s
                        best_x = xx
                prev_x = int(best_x)
                xs.append(prev_x)
            xs = self._smooth_int_sequence(xs, radius=1)
            out.append([[int(np.clip(x, 0, w - 1)), int(y)] for x, y in zip(xs, yk)])
        return out

    def _fit_horizontal_polylines(
        self,
        x_lines: list[int],
        y_lines: list[int],
        h_prob: np.ndarray,
        h_color_prob: Optional[np.ndarray] = None,
        h_line_prob: Optional[np.ndarray] = None,
    ) -> list[list[list[int]]]:
        if len(x_lines) < 2 or len(y_lines) < 2:
            return [[[int(x_lines[0]), int(y)], [int(x_lines[-1]), int(y)]] for y in y_lines]
        h, w = h_prob.shape[:2]
        hmap = h_prob
        if h_color_prob is not None and h_color_prob.shape[:2] == h_prob.shape[:2]:
            hmap = np.maximum(hmap, 0.86 * h_color_prob)
        if h_line_prob is not None and h_line_prob.shape[:2] == h_prob.shape[:2]:
            hmap = np.maximum(hmap, 0.90 * h_line_prob)

        xk = [int(np.clip(x, 0, w - 1)) for x in x_lines]
        step_med = float(np.median(np.diff(np.array(xk, dtype=np.float32)))) if len(xk) >= 3 else float(max(8, w // 20))
        xband = max(2, int(round(0.18 * max(4.0, step_med))))
        yband = max(1, int(round(0.18 * max(6.0, float(np.median(np.diff(np.array(sorted(y_lines), dtype=np.float32))) if len(y_lines) >= 3 else 24.0)))))
        ysearch = max(3, int(round(0.55 * yband)))

        out: list[list[list[int]]] = []
        for yb in y_lines:
            base_y = int(np.clip(yb, 0, h - 1))
            ys: list[int] = []
            prev_y = base_y
            for xx in xk:
                lo = max(0, min(base_y - 2 * ysearch, prev_y - ysearch))
                hi = min(h - 1, max(base_y + 2 * ysearch, prev_y + ysearch))
                best_y = prev_y
                best_s = -1e18
                x0 = max(0, xx - xband)
                x1 = min(w - 1, xx + xband)
                for yy in range(lo, hi + 1):
                    y0 = max(0, yy - yband)
                    y1 = min(h - 1, yy + yband)
                    patch = hmap[y0 : y1 + 1, x0 : x1 + 1]
                    if patch.size == 0:
                        continue
                    s = 0.72 * float(np.max(patch)) + 0.28 * float(np.mean(patch))
                    s -= 0.0045 * abs(yy - prev_y)
                    s -= 0.0018 * abs(yy - base_y)
                    if s > best_s:
                        best_s = s
                        best_y = yy
                prev_y = int(best_y)
                ys.append(prev_y)
            ys = self._smooth_int_sequence(ys, radius=1)
            out.append([[int(x), int(np.clip(y, 0, h - 1))] for x, y in zip(xk, ys)])
        return out

    @staticmethod
    def _prob_contrast(prob: np.ndarray) -> float:
        if prob.size == 0:
            return 0.0
        p95 = float(np.percentile(prob, 95.0))
        p50 = float(np.percentile(prob, 50.0))
        return max(0.0, p95 - p50)

    @staticmethod
    def _strong_ratio(prob: np.ndarray, thr: float) -> float:
        if prob.size == 0:
            return 0.0
        return float((prob >= thr).mean())

    def _decode_table_bboxes(self, table_prob: np.ndarray, v_prob: np.ndarray, h_prob: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = table_prob.shape[:2]
        t_con = self._prob_contrast(table_prob)
        v_con = self._prob_contrast(v_prob)
        h_con = self._prob_contrast(h_prob)

        # Prevent "random full-image grid" when heads are nearly flat.
        if t_con < self.cfg.min_prob_contrast and max(v_con, h_con) < self.cfg.min_line_prob_contrast:
            return []

        use_t = t_con >= self.cfg.min_prob_contrast
        use_v = v_con >= self.cfg.min_line_prob_contrast
        use_h = h_con >= self.cfg.min_line_prob_contrast
        if not (use_t or use_v or use_h):
            return []

        t = (table_prob >= self.cfg.table_thresh).astype(np.uint8) if use_t else np.zeros((h, w), dtype=np.uint8)
        lv = (v_prob >= max(0.26, self.cfg.line_thresh * 0.62)).astype(np.uint8) if use_v else np.zeros((h, w), dtype=np.uint8)
        lh = (h_prob >= max(0.26, self.cfg.line_thresh * 0.62)).astype(np.uint8) if use_h else np.zeros((h, w), dtype=np.uint8)
        mask_comb = ((t | lv | lh) * 255).astype(np.uint8)
        mask_line = ((lv | lh) * 255).astype(np.uint8)
        if mask_comb.max() <= 0 and mask_line.max() <= 0:
            return []

        # Adaptive "strong line" thresholds for table-candidate scoring.
        # Fixed high cutoffs (e.g., 0.52) are too strict when a model head is
        # calibrated lower but still encodes useful relative structure.
        vp99 = float(np.percentile(v_prob, 99.5)) if v_prob.size else 0.0
        hp99 = float(np.percentile(h_prob, 99.5)) if h_prob.size else 0.0
        base_hi = max(0.22, float(self.cfg.line_thresh) + 0.02)
        v_hi = min(0.48, base_hi)
        h_hi = min(0.48, base_hi)
        if vp99 > 1e-6:
            v_hi = min(v_hi, max(0.12, 0.92 * vp99))
        if hp99 > 1e-6:
            h_hi = min(h_hi, max(0.12, 0.92 * hp99))

        def _prep_mask(mask_u8: np.ndarray, line_only: bool) -> np.ndarray:
            if mask_u8.max() <= 0:
                return mask_u8
            if line_only:
                k = max(5, int(round(0.006 * max(h, w))) | 1)
                m = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
                return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            k = max(5, int(round(0.01 * max(h, w))) | 1)
            m = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)), iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
            return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        def _collect(mask_u8: np.ndarray, line_only: bool) -> list[tuple[float, tuple[int, int, int, int], float]]:
            if mask_u8.max() <= 0:
                return []
            n, _, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
            min_area = float(self.cfg.min_table_area_ratio) * float(h * w)
            out: list[tuple[float, tuple[int, int, int, int], float]] = []
            for i in range(1, n):
                x, y, bw, bh, area = stats[i]
                if area < min_area:
                    continue
                min_bw = max(60, int((0.09 if line_only else 0.12) * w))
                min_bh = max(55, int((0.05 if line_only else 0.07) * h))
                if bw < min_bw or bh < min_bh:
                    continue

                x0 = int(max(0, x - round(0.01 * w)))
                y0 = int(max(0, y - round(0.01 * h)))
                x1 = int(min(w - 1, x + bw - 1 + round(0.01 * w)))
                y1 = int(min(h - 1, y + bh - 1 + round(0.01 * h)))

                crop_v = v_prob[y0 : y1 + 1, x0 : x1 + 1]
                crop_h = h_prob[y0 : y1 + 1, x0 : x1 + 1]
                line_energy = 0.5 * (
                    self._strong_ratio(crop_v, v_hi)
                    + self._strong_ratio(crop_h, h_hi)
                )
                fill = float(area) / float(max(1, bw * bh))
                if (not line_only) and fill > float(self.cfg.max_component_fill_ratio) and line_energy < max(
                    float(self.cfg.min_box_line_energy) * 1.8, 0.012
                ):
                    continue
                if line_energy < float(self.cfg.min_box_line_energy):
                    continue
                energy = float(crop_v.mean() + crop_h.mean())
                table_crop = table_prob[y0 : y1 + 1, x0 : x1 + 1]
                t_mean = float(np.mean(table_crop)) if table_crop.size else 0.0
                t_p90 = float(np.percentile(table_crop, 90.0)) if table_crop.size else 0.0
                ar = float(max(1, bw * bh)) / float(max(1, h * w))
                # In line-only mode, favor line support more than component fill/area.
                area_w = 0.50 if line_only else 0.90
                line_w = 2.25 if line_only else 1.15
                score = float(area) * (area_w + line_w * max(line_energy, 0.0) + 0.50 * energy) * (0.40 + 0.90 * fill)
                score *= float(0.55 + 0.80 * t_mean + 0.40 * t_p90)
                # Penalize near-full-frame boxes unless table-confidence support is
                # extremely high. This protects cards like White Horse from
                # over-expanding to page-wide line structures.
                if ar > 0.86 and t_p90 < 0.88:
                    score *= float(max(0.15, 1.0 - 1.8 * (ar - 0.86)))
                if bool(self.cfg.conservative_table_bbox):
                    compact = 1.0 - ar
                    score *= float(1.0 + float(self.cfg.conservative_prefer_compact_bonus) * max(0.0, compact))
                out.append((score, (x0, y0, x1, y1), fill))
            return out

        cands_comb = _collect(_prep_mask(mask_comb, line_only=False), line_only=False)
        cands_line = _collect(_prep_mask(mask_line, line_only=True), line_only=True)

        cands = cands_comb
        if not cands and cands_line:
            cands = cands_line
        elif cands and cands_line:
            # If combined mask is dominated by near-full components, prefer line-only candidates.
            top_fill = max(float(c[2]) for c in cands)
            if top_fill >= min(0.995, float(self.cfg.max_component_fill_ratio) + 0.01) and not bool(self.cfg.conservative_table_bbox):
                cands = cands_line
            else:
                # Also allow extra non-overlapping line-only candidates.
                ext = list(cands)
                for s, b, f in cands_line:
                    if all(_bbox_iou_xyxy(b, cb) < 0.20 for _, cb, _ in cands):
                        ext.append((s, b, f))
                cands = ext

        if not cands:
            return []

        cands.sort(key=lambda t: t[0], reverse=True)
        chosen: list[tuple[int, int, int, int]] = []
        for _, box, _ in cands:
            if bool(self.cfg.conservative_table_bbox):
                x0, y0, x1, y1 = box
                bw = x1 - x0 + 1
                bh = y1 - y0 + 1
                if float(bw) / float(max(1, w)) > float(self.cfg.conservative_max_width_ratio):
                    continue
                if float(bh) / float(max(1, h)) > float(self.cfg.conservative_max_height_ratio):
                    continue
            if any(_bbox_iou_xyxy(box, c) > 0.45 for c in chosen):
                continue
            chosen.append(box)
            if len(chosen) >= int(self.cfg.max_tables):
                break
        if not chosen and bool(self.cfg.conservative_table_bbox):
            # Conservative pass may filter everything; gracefully fallback.
            for _, box, _ in cands:
                if any(_bbox_iou_xyxy(box, c) > 0.45 for c in chosen):
                    continue
                chosen.append(box)
                if len(chosen) >= int(self.cfg.max_tables):
                    break
        if not chosen:
            return []
        chosen.sort(key=lambda b: (b[0], b[1]))

        # Merge vertically stacked slices that likely belong to one table
        # (common when a colored band interrupts the mask).
        merged = list(chosen)
        changed = True
        while changed and len(merged) >= 2:
            changed = False
            out: list[tuple[int, int, int, int]] = []
            used = [False] * len(merged)
            for i in range(len(merged)):
                if used[i]:
                    continue
                ax0, ay0, ax1, ay1 = merged[i]
                used[i] = True
                for j in range(i + 1, len(merged)):
                    if used[j]:
                        continue
                    bx0, by0, bx1, by1 = merged[j]
                    # compare in vertical order without mutating the anchor box
                    tx0, ty0, tx1, ty1 = ax0, ay0, ax1, ay1
                    ux0, uy0, ux1, uy1 = bx0, by0, bx1, by1
                    if uy0 < ty0:
                        tx0, ty0, tx1, ty1, ux0, uy0, ux1, uy1 = ux0, uy0, ux1, uy1, tx0, ty0, tx1, ty1
                    gap = max(0, uy0 - ty1 - 1)
                    xo = max(0, min(tx1, ux1) - max(tx0, ux0) + 1)
                    xw = max(1, max(tx1, ux1) - min(tx0, ux0) + 1)
                    x_overlap = float(xo) / float(xw)
                    small_gap = gap <= max(14, int(0.03 * h))
                    if x_overlap >= 0.82 and small_gap:
                        ax0 = min(ax0, bx0)
                        ay0 = min(ay0, by0)
                        ax1 = max(ax1, bx1)
                        ay1 = max(ay1, by1)
                        used[j] = True
                        changed = True
                out.append((ax0, ay0, ax1, ay1))
            merged = sorted(out, key=lambda b: (b[0], b[1]))

        return merged

    def _axis_open_mask(self, line_prob: np.ndarray, axis: str, t_hi: float, t_lo: float) -> np.ndarray:
        h, w = line_prob.shape[:2]
        hi = (line_prob >= t_hi).astype(np.uint8) * 255
        lo = (line_prob >= t_lo).astype(np.uint8) * 255

        if axis == "x":
            k1 = max(7, int(round(0.08 * h)) | 1)
            k2 = max(11, int(round(0.13 * h)) | 1)
            o1 = cv2.morphologyEx(hi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k1)), iterations=1)
            o2 = cv2.morphologyEx(lo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k2)), iterations=1)
            m = cv2.bitwise_or(o1, o2)
        else:
            k1 = max(7, int(round(0.08 * w)) | 1)
            k2 = max(11, int(round(0.13 * w)) | 1)
            o1 = cv2.morphologyEx(hi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k1, 1)), iterations=1)
            o2 = cv2.morphologyEx(lo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k2, 1)), iterations=1)
            m = cv2.bitwise_or(o1, o2)

        return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    def _axis_component_candidates(self, axis_mask: np.ndarray, axis: str) -> list[tuple[int, float]]:
        h, w = axis_mask.shape[:2]
        n, _, stats, cent = cv2.connectedComponentsWithStats((axis_mask > 0).astype(np.uint8), connectivity=8)
        out: list[tuple[int, float]] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            cx, cy = cent[i]
            if axis == "x":
                span = bh
                thick = bw
                if span < max(12, int(0.22 * h)):
                    continue
                if thick > max(14, int(0.14 * w)):
                    continue
                pos = int(round(cx))
                score = float(span) * (0.4 + float(area) / float(max(1, bw * bh)))
                out.append((pos, score))
            else:
                span = bw
                thick = bh
                if span < max(12, int(0.22 * w)):
                    continue
                if thick > max(14, int(0.14 * h)):
                    continue
                pos = int(round(cy))
                score = float(span) * (0.4 + float(area) / float(max(1, bw * bh)))
                out.append((pos, score))
        return out

    def _cluster_positions(self, items: list[tuple[int, float]], gap: int) -> list[tuple[int, float]]:
        if not items:
            return []
        vals = sorted(items, key=lambda t: t[0])
        groups: list[list[tuple[int, float]]] = []
        for p, s in vals:
            if not groups:
                groups.append([(p, s)])
                continue
            if abs(p - groups[-1][-1][0]) <= gap:
                groups[-1].append((p, s))
            else:
                groups.append([(p, s)])

        out: list[tuple[int, float]] = []
        for g in groups:
            ps = np.array([x[0] for x in g], dtype=np.float32)
            ws = np.array([max(1e-3, x[1]) for x in g], dtype=np.float32)
            p = int(round(float(np.sum(ps * ws) / np.sum(ws))))
            s = float(np.max(ws))
            out.append((p, s))
        return out

    @staticmethod
    def _flex_cov_vertical(v_prob_crop: np.ndarray, x: int, band: int, thr: float, y0: int, y1: int) -> float:
        h, w = v_prob_crop.shape[:2]
        x0 = max(0, int(x) - band)
        x1 = min(w - 1, int(x) + band)
        y0 = max(0, int(y0))
        y1 = min(h, int(y1))
        if y1 <= y0 or x1 < x0:
            return 0.0
        sl = v_prob_crop[y0:y1, x0 : x1 + 1]
        if sl.size == 0:
            return 0.0
        mx = sl.max(axis=1)
        return float((mx >= thr).mean())

    @staticmethod
    def _flex_cov_horizontal(h_prob_crop: np.ndarray, y: int, band: int, thr: float, x0: int, x1: int) -> float:
        h, w = h_prob_crop.shape[:2]
        y0 = max(0, int(y) - band)
        y1 = min(h - 1, int(y) + band)
        x0 = max(0, int(x0))
        x1 = min(w, int(x1))
        if x1 <= x0 or y1 < y0:
            return 0.0
        sl = h_prob_crop[y0 : y1 + 1, x0:x1]
        if sl.size == 0:
            return 0.0
        # Alignment-aware horizontal support:
        # require that per-column peaks are not only strong but also concentrated
        # around a consistent y-offset. This suppresses text baselines/characters.
        mx = sl.max(axis=0)
        valid = mx >= thr
        if not np.any(valid):
            return 0.0
        y_arg = sl.argmax(axis=0).astype(np.float32)
        med = float(np.median(y_arg[valid]))
        align_tol = max(1.0, float(band) * 0.34)
        aligned = valid & (np.abs(y_arg - med) <= align_tol)
        cov = float(aligned.mean())

        # Penalize highly jittery peak positions across columns.
        if np.any(aligned):
            std = float(np.std(y_arg[aligned]))
            jitter_pen = max(0.0, 1.0 - 0.35 * std / max(1.0, float(band)))
            cov *= jitter_pen
        return cov

    def _regularize_lines(self, cands: list[tuple[int, float]], proj: np.ndarray, low: int, high: int, min_gap: int, max_lines: int) -> list[int]:
        items = [(int(np.clip(p, low, high)), float(s)) for p, s in cands]
        items.append((low, 1e9))
        items.append((high, 1e9))
        items = self._cluster_positions(items, gap=max(1, min_gap // 2))

        # Deduplicate near positions by keeping stronger score.
        items = sorted(items, key=lambda t: t[0])
        dedup: list[tuple[int, float]] = []
        for p, s in items:
            if not dedup:
                dedup.append((p, s))
                continue
            q, qs = dedup[-1]
            if p - q < min_gap and p not in (low, high) and q not in (low, high):
                if s > qs:
                    dedup[-1] = (p, s)
            else:
                dedup.append((p, s))

        vals = sorted(set(p for p, _ in dedup))
        if low not in vals:
            vals.insert(0, low)
        if high not in vals:
            vals.append(high)

        # Optional prior-only fill for very large gaps.
        # Disabled by default: it can over-split merged cells when no line evidence exists.
        if bool(self.cfg.prior_gap_fill_enable) and len(vals) >= 3:
            gaps = np.diff(np.array(vals, dtype=np.int32))
            pitch = float(np.median(gaps)) if gaps.size else float(min_gap)
            pitch = max(float(min_gap), pitch)
            inserts: list[int] = []
            p60 = float(np.percentile(proj, 60.0)) if proj.size else 0.0
            p90 = float(np.percentile(proj, 90.0)) if proj.size else 0.0
            p_req = p60 + 0.22 * max(0.0, p90 - p60)
            for i, g in enumerate(gaps):
                if g <= 2.25 * pitch:
                    continue
                n_new = max(1, min(3, int(round(float(g) / pitch)) - 1))
                for k in range(n_new):
                    v = int(round(vals[i] + (k + 1) * g / float(n_new + 1)))
                    if low < v < high:
                        lo = max(low, v - 1)
                        hi = min(high, v + 1)
                        pv = float(np.mean(proj[lo : hi + 1])) if 0 <= lo <= hi < proj.size else 0.0
                        if pv >= p_req:
                            inserts.append(v)
            if inserts:
                vals = sorted(set(vals + inserts))

        vals = _prune_near(vals, max(2, int(round(0.65 * min_gap))))

        if len(vals) > max_lines:
            interior = [v for v in vals if v not in (low, high)]
            scored: list[tuple[float, int]] = []
            for v in interior:
                lo = max(low, v - 1)
                hi = min(high, v + 1)
                sv = float(np.mean(proj[lo : hi + 1])) if 0 <= lo <= hi < proj.size else 0.0
                scored.append((sv, v))
            scored.sort(key=lambda t: t[0], reverse=True)
            keep_inner = sorted(v for _, v in scored[: max(0, max_lines - 2)])
            vals = [low] + keep_inner + [high]

        vals = _prune_near(vals, max(2, int(round(0.58 * min_gap))))
        if len(vals) < 2:
            vals = [low, high]
        vals[0] = low
        vals[-1] = high
        return vals

    def _optimize_axis_lines_dp(
        self,
        axis: str,
        cand_positions: list[int],
        line_prob_crop: np.ndarray,
        junc_prob_crop: np.ndarray,
        proj: np.ndarray,
        line_thr: float,
        min_gap: int,
        min_lines: int,
        max_lines: int,
        band: int,
        soft_prob_crop: Optional[np.ndarray] = None,
    ) -> list[int]:
        h, w = line_prob_crop.shape[:2]
        if h < 2 or w < 2:
            return [0, max(1, (w - 1) if axis == "x" else (h - 1))]

        low = 0
        high = (w - 1) if axis == "x" else (h - 1)
        vals = sorted(set(int(np.clip(v, low, high)) for v in cand_positions))
        if low not in vals:
            vals.insert(0, low)
        if high not in vals:
            vals.append(high)
        vals = _prune_near(vals, max(1, int(round(0.45 * max(1, min_gap)))))
        if not vals or vals[0] != low:
            vals = [low] + [v for v in vals if v != low]
        if vals[-1] != high:
            vals.append(high)

        def _evidence(positions: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            hard = np.zeros((len(positions),), dtype=np.float32)
            soft = np.zeros((len(positions),), dtype=np.float32)
            pnorm = np.zeros((len(positions),), dtype=np.float32)
            p_den = float(np.percentile(proj, 98.0)) if proj.size else 1.0
            p_den = max(p_den, 1e-6)
            for ii, p in enumerate(positions):
                pv = float(np.clip(float(proj[p]) / p_den, 0.0, 1.0)) if 0 <= p < proj.size else 0.0
                pnorm[ii] = pv
                if axis == "x":
                    cov = self._flex_cov_vertical(line_prob_crop, p, band=band, thr=line_thr * 0.92, y0=0, y1=h)
                    px0 = max(0, p - band)
                    px1 = min(w - 1, p + band)
                    js = float(junc_prob_crop[:, px0 : px1 + 1].max(axis=1).mean()) if px1 >= px0 else 0.0
                    hard[ii] = float(0.56 * cov + 0.28 * pv + 0.16 * js)
                    if soft_prob_crop is not None and soft_prob_crop.shape[:2] == line_prob_crop.shape[:2]:
                        sb = max(1, int(round(0.55 * band)))
                        scov = self._flex_cov_vertical(soft_prob_crop, p, band=sb, thr=0.35, y0=0, y1=h)
                        sx0 = max(0, p - sb)
                        sx1 = min(w - 1, p + sb)
                        sseg = (
                            np.max(soft_prob_crop[:, sx0 : sx1 + 1], axis=1) >= 0.35
                        ) if sx1 >= sx0 else np.zeros((0,), dtype=np.uint8)
                        srun = _max_true_run_ratio(sseg)
                        soft[ii] = float(0.70 * scov + 0.22 * srun + 0.08 * js)
                else:
                    cov = self._flex_cov_horizontal(line_prob_crop, p, band=band, thr=line_thr * 0.92, x0=0, x1=w)
                    py0 = max(0, p - band)
                    py1 = min(h - 1, p + band)
                    js = float(junc_prob_crop[py0 : py1 + 1, :].max(axis=0).mean()) if py1 >= py0 else 0.0
                    hard[ii] = float(0.58 * cov + 0.24 * pv + 0.18 * js)
                    if soft_prob_crop is not None:
                        sb = max(1, int(round(0.55 * band)))
                        sthr = max(0.34, float(self.cfg.h_color_peak_min) - 0.08)
                        scov = self._flex_cov_horizontal(soft_prob_crop, p, band=sb, thr=sthr, x0=0, x1=w)
                        soft[ii] = float(0.82 * scov + 0.18 * js)
            return hard, soft, pnorm

        hard, soft, pnorm = _evidence(vals)

        # Candidate pruning for tractable global search.
        max_cands = max(12, int(self.cfg.axis_dp_max_candidates))
        if len(vals) > max_cands:
            keep_n = max(0, max_cands - 2)
            interior = list(range(1, len(vals) - 1))
            sw = float(self.cfg.axis_dp_soft_h_weight) if axis == "y" else float(self.cfg.axis_dp_soft_v_weight)
            comb = [(float(hard[i] + sw * soft[i] + 0.06 * pnorm[i]), i) for i in interior]
            comb.sort(key=lambda t: t[0], reverse=True)
            keep_idx = sorted([idx for _, idx in comb[:keep_n]])
            vals = [vals[0]] + [vals[i] for i in keep_idx] + [vals[-1]]
            vals = _prune_near(vals, max(1, int(round(0.45 * max(1, min_gap)))))
            if vals[0] != low:
                vals = [low] + [v for v in vals if v != low]
            if vals[-1] != high:
                vals.append(high)
            hard, soft, pnorm = _evidence(vals)

        m = len(vals)
        if m < 2:
            return [low, high]

        k_min = int(max(2, min_lines))
        k_max = int(min(max_lines, m))
        if k_min > k_max:
            k_min = k_max
        if k_max < 2:
            return [low, high]

        sw = float(self.cfg.axis_dp_soft_h_weight) if axis == "y" else float(self.cfg.axis_dp_soft_v_weight)
        node = np.zeros((m,), dtype=np.float32)
        for i in range(1, m - 1):
            node[i] = float(hard[i] + sw * soft[i] - float(self.cfg.axis_dp_node_cost))
        node[0] = 0.0
        node[-1] = 0.0

        best_score = -1e18
        best_path: Optional[list[int]] = None
        k_target = 0.5 * float(k_min + k_max)

        for k in range(k_min, k_max + 1):
            dp = np.full((k + 1, m, m), -1e18, dtype=np.float32)
            prv = np.full((k + 1, m, m), -1, dtype=np.int16)

            for j in range(1, m):
                gap = vals[j] - vals[0]
                if gap < min_gap:
                    continue
                if j == m - 1 and k > 2:
                    continue
                if (m - 1 - j) < (k - 2):
                    continue
                dp[2, 0, j] = node[j]

            for t in range(2, k):
                rem_after = k - (t + 1)
                for i in range(0, m - 1):
                    for j in range(i + 1, m):
                        s = float(dp[t, i, j])
                        if s <= -1e17:
                            continue
                        gap_prev = vals[j] - vals[i]
                        if gap_prev < min_gap:
                            continue
                        for kk in range(j + 1, m):
                            gap = vals[kk] - vals[j]
                            if gap < min_gap:
                                continue
                            if (m - 1 - kk) < rem_after:
                                continue
                            if (t + 1) == k and kk != (m - 1):
                                continue
                            if kk == (m - 1) and (t + 1) < k:
                                continue

                            smooth = abs(float(gap - gap_prev)) / max(1.0, float(min(gap, gap_prev)))
                            big_pen = max(0.0, float(gap) / max(1.0, float(gap_prev)) - 2.2)
                            val = (
                                s
                                + float(node[kk])
                                - float(self.cfg.axis_dp_gap_smooth_w) * smooth
                                - float(self.cfg.axis_dp_big_gap_w) * big_pen
                            )
                            if val > float(dp[t + 1, j, kk]):
                                dp[t + 1, j, kk] = val
                                prv[t + 1, j, kk] = i

            for i in range(0, m - 1):
                s = float(dp[k, i, m - 1])
                if s <= -1e17:
                    continue
                s -= float(self.cfg.axis_dp_count_penalty) * abs(float(k) - k_target)
                if s <= best_score:
                    continue

                path_rev = [m - 1, i]
                ok = True
                tcur = k
                icur = i
                jcur = m - 1
                while tcur > 2:
                    pidx = int(prv[tcur, icur, jcur])
                    if pidx < 0:
                        ok = False
                        break
                    path_rev.append(pidx)
                    jcur = icur
                    icur = pidx
                    tcur -= 1
                path_rev.append(0)
                if not ok:
                    continue
                path = list(reversed(path_rev))
                if path[0] != 0 or path[-1] != (m - 1):
                    continue
                best_score = s
                best_path = path

        if not best_path:
            return []

        out = [vals[i] for i in best_path]
        out = _prune_near(sorted(set(out)), max(2, int(round(0.58 * max(1, min_gap)))))
        if not out or out[0] != low:
            out = [low] + [v for v in out if v != low]
        if out[-1] != high:
            out.append(high)
        if len(out) < 2:
            return [low, high]
        return out

    def _decode_axis_lines(
        self,
        line_prob_crop: np.ndarray,
        junc_prob_crop: np.ndarray,
        axis: str,
        line_thr: float,
        min_cov: float,
        peak_thr: float,
        min_gap: int,
        min_lines: int,
        max_lines: int,
        band: int,
        soft_prob_crop: Optional[np.ndarray] = None,
    ) -> list[int]:
        h, w = line_prob_crop.shape[:2]
        if h < 2 or w < 2:
            return [0, max(1, (w - 1) if axis == "x" else (h - 1))]

        low_thr = max(0.20, line_thr - 0.14)
        axis_mask = self._axis_open_mask(line_prob_crop, axis=axis, t_hi=line_thr, t_lo=low_thr)

        if axis == "x":
            proj_prob = line_prob_crop.mean(axis=0).astype(np.float32)
            proj_mask = (axis_mask > 0).mean(axis=0).astype(np.float32)
            proj = 0.72 * _smooth_1d(proj_prob, max(7, int(round(0.02 * w)) | 1)) + 0.28 * _smooth_1d(
                proj_mask, max(7, int(round(0.02 * w)) | 1)
            )
            soft_proj: Optional[np.ndarray] = None
            if soft_prob_crop is not None and soft_prob_crop.shape[:2] == line_prob_crop.shape[:2]:
                soft_proj = _smooth_1d(
                    soft_prob_crop.mean(axis=0).astype(np.float32),
                    max(7, int(round(0.02 * w)) | 1),
                )
                proj = (0.58 * proj + 0.42 * soft_proj).astype(np.float32)

            dyn_gap = max(min_gap, int(round(0.024 * w)))
            peaks = _projection_peaks(proj, rel_thresh=peak_thr, min_gap=dyn_gap)
            cands = self._axis_component_candidates(axis_mask, axis="x")
            for p in peaks:
                cov = self._flex_cov_vertical(line_prob_crop, p, band=band, thr=line_thr * 0.90, y0=0, y1=h)
                if cov >= max(0.12, 0.75 * min_cov):
                    px0 = max(0, p - band)
                    px1 = min(w - 1, p + band)
                    jsup = float(junc_prob_crop[:, px0 : px1 + 1].max(axis=1).mean()) if px1 >= px0 else 0.0
                    scov = 0.0
                    if soft_prob_crop is not None and soft_prob_crop.shape[:2] == line_prob_crop.shape[:2]:
                        sb = max(1, int(round(0.55 * band)))
                        scov = self._flex_cov_vertical(soft_prob_crop, p, band=sb, thr=0.35, y0=0, y1=h)
                    s = float(proj[p]) + 1.30 * cov + 0.50 * jsup + 0.95 * scov
                    cands.append((int(p), s))

            if soft_proj is not None:
                s_peaks = _projection_peaks(
                    soft_proj,
                    rel_thresh=max(0.07, 0.75 * peak_thr),
                    min_gap=max(4, int(round(0.65 * dyn_gap))),
                )
                for p in s_peaks:
                    cands.append((int(p), float(0.55 * soft_proj[p])))

            base = self._regularize_lines(cands, proj=proj, low=0, high=w - 1, min_gap=dyn_gap, max_lines=max_lines)
            if not bool(self.cfg.axis_dp_enable):
                return base
            cand_pos = [int(p) for p, _ in cands] + list(base)
            opt = self._optimize_axis_lines_dp(
                axis="x",
                cand_positions=cand_pos,
                line_prob_crop=line_prob_crop,
                junc_prob_crop=junc_prob_crop,
                proj=proj,
                line_thr=line_thr,
                min_gap=dyn_gap,
                min_lines=min_lines,
                max_lines=max_lines,
                band=band,
                soft_prob_crop=soft_prob_crop,
            )
            return opt if len(opt) >= 2 else base

        proj_prob = line_prob_crop.mean(axis=1).astype(np.float32)
        proj_mask = (axis_mask > 0).mean(axis=1).astype(np.float32)
        proj = 0.72 * _smooth_1d(proj_prob, max(7, int(round(0.02 * h)) | 1)) + 0.28 * _smooth_1d(
            proj_mask, max(7, int(round(0.02 * h)) | 1)
        )

        dyn_gap = max(min_gap, int(round(0.024 * h)))
        peaks = _projection_peaks(proj, rel_thresh=peak_thr, min_gap=dyn_gap)
        cands = self._axis_component_candidates(axis_mask, axis="y")
        for p in peaks:
            cov = self._flex_cov_horizontal(line_prob_crop, p, band=band, thr=line_thr * 0.90, x0=0, x1=w)
            if cov >= max(0.12, 0.75 * min_cov):
                py0 = max(0, p - band)
                py1 = min(h - 1, p + band)
                jsup = float(junc_prob_crop[py0 : py1 + 1, :].max(axis=0).mean()) if py1 >= py0 else 0.0
                s = float(proj[p]) + 1.45 * cov + 0.50 * jsup
                cands.append((int(p), s))

        # Soft horizontal boundary candidates (color transitions) for rows only.
        if soft_prob_crop is not None and soft_prob_crop.shape[:2] == line_prob_crop.shape[:2]:
            soft_proj = _smooth_1d(
                soft_prob_crop.mean(axis=1).astype(np.float32),
                max(7, int(round(0.02 * h)) | 1),
            )
            s_peaks = _projection_peaks(soft_proj, rel_thresh=max(0.07, 0.75 * peak_thr), min_gap=max(4, int(round(0.65 * dyn_gap))))
            for p in s_peaks:
                cands.append((int(p), float(0.45 * soft_proj[p])))

        base = self._regularize_lines(cands, proj=proj, low=0, high=h - 1, min_gap=dyn_gap, max_lines=max_lines)
        if not bool(self.cfg.axis_dp_enable):
            return base
        cand_pos = [int(p) for p, _ in cands] + list(base)
        opt = self._optimize_axis_lines_dp(
            axis="y",
            cand_positions=cand_pos,
            line_prob_crop=line_prob_crop,
            junc_prob_crop=junc_prob_crop,
            proj=proj,
            line_thr=line_thr,
            min_gap=dyn_gap,
            min_lines=min_lines,
            max_lines=max_lines,
            band=band,
            soft_prob_crop=soft_prob_crop,
        )
        return opt if len(opt) >= 2 else base

    def _snap_all_lines_local(
        self,
        x_lines: list[int],
        y_lines: list[int],
        v_prob_crop: np.ndarray,
        h_prob_crop: np.ndarray,
        j_prob_crop: Optional[np.ndarray],
        line_thr: float,
        v_line_prob_crop: Optional[np.ndarray] = None,
    ) -> tuple[list[int], list[int]]:
        if len(x_lines) < 2 or len(y_lines) < 2:
            return x_lines, y_lines
        h, w = v_prob_crop.shape[:2]
        if h < 4 or w < 4:
            return x_lines, y_lines

        band = max(2, int(self.cfg.flexible_band_px))
        band_pos = max(2, min(5, int(round(0.45 * float(band)))))
        win_x = max(2, int(round(0.008 * w)))
        win_y = max(2, int(round(0.008 * h)))
        thr = max(0.20, float(line_thr) * 0.90)
        v_proj = v_prob_crop.mean(axis=0).astype(np.float32)
        h_proj = h_prob_crop.mean(axis=1).astype(np.float32)
        l_proj = None
        if v_line_prob_crop is not None and v_line_prob_crop.shape[:2] == v_prob_crop.shape[:2]:
            l_proj = v_line_prob_crop.mean(axis=0).astype(np.float32)
        v_mx = max(1e-6, float(np.max(v_proj)) if v_proj.size else 1.0)
        h_mx = max(1e-6, float(np.max(h_proj)) if h_proj.size else 1.0)
        l_mx = max(1e-6, float(np.max(l_proj)) if l_proj is not None and l_proj.size else 1.0)

        xs = list(int(v) for v in x_lines)
        ys = list(int(v) for v in y_lines)
        j_crop = j_prob_crop if (j_prob_crop is not None and j_prob_crop.shape[:2] == v_prob_crop.shape[:2]) else np.zeros_like(v_prob_crop, dtype=np.float32)
        has_line_map = v_line_prob_crop is not None and v_line_prob_crop.shape[:2] == v_prob_crop.shape[:2]

        for i in range(len(xs)):
            x = int(xs[i])
            lo = max(0, x - win_x)
            hi = min(w - 1, x + win_x)
            if i > 0:
                lo = max(lo, int(xs[i - 1]) + 2)
            if i + 1 < len(xs):
                hi = min(hi, int(xs[i + 1]) - 2)
            if hi < lo:
                continue
            best_x = x
            best_s = -1e18
            for p in range(lo, hi + 1):
                cov = self._flex_cov_vertical(v_prob_crop, x=p, band=band_pos, thr=thr, y0=0, y1=h)
                row_s, sup_frac, _ = self._vertical_line_row_evidence(
                    x=p,
                    y_lines=y_lines,
                    v_prob_crop=v_prob_crop,
                    j_prob_crop=j_crop,
                    line_thr=line_thr,
                    band=band,
                    v_line_crop=v_line_prob_crop,
                )
                lcov = 0.0
                lrun = 0.0
                lp = 0.0
                if has_line_map:
                    sb = max(1, int(round(0.55 * band_pos)))
                    lcov = self._flex_cov_vertical(v_line_prob_crop, x=p, band=sb, thr=0.35, y0=0, y1=h)
                    sx0 = max(0, p - sb)
                    sx1 = min(w - 1, p + sb)
                    sseg = (
                        np.max(v_line_prob_crop[:, sx0 : sx1 + 1], axis=1) >= 0.35
                    ) if sx1 >= sx0 else np.zeros((0,), dtype=np.uint8)
                    lrun = _max_true_run_ratio(sseg)
                    lp = float(l_proj[p]) / l_mx if l_proj is not None else 0.0
                vp = float(v_proj[p]) / v_mx
                s = (
                    0.34 * cov
                    + 0.31 * row_s
                    + 0.13 * sup_frac
                    + 0.10 * vp
                    + 0.08 * lcov
                    + 0.04 * lrun
                )
                if has_line_map:
                    s += 0.06 * lp
                # Keep local continuity unless evidence is clearly better.
                s -= 0.0012 * abs(p - x)
                if s > best_s:
                    best_s = s
                    best_x = p
            xs[i] = int(best_x)

        for j in range(len(ys)):
            y = int(ys[j])
            lo = max(0, y - win_y)
            hi = min(h - 1, y + win_y)
            if j > 0:
                lo = max(lo, int(ys[j - 1]) + 2)
            if j + 1 < len(ys):
                hi = min(hi, int(ys[j + 1]) - 2)
            if hi < lo:
                continue
            best_y = y
            best_s = -1e18
            for p in range(lo, hi + 1):
                cov = self._flex_cov_horizontal(h_prob_crop, y=p, band=band_pos, thr=thr, x0=0, x1=w)
                s = 0.80 * cov + 0.20 * (float(h_proj[p]) / h_mx)
                if s > best_s:
                    best_s = s
                    best_y = p
            ys[j] = int(best_y)

        xs = _prune_near(sorted(set(xs)), max(2, int(round(0.58 * max(1, self.cfg.min_gap_px)))))
        ys = _prune_near(sorted(set(ys)), max(2, int(round(0.58 * max(1, self.cfg.min_gap_px)))))
        return xs, ys

    def _score_grid(
        self,
        x_lines: list[int],
        y_lines: list[int],
        v_prob_crop: np.ndarray,
        h_prob_crop: np.ndarray,
        j_prob_crop: np.ndarray,
        line_thr: float,
        band: int,
    ) -> float:
        cols = max(0, len(x_lines) - 1)
        rows = max(0, len(y_lines) - 1)
        if rows <= 0 or cols <= 0:
            return -1e9

        score = 0.0

        if self.cfg.min_rows <= rows <= self.cfg.max_rows:
            score += 2.2
        else:
            score -= 0.45 * abs(rows - np.clip(rows, self.cfg.min_rows, self.cfg.max_rows))
        if self.cfg.min_cols <= cols <= self.cfg.max_cols:
            score += 2.2
        else:
            score -= 0.45 * abs(cols - np.clip(cols, self.cfg.min_cols, self.cfg.max_cols))

        # Keep soft priors only; rely mainly on line evidence.
        score -= 0.10 * abs(rows - 12)
        score -= 0.08 * abs(cols - 16)

        if len(x_lines) >= 4:
            gx = np.diff(np.array(x_lines, dtype=np.float32))
            mx = float(np.mean(gx))
            if mx > 1e-6:
                score -= 1.15 * float(np.std(gx) / mx)
        if len(y_lines) >= 4:
            gy = np.diff(np.array(y_lines, dtype=np.float32))
            my = float(np.mean(gy))
            if my > 1e-6:
                score -= 1.15 * float(np.std(gy) / my)

        # Flexible line support across full spans.
        v_cov = [self._flex_cov_vertical(v_prob_crop, x, band=band, thr=line_thr, y0=0, y1=v_prob_crop.shape[0]) for x in x_lines]
        h_cov = [
            self._flex_cov_horizontal(h_prob_crop, y, band=band, thr=line_thr, x0=0, x1=h_prob_crop.shape[1])
            for y in y_lines
        ]
        mean_v = float(np.mean(v_cov)) if v_cov else 0.0
        mean_h = float(np.mean(h_cov)) if h_cov else 0.0
        low_v = float(np.percentile(v_cov, 25.0)) if v_cov else 0.0
        low_h = float(np.percentile(h_cov, 25.0)) if h_cov else 0.0
        min_v = float(np.min(v_cov)) if v_cov else 0.0
        min_h = float(np.min(h_cov)) if h_cov else 0.0
        score += 5.2 * mean_v
        score += 5.2 * mean_h
        score += 1.2 * low_v
        score += 1.2 * low_h
        score += 2.8 * min_v
        score += 2.8 * min_h
        score -= 4.0 * max(0.0, 0.20 - low_v)
        score -= 4.0 * max(0.0, 0.20 - low_h)

        # Junction agreement at inferred intersections.
        j_sum = 0.0
        n_j = 0
        for y in y_lines:
            y0 = max(0, y - band)
            y1 = min(j_prob_crop.shape[0] - 1, y + band)
            for x in x_lines:
                x0 = max(0, x - band)
                x1 = min(j_prob_crop.shape[1] - 1, x + band)
                patch = j_prob_crop[y0 : y1 + 1, x0 : x1 + 1]
                if patch.size:
                    j_sum += float(np.max(patch))
                    n_j += 1
        if n_j > 0:
            score += 2.6 * (j_sum / float(n_j))

        if rows < 3 or cols < 3:
            score -= 3.0
        return score

    def _snap_outer_lines(
        self,
        x_lines: list[int],
        y_lines: list[int],
        v_prob_crop: np.ndarray,
        h_prob_crop: np.ndarray,
        line_thr: float,
        v_line_prob_crop: Optional[np.ndarray] = None,
    ) -> tuple[list[int], list[int]]:
        if len(x_lines) < 2 or len(y_lines) < 2:
            return x_lines, y_lines
        h, w = v_prob_crop.shape[:2]
        if h < 4 or w < 4:
            return x_lines, y_lines

        win_x = max(8, int(round(float(self.cfg.border_snap_window_ratio) * w)))
        win_y = max(8, int(round(float(self.cfg.border_snap_window_ratio) * h)))
        band = max(2, int(self.cfg.flexible_band_px))
        min_cov = float(max(0.08, self.cfg.border_snap_min_cov))

        v_proj = v_prob_crop.mean(axis=0).astype(np.float32)
        h_proj = h_prob_crop.mean(axis=1).astype(np.float32)
        l_proj = None
        has_line_map = v_line_prob_crop is not None and v_line_prob_crop.shape[:2] == v_prob_crop.shape[:2]
        if has_line_map:
            l_proj = v_line_prob_crop.mean(axis=0).astype(np.float32)

        def _best_x(lo: int, hi: int) -> Optional[int]:
            lo = max(0, lo)
            hi = min(w - 1, hi)
            if hi < lo:
                return None
            best = None
            best_s = -1e9
            mx = float(np.max(v_proj[lo : hi + 1])) if hi >= lo else 1.0
            mx = max(mx, 1e-6)
            lmx = float(np.max(l_proj[lo : hi + 1])) if (has_line_map and l_proj is not None and hi >= lo) else 1.0
            lmx = max(lmx, 1e-6)
            for p in range(lo, hi + 1):
                cov = self._flex_cov_vertical(v_prob_crop, p, band=band, thr=line_thr * 0.88, y0=0, y1=h)
                if cov < min_cov:
                    continue
                scov = 0.0
                srun = 0.0
                sp = 0.0
                if has_line_map and l_proj is not None:
                    sb = max(1, int(round(0.55 * band)))
                    scov = self._flex_cov_vertical(v_line_prob_crop, p, band=sb, thr=0.35, y0=0, y1=h)
                    sx0 = max(0, p - sb)
                    sx1 = min(w - 1, p + sb)
                    sseg = (
                        np.max(v_line_prob_crop[:, sx0 : sx1 + 1], axis=1) >= 0.35
                    ) if sx1 >= sx0 else np.zeros((0,), dtype=np.uint8)
                    srun = _max_true_run_ratio(sseg)
                    sp = float(l_proj[p]) / lmx
                s = 0.56 * cov + 0.24 * (float(v_proj[p]) / mx) + 0.14 * scov + 0.04 * srun + 0.02 * sp
                if s > best_s:
                    best_s = s
                    best = p
            return best

        def _best_y(lo: int, hi: int) -> Optional[int]:
            lo = max(0, lo)
            hi = min(h - 1, hi)
            if hi < lo:
                return None
            best = None
            best_s = -1e9
            mx = float(np.max(h_proj[lo : hi + 1])) if hi >= lo else 1.0
            mx = max(mx, 1e-6)
            for p in range(lo, hi + 1):
                cov = self._flex_cov_horizontal(h_prob_crop, p, band=band, thr=line_thr * 0.88, x0=0, x1=w)
                if cov < min_cov:
                    continue
                s = 0.72 * cov + 0.28 * (float(h_proj[p]) / mx)
                if s > best_s:
                    best_s = s
                    best = p
            return best

        xs = list(x_lines)
        ys = list(y_lines)
        lx = _best_x(0, win_x)
        rx = _best_x(max(0, w - 1 - win_x), w - 1)
        ty = _best_y(0, win_y)
        by = _best_y(max(0, h - 1 - win_y), h - 1)
        if lx is not None:
            xs[0] = int(lx)
        if rx is not None:
            xs[-1] = int(rx)
        if ty is not None:
            ys[0] = int(ty)
        if by is not None:
            ys[-1] = int(by)

        xs = _prune_near(sorted(set(xs)), max(2, int(round(0.58 * max(1, self.cfg.min_gap_px)))))
        ys = _prune_near(sorted(set(ys)), max(2, int(round(0.58 * max(1, self.cfg.min_gap_px)))))
        return xs, ys

    def _refine_horizontal_lines_local(
        self,
        x_lines: list[int],
        y_lines: list[int],
        v_prob_crop: np.ndarray,
        h_prob_crop: np.ndarray,
        j_prob_crop: np.ndarray,
        h_color_crop: Optional[np.ndarray],
        line_thr: float,
    ) -> list[int]:
        if len(y_lines) < 2 or len(x_lines) < 2:
            return y_lines

        h, w = h_prob_crop.shape[:2]
        band = max(2, int(self.cfg.flexible_band_px))
        min_gap = max(6, int(self.cfg.min_gap_px))

        def _h_score(y: int) -> float:
            cov = self._flex_cov_horizontal(h_prob_crop, y=y, band=band, thr=line_thr, x0=0, x1=w)
            # Cross-support from vertical and junction maps at candidate intersections.
            vxs = []
            jxs = []
            for x in x_lines:
                x0 = max(0, int(x) - band)
                x1 = min(w - 1, int(x) + band)
                y0 = max(0, int(y) - band)
                y1 = min(h - 1, int(y) + band)
                pv = v_prob_crop[y0 : y1 + 1, x0 : x1 + 1]
                pj = j_prob_crop[y0 : y1 + 1, x0 : x1 + 1]
                if pv.size:
                    vxs.append(float(np.max(pv)))
                if pj.size:
                    jxs.append(float(np.max(pj)))
            v_cross = float(np.mean(vxs)) if vxs else 0.0
            j_cross = float(np.mean(jxs)) if jxs else 0.0
            c_edge = 0.0
            if h_color_crop is not None and 0 <= y < h_color_crop.shape[0]:
                c_edge = float(np.mean(h_color_crop[max(0, y - 1) : min(h_color_crop.shape[0], y + 2), :]))
            # Line evidence dominates; color edge is only a weak tie-breaker.
            return 0.58 * cov + 0.22 * v_cross + 0.16 * j_cross + 0.04 * c_edge

        def _j_hit(y: int) -> float:
            hits = 0
            tot = 0
            for x in x_lines:
                x0 = max(0, int(x) - band)
                x1 = min(w - 1, int(x) + band)
                y0 = max(0, int(y) - band)
                y1 = min(h - 1, int(y) + band)
                pj = j_prob_crop[y0 : y1 + 1, x0 : x1 + 1]
                if pj.size == 0:
                    continue
                tot += 1
                if float(np.max(pj)) >= max(0.33, line_thr - 0.08):
                    hits += 1
            if tot <= 0:
                return 0.0
            return float(hits) / float(tot)

        def _color_cov(y: int) -> float:
            if h_color_crop is None or not (0 <= y < h_color_crop.shape[0]):
                return 0.0
            row = h_color_crop[max(0, y - 1) : min(h_color_crop.shape[0], y + 2), :]
            if row.size == 0:
                return 0.0
            v = np.max(row, axis=0)
            return float((v >= max(0.28, float(self.cfg.h_color_peak_min) - 0.06)).mean())

        # 1) Prune weak interior lines (common false positives from text baselines).
        ys = sorted(set(int(y) for y in y_lines))
        if len(ys) >= 3:
            keep = [ys[0]]
            for y in ys[1:-1]:
                s = _h_score(y)
                # Strong top-band guard: text baselines in first data row often appear here.
                top_guard = max(min_gap, int(round(float(self.cfg.top_hline_guard_ratio) * h)))
                if y <= ys[0] + top_guard:
                    if _color_cov(int(y)) < max(0.45, float(self.cfg.color_line_min_cov) - 0.08):
                        continue
                if s >= float(self.cfg.weak_hline_prune_thresh) and _j_hit(int(y)) >= max(0.36, float(self.cfg.min_hline_junction_hit) - 0.08):
                    keep.append(y)
            keep.append(ys[-1])
            ys = _prune_near(sorted(set(keep)), max(2, int(round(0.58 * min_gap))))
            if len(ys) < 2:
                ys = sorted(set(int(y) for y in y_lines))

        # 2) Fill obvious missing lines inside large row gaps.
        if len(ys) >= 3:
            gaps = np.diff(np.array(ys, dtype=np.int32))
            med_gap = float(np.median(gaps)) if gaps.size else float(min_gap)
            med_gap = max(float(min_gap), med_gap)
            insertions: list[int] = []
            for i, g in enumerate(gaps):
                if float(g) < float(self.cfg.gap_fill_ratio_h) * med_gap:
                    continue
                lo = ys[i] + min_gap
                hi = ys[i + 1] - min_gap
                if hi <= lo:
                    continue
                # Candidate search in horizontal evidence projection.
                proj = h_prob_crop.mean(axis=1).astype(np.float32)
                seg = proj[lo : hi + 1]
                if seg.size < 3:
                    continue

                cand_set: set[int] = set()
                # Always include strongest raw peak.
                cand_set.add(int(lo + int(np.argmax(seg))))
                # Also include local peaks for large gaps.
                peak_min_gap = max(4, int(round(0.55 * med_gap)))
                rel_thr = max(0.08, float(self.cfg.peak_rel_thresh) * 0.75)
                loc_peaks = _projection_peaks(seg, rel_thresh=rel_thr, min_gap=peak_min_gap)
                for p in loc_peaks:
                    cand_set.add(int(lo + int(p)))

                # Color-boundary candidate for band transitions.
                col_proj: Optional[np.ndarray] = None
                if h_color_crop is not None:
                    col_proj = h_color_crop.mean(axis=1).astype(np.float32)
                    cseg = col_proj[lo : hi + 1]
                    if cseg.size >= 3:
                        yc = int(lo + int(np.argmax(cseg)))
                        cpk = float(cseg.max())
                        if cpk >= float(self.cfg.h_color_peak_min):
                            cand_set.add(int(yc))

                scored: list[tuple[float, int]] = []
                large_gap = float(g) >= 1.65 * med_gap
                # For very large gaps, ignore candidates hugging either edge;
                # these are often text/border artifacts.
                edge_margin = max(min_gap, int(round(0.14 * float(g)))) if large_gap else min_gap
                for yc in sorted(cand_set):
                    if yc <= lo or yc >= hi:
                        continue
                    if large_gap:
                        if (yc - ys[i]) < edge_margin or (ys[i + 1] - yc) < edge_margin:
                            continue
                    sc = _h_score(int(yc))
                    covh = self._flex_cov_horizontal(h_prob_crop, y=yc, band=band, thr=line_thr, x0=0, x1=w)
                    jhc = _j_hit(int(yc))
                    covc = _color_cov(int(yc))

                    strong_line_ok = (
                        sc >= max(0.22, float(self.cfg.weak_hline_prune_thresh) - 0.02)
                        and covh >= max(0.16, 0.74 * float(self.cfg.min_line_cov))
                        and jhc >= max(0.34, float(self.cfg.min_hline_junction_hit) - 0.10)
                    )
                    large_gap_color_ok = (
                        large_gap
                        and covc >= max(0.60, float(self.cfg.color_line_min_cov) - 0.04)
                        and covh >= max(0.06, 0.35 * float(self.cfg.min_line_cov))
                        and jhc >= max(0.55, float(self.cfg.min_hline_junction_hit) + 0.02)
                    )
                    if not (strong_line_ok or large_gap_color_ok):
                        continue

                    qual = float(sc + 0.35 * covh + 0.25 * jhc + 0.05 * covc)
                    scored.append((qual, int(yc)))

                if scored:
                    scored.sort(key=lambda t: t[0], reverse=True)
                    target_new = max(1, min(2, int(round(float(g) / max(1.0, med_gap))) - 1))
                    picked: list[int] = []
                    for _, yc in scored:
                        if all(abs(yc - yp) >= max(3, int(round(0.58 * min_gap))) for yp in picked):
                            picked.append(int(yc))
                        if len(picked) >= target_new:
                            break
                    insertions.extend(picked)
                else:
                    # Structural fallback: if the gap is abnormally large and no
                    # hard-line candidate survives, place candidates near expected
                    # split locations using weak evidence + color transitions.
                    large_gap = float(g) >= 1.55 * med_gap
                    if large_gap:
                        target_new = max(1, min(2, int(round(float(g) / max(1.0, med_gap))) - 1))
                        pmax = float(np.max(seg)) if seg.size else 0.0
                        pmax = max(pmax, 1e-6)
                        cmax = 0.0
                        if col_proj is not None:
                            cseg = col_proj[lo : hi + 1]
                            cmax = float(np.max(cseg)) if cseg.size else 0.0
                        cmax = max(cmax, 1e-6)

                        picked_fb: list[int] = []
                        for kk in range(target_new):
                            frac = float(kk + 1) / float(target_new + 1)
                            y_exp = int(round(float(ys[i]) + frac * float(g)))
                            win = max(4, int(round(0.24 * med_gap)))
                            lo2 = max(lo, y_exp - win)
                            hi2 = min(hi, y_exp + win)
                            if hi2 <= lo2:
                                continue
                            best_q = -1e18
                            best_y = None
                            for yc in range(lo2, hi2 + 1):
                                sc = _h_score(int(yc))
                                covh = self._flex_cov_horizontal(h_prob_crop, y=yc, band=band, thr=line_thr, x0=0, x1=w)
                                jhc = _j_hit(int(yc))
                                covc = _color_cov(int(yc))
                                pnorm = float(proj[yc]) / pmax if 0 <= yc < proj.shape[0] else 0.0
                                cnorm = float(col_proj[yc]) / cmax if (col_proj is not None and 0 <= yc < col_proj.shape[0]) else 0.0
                                # Favor structural consistency, but still require some
                                # evidence to avoid pure midpoint hallucinations.
                                if not (
                                    (covc >= max(0.52, float(self.cfg.color_line_min_cov) - 0.12) and jhc >= 0.10)
                                    or (sc >= max(0.20, float(self.cfg.weak_hline_prune_thresh) - 0.05) and jhc >= 0.12)
                                ):
                                    continue
                                q = float(0.42 * sc + 0.18 * covh + 0.10 * jhc + 0.20 * covc + 0.06 * pnorm + 0.04 * cnorm)
                                if q > best_q:
                                    best_q = q
                                    best_y = int(yc)
                            if best_y is not None:
                                if all(abs(best_y - yp) >= max(3, int(round(0.58 * min_gap))) for yp in picked_fb):
                                    picked_fb.append(int(best_y))
                        insertions.extend(picked_fb)
            if insertions:
                ys = _prune_near(sorted(set(ys + insertions)), max(2, int(round(0.58 * min_gap))))

        # 2b) Remove near-duplicate horizontal lines globally (not just top cluster).
        if len(ys) >= 3:
            for _ in range(12):
                gaps = np.diff(np.array(ys, dtype=np.int32))
                if gaps.size <= 0:
                    break
                med_gap = float(np.median(gaps))
                med_gap = max(float(min_gap), med_gap)
                tiny_thr = max(3, int(round(0.42 * med_gap)))
                bad = np.where(gaps < tiny_thr)[0]
                if bad.size <= 0:
                    break
                k = int(bad[np.argmin(gaps[bad])])
                # Prefer keeping outer borders; drop the weaker interior neighbor.
                if k == 0:
                    drop_idx = 1
                elif (k + 1) == (len(ys) - 1):
                    drop_idx = k
                else:
                    sa = _h_score(int(ys[k])) + 0.30 * _j_hit(int(ys[k]))
                    sb = _h_score(int(ys[k + 1])) + 0.30 * _j_hit(int(ys[k + 1]))
                    drop_idx = k if sa < sb else (k + 1)
                ys = ys[:drop_idx] + ys[drop_idx + 1 :]
                if len(ys) < 2:
                    break

        # 2c) Top-gap recovery: if the first row gap is unusually large, recover
        # a missing divider using line evidence (coverage + junction support).
        if len(ys) >= 3:
            gaps = np.diff(np.array(ys, dtype=np.int32))
            if gaps.size >= 2:
                med_gap = float(np.median(gaps))
                med_gap = max(float(min_gap), med_gap)
                g0 = float(ys[1] - ys[0])
                if g0 >= 1.42 * med_gap:
                    lo = ys[0] + min_gap
                    hi = ys[1] - min_gap
                    if hi > lo:
                        proj = h_prob_crop.mean(axis=1).astype(np.float32)
                        pseg = proj[lo : hi + 1]
                        pmax = float(np.max(pseg)) if pseg.size else 0.0
                        pmax = max(1e-6, pmax)
                        best_q = -1e18
                        best_y = None
                        for yc in range(lo, hi + 1):
                            covh = self._flex_cov_horizontal(h_prob_crop, y=yc, band=band, thr=line_thr, x0=0, x1=w)
                            jhc = _j_hit(int(yc))
                            sc = _h_score(int(yc))
                            pnorm = float(proj[yc]) / pmax if 0 <= yc < proj.shape[0] else 0.0
                            if not (
                                covh >= max(0.14, 0.58 * float(self.cfg.min_line_cov))
                                and (jhc >= 0.14 or sc >= max(0.20, float(self.cfg.weak_hline_prune_thresh) - 0.04))
                            ):
                                continue
                            q = float(0.50 * covh + 0.26 * jhc + 0.18 * sc + 0.06 * pnorm)
                            if q > best_q:
                                best_q = q
                                best_y = int(yc)
                        if best_y is not None and all(abs(best_y - v) >= max(3, int(round(0.58 * min_gap))) for v in ys):
                            ys = _prune_near(sorted(set(ys + [best_y])), max(2, int(round(0.58 * min_gap))))

        # 2d) Extrapolated top-border recovery:
        # if first line sits near the crop top, search one expected row-gap above
        # and pick the best nearby evidence line.
        if len(ys) >= 4:
            gaps = np.diff(np.array(ys, dtype=np.int32))
            if gaps.size >= 2:
                med_gap = float(np.median(gaps[: min(4, gaps.size)]))
                med_gap = max(float(min_gap), med_gap)
                if ys[0] <= int(round(0.22 * h)):
                    y_est = int(round(float(ys[0]) - med_gap))
                    win = max(4, int(round(0.36 * med_gap)))
                    lo = max(0, y_est - win)
                    hi = min(ys[0] - min_gap, y_est + win)
                    if hi > lo:
                        proj = h_prob_crop.mean(axis=1).astype(np.float32)
                        pseg = proj[lo : hi + 1]
                        pmax = float(np.max(pseg)) if pseg.size else 0.0
                        pmax = max(1e-6, pmax)
                        best_q = -1e18
                        best_y = None
                        for yc in range(lo, hi + 1):
                            covh = self._flex_cov_horizontal(
                                h_prob_crop,
                                y=yc,
                                band=band,
                                thr=max(0.20, float(line_thr) * 0.84),
                                x0=0,
                                x1=w,
                            )
                            jhc = _j_hit(int(yc))
                            sc = _h_score(int(yc))
                            covc = _color_cov(int(yc))
                            pnorm = float(proj[yc]) / pmax if 0 <= yc < proj.shape[0] else 0.0
                            if not (covh >= 0.10 and (jhc >= 0.10 or covc >= 0.52 or sc >= 0.20)):
                                continue
                            q = float(0.44 * covh + 0.24 * jhc + 0.16 * covc + 0.10 * sc + 0.06 * pnorm)
                            if q > best_q:
                                best_q = q
                                best_y = int(yc)
                        if best_y is not None and best_q >= 0.18:
                            if all(abs(best_y - v) >= max(3, int(round(0.58 * min_gap))) for v in ys):
                                ys = _prune_near(sorted(set(ys + [best_y])), max(2, int(round(0.58 * min_gap))))

        if bool(self.cfg.top_hline_cleanup_enable):
            # 3) Top-cluster cleanup: if two short top gaps are present, drop the
            # weaker of the two interior lines to suppress text-driven over-splits.
            if len(ys) >= 5:
                gaps = np.diff(np.array(ys, dtype=np.int32))
                med_gap = float(np.median(gaps)) if gaps.size else float(min_gap)
                med_gap = max(float(min_gap), med_gap)
                short_thr = float(self.cfg.top_cluster_short_gap_ratio) * med_gap
                g0 = float(ys[1] - ys[0])
                g1 = float(ys[2] - ys[1])
                top_guard = max(min_gap, int(round(float(self.cfg.top_hline_guard_ratio) * h)))
                if g0 <= short_thr and g1 <= short_thr and ys[2] <= ys[0] + 2 * top_guard:
                    s1 = _h_score(int(ys[1]))
                    s2 = _h_score(int(ys[2]))
                    if s1 <= s2:
                        ys = [ys[0]] + ys[2:]
                    else:
                        ys = [ys[0], ys[1]] + ys[3:]

            # 4) Final top duplicate guard.
            if len(ys) >= 3:
                gaps = np.diff(np.array(ys, dtype=np.int32))
                med_gap = float(np.median(gaps)) if gaps.size else float(min_gap)
                med_gap = max(float(min_gap), med_gap)
                if float(ys[1] - ys[0]) < 0.58 * med_gap:
                    ys = [ys[0]] + ys[2:]

            # 5) Top-cluster pattern fix:
            # short, short, then large gap -> drop weaker of the two top interior lines.
            if len(ys) >= 5:
                gaps = np.diff(np.array(ys, dtype=np.int32))
                med_gap = float(np.median(gaps)) if gaps.size else float(min_gap)
                med_gap = max(float(min_gap), med_gap)
                g0 = float(ys[1] - ys[0])
                g1 = float(ys[2] - ys[1])
                g2 = float(ys[3] - ys[2])
                if g0 < 0.78 * med_gap and g1 < 0.78 * med_gap and g2 > 1.20 * med_gap:
                    s1 = _h_score(int(ys[1])) + 0.35 * _j_hit(int(ys[1]))
                    s2 = _h_score(int(ys[2])) + 0.35 * _j_hit(int(ys[2]))
                    if s1 <= s2:
                        ys = [ys[0]] + ys[2:]
                    else:
                        ys = [ys[0], ys[1]] + ys[3:]

        if len(ys) < 2:
            return y_lines
        return ys

    def _vertical_line_row_evidence(
        self,
        x: int,
        y_lines: list[int],
        v_prob_crop: np.ndarray,
        j_prob_crop: np.ndarray,
        line_thr: float,
        band: int,
        v_line_crop: Optional[np.ndarray] = None,
    ) -> tuple[float, float, float]:
        rows = max(0, len(y_lines) - 1)
        if rows <= 0:
            return 0.0, 0.0, 0.0
        h, w = v_prob_crop.shape[:2]
        if h <= 1 or w <= 1:
            return 0.0, 0.0, 0.0

        x = int(np.clip(int(x), 0, w - 1))
        # Use a relatively narrow spatial band for line-position scoring so wide
        # dark/text regions do not create broad flat maxima.
        pos_band = max(2, min(int(band), 4))
        thr = max(0.20, float(line_thr) * 0.90)
        row_heights = np.diff(np.array(y_lines, dtype=np.int32)).astype(np.float32)
        med_h = float(np.median(row_heights)) if row_heights.size else 1.0
        med_h = max(1.0, med_h)
        has_line_map = v_line_crop is not None and v_line_crop.shape[:2] == v_prob_crop.shape[:2]

        covs: list[float] = []
        runs: list[float] = []
        lcovs: list[float] = []
        lruns: list[float] = []
        supports = 0
        for r in range(rows):
            y0 = int(y_lines[r])
            y1 = int(y_lines[r + 1])
            if y1 <= y0:
                continue
            cov = self._flex_cov_vertical(v_prob_crop, x=x, band=pos_band, thr=thr, y0=y0, y1=y1)
            xx0 = max(0, int(x) - pos_band)
            xx1 = min(w - 1, int(x) + pos_band)
            seg = (np.max(v_prob_crop[y0:y1, xx0 : xx1 + 1], axis=1) >= thr) if (xx1 >= xx0 and y1 > y0) else np.zeros((0,), dtype=np.uint8)
            run = _max_true_run_ratio(seg)
            lcov = 0.0
            lrun = 0.0
            if has_line_map:
                lthr = 0.35
                lband = max(1, pos_band - 1)
                lcov = self._flex_cov_vertical(v_line_crop, x=x, band=lband, thr=lthr, y0=y0, y1=y1)
                lxx0 = max(0, int(x) - lband)
                lxx1 = min(w - 1, int(x) + lband)
                lseg = (
                    np.max(v_line_crop[y0:y1, lxx0 : lxx1 + 1], axis=1) >= lthr
                ) if (lxx1 >= lxx0 and y1 > y0) else np.zeros((0,), dtype=np.uint8)
                lrun = _max_true_run_ratio(lseg)

            row_h = float(y1 - y0)
            min_cov = max(0.12, 0.55 * float(self.cfg.min_line_cov))
            min_run = max(0.40, float(self.cfg.presence_min_run_ratio) - 0.06)
            if row_h < 0.78 * med_h:
                min_run = max(min_run, 0.60)
            if r == (rows - 1):
                min_cov = max(min_cov, 0.36)
                min_run = max(min_run, 0.84)
            line_ok = has_line_map and (lcov >= 0.24 and lrun >= (0.86 if r == (rows - 1) else 0.62))
            if (cov >= min_cov and run >= min_run) or line_ok:
                supports += 1
            covs.append(float(cov))
            runs.append(float(run))
            lcovs.append(float(lcov))
            lruns.append(float(lrun))

        jvals: list[float] = []
        for y in y_lines:
            y0 = max(0, int(y) - pos_band)
            y1 = min(j_prob_crop.shape[0] - 1, int(y) + pos_band)
            x0 = max(0, int(x) - pos_band)
            x1 = min(j_prob_crop.shape[1] - 1, int(x) + pos_band)
            pj = j_prob_crop[y0 : y1 + 1, x0 : x1 + 1]
            if pj.size:
                jvals.append(float(np.max(pj)))

        sup_frac = float(supports) / float(max(1, rows))
        cov_mean = float(np.mean(covs)) if covs else 0.0
        run_mean = float(np.mean(runs)) if runs else 0.0
        lcov_mean = float(np.mean(lcovs)) if lcovs else 0.0
        lrun_mean = float(np.mean(lruns)) if lruns else 0.0
        j_mean = float(np.mean(jvals)) if jvals else 0.0
        score = float(
            0.42 * sup_frac
            + 0.16 * cov_mean
            + 0.14 * run_mean
            + 0.08 * j_mean
            + 0.10 * lcov_mean
            + 0.10 * lrun_mean
        )
        cov_blend = float(max(cov_mean, 0.82 * lcov_mean))
        return score, sup_frac, cov_blend

    def _refine_vertical_lines_local(
        self,
        x_lines: list[int],
        y_lines: list[int],
        v_prob_crop: np.ndarray,
        j_prob_crop: np.ndarray,
        line_thr: float,
        v_line_crop: Optional[np.ndarray] = None,
    ) -> list[int]:
        if len(x_lines) < 2 or len(y_lines) < 2:
            return x_lines
        xs = sorted(set(int(v) for v in x_lines))
        if len(xs) < 2:
            return x_lines

        band = max(2, int(self.cfg.flexible_band_px))
        min_gap = max(6, int(self.cfg.min_gap_px))

        # 1) Snap interior lines to the best nearby row-wise evidence.
        for i in range(1, len(xs) - 1):
            x = int(xs[i])
            gl = int(xs[i] - xs[i - 1])
            gr = int(xs[i + 1] - xs[i])
            win = max(5, int(round(0.35 * float(max(gl, gr)))))
            lo = max(int(xs[i - 1]) + 2, x - win)
            hi = min(int(xs[i + 1]) - 2, x + win)
            if hi < lo:
                continue
            best_x = x
            best_s = -1e18
            for p in range(lo, hi + 1):
                s, _, _ = self._vertical_line_row_evidence(
                    x=p,
                    y_lines=y_lines,
                    v_prob_crop=v_prob_crop,
                    j_prob_crop=j_prob_crop,
                    line_thr=line_thr,
                    band=band,
                    v_line_crop=v_line_crop,
                )
                s_adj = float(s - 0.0018 * abs(p - x))
                if s_adj > best_s:
                    best_s = s_adj
                    best_x = int(p)
            xs[i] = int(best_x)

        xs = _prune_near(sorted(set(xs)), max(2, int(round(0.58 * min_gap))))
        if len(xs) < 2:
            return x_lines

        # 2) Prune clearly weak interior lines.
        if len(xs) > 2:
            ev = []
            for i in range(1, len(xs) - 1):
                s, sf, cv = self._vertical_line_row_evidence(
                    x=xs[i],
                    y_lines=y_lines,
                    v_prob_crop=v_prob_crop,
                    j_prob_crop=j_prob_crop,
                    line_thr=line_thr,
                    band=band,
                    v_line_crop=v_line_crop,
                )
                ev.append((float(s), float(sf), float(cv), i))
            if ev:
                med_s = float(np.median([v[0] for v in ev]))
                drop_idx: list[int] = []
                for s, sf, cv, i in ev:
                    if sf < 0.30 and s < max(0.20, 0.64 * med_s) and cv < 0.22:
                        drop_idx.append(int(i))
                for i in sorted(drop_idx, reverse=True):
                    if 0 < i < (len(xs) - 1):
                        xs.pop(i)

        xs = _prune_near(sorted(set(xs)), max(2, int(round(0.58 * min_gap))))
        if len(xs) < 2:
            return x_lines

        # 3) Fill missing interior lines in large gaps using row-wise evidence.
        if len(xs) >= 3:
            gaps = np.diff(np.array(xs, dtype=np.int32))
            med_gap = float(np.median(gaps)) if gaps.size else float(min_gap)
            med_gap = max(float(min_gap), med_gap)
            add: list[int] = []
            for i, g in enumerate(gaps):
                gg = float(g)
                if gg < 1.55 * med_gap:
                    continue
                # Do not over-split the first/last (label/footer) spans.
                edge_gap = (i == 0 or i == len(gaps) - 1)
                target_new = 1 if edge_gap else max(1, min(2, int(round(gg / med_gap)) - 1))
                for kk in range(target_new):
                    if edge_gap and i == 0:
                        x_exp = int(round(float(xs[i + 1]) - med_gap))
                    elif edge_gap and i == (len(gaps) - 1):
                        x_exp = int(round(float(xs[i]) + med_gap))
                    else:
                        frac = float(kk + 1) / float(target_new + 1)
                        x_exp = int(round(float(xs[i]) + frac * gg))
                    win = max(5, int(round(0.30 * med_gap)))
                    lo = max(int(xs[i]) + min_gap, x_exp - win)
                    hi = min(int(xs[i + 1]) - min_gap, x_exp + win)
                    if hi <= lo:
                        continue
                    best_x = None
                    best_s = -1e18
                    for p in range(lo, hi + 1):
                        s, sf, _ = self._vertical_line_row_evidence(
                            x=p,
                            y_lines=y_lines,
                            v_prob_crop=v_prob_crop,
                            j_prob_crop=j_prob_crop,
                            line_thr=line_thr,
                            band=band,
                            v_line_crop=v_line_crop,
                        )
                        if sf < 0.32:
                            continue
                        s_adj = float(s - 0.0013 * abs(p - x_exp))
                        if s_adj > best_s:
                            best_s = s_adj
                            best_x = int(p)
                        if best_x is not None and best_s >= 0.24:
                            if all(abs(best_x - v) >= max(3, int(round(0.58 * min_gap))) for v in (xs + add)):
                                add.append(int(best_x))
            if add:
                xs = _prune_near(sorted(set(xs + add)), max(2, int(round(0.58 * min_gap))))

        # 4) Remove tiny adjacent vertical gaps by keeping the stronger line.
        if len(xs) >= 4:
            for _ in range(12):
                gaps = np.diff(np.array(xs, dtype=np.int32))
                if gaps.size <= 0:
                    break
                med_gap = float(np.median(gaps))
                med_gap = max(float(min_gap), med_gap)
                tiny_thr = max(4, int(round(0.45 * med_gap)))
                bad = np.where(gaps < tiny_thr)[0]
                if bad.size <= 0:
                    break
                k = int(bad[np.argmin(gaps[bad])])
                if k == 0:
                    drop_idx = 1
                elif (k + 1) == (len(xs) - 1):
                    drop_idx = k
                else:
                    sa, _, _ = self._vertical_line_row_evidence(
                        x=xs[k],
                        y_lines=y_lines,
                        v_prob_crop=v_prob_crop,
                        j_prob_crop=j_prob_crop,
                        line_thr=line_thr,
                        band=band,
                        v_line_crop=v_line_crop,
                    )
                    sb, _, _ = self._vertical_line_row_evidence(
                        x=xs[k + 1],
                        y_lines=y_lines,
                        v_prob_crop=v_prob_crop,
                        j_prob_crop=j_prob_crop,
                        line_thr=line_thr,
                        band=band,
                        v_line_crop=v_line_crop,
                    )
                    drop_idx = k if sa < sb else (k + 1)
                if 0 < drop_idx < (len(xs) - 1):
                    xs = xs[:drop_idx] + xs[drop_idx + 1 :]
                if len(xs) < 2:
                    break

        if len(xs) < 2:
            return x_lines
        return xs

    def _snap_vertical_lines_to_line_evidence(
        self,
        x_lines: list[int],
        y_lines: list[int],
        v_prob_crop: np.ndarray,
        j_prob_crop: np.ndarray,
        line_thr: float,
        v_line_crop: Optional[np.ndarray] = None,
    ) -> list[int]:
        if v_line_crop is None or v_line_crop.shape[:2] != v_prob_crop.shape[:2]:
            return x_lines
        if len(x_lines) < 2 or len(y_lines) < 2:
            return x_lines
        h, w = v_prob_crop.shape[:2]
        if h < 4 or w < 4:
            return x_lines

        xs = sorted(set(int(v) for v in x_lines))
        if len(xs) < 2:
            return x_lines
        band = max(2, int(self.cfg.flexible_band_px))
        min_gap = max(6, int(self.cfg.min_gap_px))
        win = max(2, int(round(0.006 * w)))
        sband = max(1, int(round(0.45 * float(band))))

        for i in range(1, len(xs) - 1):
            x = int(xs[i])
            lo = max(int(xs[i - 1]) + 2, x - win)
            hi = min(int(xs[i + 1]) - 2, x + win)
            if hi < lo:
                continue
            best_x = x
            best_s = -1e18
            for p in range(lo, hi + 1):
                row_s, sup_frac, _ = self._vertical_line_row_evidence(
                    x=p,
                    y_lines=y_lines,
                    v_prob_crop=v_prob_crop,
                    j_prob_crop=j_prob_crop,
                    line_thr=line_thr,
                    band=band,
                    v_line_crop=v_line_crop,
                )
                cov = self._flex_cov_vertical(v_prob_crop, x=p, band=sband, thr=max(0.20, float(line_thr) * 0.90), y0=0, y1=h)
                lcov = self._flex_cov_vertical(v_line_crop, x=p, band=sband, thr=0.35, y0=0, y1=h)
                x0 = max(0, p - sband)
                x1 = min(w - 1, p + sband)
                lseg = (np.max(v_line_crop[:, x0 : x1 + 1], axis=1) >= 0.35) if x1 >= x0 else np.zeros((0,), dtype=np.uint8)
                lrun = _max_true_run_ratio(lseg)
                s = float(
                    0.40 * row_s
                    + 0.18 * sup_frac
                    + 0.18 * cov
                    + 0.16 * lcov
                    + 0.08 * lrun
                    - 0.0010 * abs(p - x)
                )
                if s > best_s:
                    best_s = s
                    best_x = int(p)
            xs[i] = int(best_x)

        xs = _prune_near(sorted(set(xs)), max(2, int(round(0.58 * min_gap))))
        if len(xs) < 2:
            return x_lines
        return xs

    def _resnap_vertical_lines_by_segments(
        self,
        x_lines: list[int],
        y_lines: list[int],
        v_prob_crop: np.ndarray,
        line_thr: float,
        v_line_crop: Optional[np.ndarray] = None,
    ) -> list[int]:
        # Final position cleanup: maximize true line evidence per row-segment.
        # This reduces persistent x-offset drift that later causes missed/false
        # vertical segment decisions in merged/footer regions.
        if v_line_crop is None or v_line_crop.shape[:2] != v_prob_crop.shape[:2]:
            return x_lines
        if len(x_lines) < 3 or len(y_lines) < 2:
            return x_lines

        h, w = v_prob_crop.shape[:2]
        rows = max(0, len(y_lines) - 1)
        if h < 4 or w < 4 or rows <= 0:
            return x_lines

        xs = sorted(set(int(v) for v in x_lines))
        if len(xs) < 3:
            return x_lines

        min_gap = max(6, int(self.cfg.min_gap_px))
        band = max(2, int(self.cfg.flexible_band_px))
        sband = max(1, int(round(0.45 * float(band))))
        thr = max(0.20, float(line_thr) * 0.90)
        row_heights = np.diff(np.array(y_lines, dtype=np.int32)).astype(np.float32)
        med_h = float(np.median(row_heights)) if row_heights.size else 1.0
        med_h = max(1.0, med_h)

        for i in range(1, len(xs) - 1):
            x0 = int(xs[i])
            gl = int(xs[i] - xs[i - 1])
            gr = int(xs[i + 1] - xs[i])
            win = max(4, int(round(0.20 * float(max(gl, gr)))))
            lo = max(int(xs[i - 1]) + 2, x0 - win)
            hi = min(int(xs[i + 1]) - 2, x0 + win)
            if hi <= lo:
                continue

            picks: list[tuple[int, float]] = []
            for r in range(rows):
                y0 = int(y_lines[r])
                y1 = int(y_lines[r + 1])
                if y1 <= y0:
                    continue
                row_h = float(y1 - y0)
                best_p = x0
                best_s = -1e18
                best_lcov = 0.0
                best_lrun = 0.0
                for p in range(lo, hi + 1):
                    cov = self._flex_cov_vertical(v_prob_crop, x=p, band=sband, thr=thr, y0=y0, y1=y1)
                    xw0 = max(0, p - sband)
                    xw1 = min(w - 1, p + sband)
                    seg = (
                        np.max(v_prob_crop[y0:y1, xw0 : xw1 + 1], axis=1) >= thr
                    ) if (xw1 >= xw0 and y1 > y0) else np.zeros((0,), dtype=np.uint8)
                    run = _max_true_run_ratio(seg)

                    lcov = self._flex_cov_vertical(v_line_crop, x=p, band=sband, thr=0.35, y0=y0, y1=y1)
                    lseg = (
                        np.max(v_line_crop[y0:y1, xw0 : xw1 + 1], axis=1) >= 0.35
                    ) if (xw1 >= xw0 and y1 > y0) else np.zeros((0,), dtype=np.uint8)
                    lrun = _max_true_run_ratio(lseg)

                    s = float(
                        0.22 * cov
                        + 0.10 * run
                        + 0.40 * lcov
                        + 0.28 * lrun
                        - 0.0011 * abs(p - x0)
                    )
                    if s > best_s:
                        best_s = s
                        best_p = int(p)
                        best_lcov = float(lcov)
                        best_lrun = float(lrun)

                min_row_s = 0.22 if row_h >= 0.78 * med_h else 0.25
                if best_s >= min_row_s and (best_lcov >= 0.20 or best_lrun >= 0.48):
                    wt = float(max(0.0, best_s - 0.12))
                    wt *= float(np.clip(row_h / med_h, 0.70, 1.35))
                    if wt > 0.0:
                        picks.append((int(best_p), wt))

            need = max(3, int(round(0.34 * float(rows))))
            if len(picks) < need:
                continue
            pw = np.array([p[0] for p in picks], dtype=np.float32)
            ww = np.array([max(1e-4, p[1]) for p in picks], dtype=np.float32)
            new_x = int(round(float(np.sum(pw * ww) / np.sum(ww))))
            move_cap = max(2, int(round(0.36 * float(max(gl, gr)))))
            if abs(new_x - x0) > move_cap:
                new_x = int(np.clip(new_x, x0 - move_cap, x0 + move_cap))
            xs[i] = int(np.clip(new_x, xs[i - 1] + 1, xs[i + 1] - 1))

        xs = _prune_near(sorted(set(xs)), max(2, int(round(0.58 * min_gap))))
        if len(xs) < 2:
            return x_lines
        return xs

    def _decode_best_lines(
        self,
        v_prob: np.ndarray,
        h_prob: np.ndarray,
        j_prob: np.ndarray,
        bbox_xyxy: tuple[int, int, int, int],
        h_color_prob: Optional[np.ndarray] = None,
        v_line_prob: Optional[np.ndarray] = None,
    ) -> tuple[list[int], list[int], float, float]:
        x0, y0, x1, y1 = bbox_xyxy
        h, w = v_prob.shape[:2]
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 <= x0:
            x0, x1 = 0, max(1, w - 1)
        if y1 <= y0:
            y0, y1 = 0, max(1, h - 1)

        # Expand around coarse table box so weak outer/header lines can still be
        # recovered by evidence-based decoding.
        bw = int(x1 - x0 + 1)
        bh = int(y1 - y0 + 1)
        pad_l = max(2, int(round(0.006 * bw)))
        pad_r = max(2, int(round(0.006 * bw)))
        pad_t = max(8, int(round(0.12 * bh)))
        pad_b = max(3, int(round(0.02 * bh)))
        x0e = int(max(0, x0 - pad_l))
        x1e = int(min(w - 1, x1 + pad_r))
        y0e = int(max(0, y0 - pad_t))
        y1e = int(min(h - 1, y1 + pad_b))

        v_crop = v_prob[y0e : y1e + 1, x0e : x1e + 1]
        h_crop = h_prob[y0e : y1e + 1, x0e : x1e + 1]
        j_crop = j_prob[y0e : y1e + 1, x0e : x1e + 1]
        h_color_crop = None
        if h_color_prob is not None and h_color_prob.shape[:2] == h_prob.shape[:2]:
            h_color_crop = h_color_prob[y0e : y1e + 1, x0e : x1e + 1]
        v_line_crop = None
        if v_line_prob is not None and v_line_prob.shape[:2] == v_prob.shape[:2]:
            v_line_crop = v_line_prob[y0e : y1e + 1, x0e : x1e + 1]

        base = (float(self.cfg.line_thresh), float(self.cfg.min_line_cov), float(self.cfg.peak_rel_thresh), int(self.cfg.min_gap_px))
        params = [base]
        if self.cfg.auto_tune_decode:
            params.extend(
                [
                    (max(0.26, base[0] - 0.10), max(0.11, base[1] - 0.05), max(0.07, base[2] - 0.05), max(6, int(round(base[3] * 0.86)))),
                    (max(0.30, base[0] - 0.07), max(0.12, base[1] - 0.03), max(0.08, base[2] - 0.03), max(6, int(round(base[3] * 0.88)))),
                    (max(0.30, base[0] - 0.03), max(0.12, base[1] - 0.02), max(0.08, base[2] - 0.02), max(6, int(round(base[3] * 0.95)))),
                    (min(0.82, base[0] + 0.08), min(0.60, base[1] + 0.10), min(0.45, base[2] + 0.08), int(round(base[3] * 1.20))),
                ]
            )

        best_s = -1e18
        best_x = [0, max(1, v_crop.shape[1] - 1)]
        best_y = [0, max(1, v_crop.shape[0] - 1)]
        best_thr = base[0]

        for line_thr, min_cov, peak_thr, min_gap in params:
            x_local = self._decode_axis_lines(
                v_crop,
                j_crop,
                axis="x",
                line_thr=line_thr,
                min_cov=min_cov,
                peak_thr=peak_thr,
                min_gap=min_gap,
                min_lines=max(2, self.cfg.min_cols + 1),
                max_lines=max(2, self.cfg.max_cols + 1),
                band=max(2, int(self.cfg.flexible_band_px)),
                soft_prob_crop=v_line_crop,
            )
            y_local = self._decode_axis_lines(
                h_crop,
                j_crop,
                axis="y",
                line_thr=line_thr,
                min_cov=min_cov,
                peak_thr=peak_thr,
                min_gap=min_gap,
                min_lines=max(2, self.cfg.min_rows + 1),
                max_lines=max(2, self.cfg.max_rows + 1),
                band=max(2, int(self.cfg.flexible_band_px)),
                soft_prob_crop=h_color_crop,
            )
            # Prune weak interior horizontal lines only when line count is high.
            if len(y_local) >= max(8, self.cfg.min_rows + 4):
                keep_y = [y_local[0]]
                thr_keep_y = max(0.12, 0.55 * float(min_cov))
                band_e = max(2, int(self.cfg.flexible_band_px))
                for yv in y_local[1:-1]:
                    ch = self._flex_cov_horizontal(h_crop, y=int(yv), band=band_e, thr=float(line_thr), x0=0, x1=h_crop.shape[1])
                    if ch >= thr_keep_y:
                        keep_y.append(int(yv))
                keep_y.append(y_local[-1])
                y_local = _prune_near(sorted(set(keep_y)), max(2, int(round(0.58 * max(1, min_gap)))))
            s = self._score_grid(
                x_local,
                y_local,
                v_prob_crop=v_crop,
                h_prob_crop=h_crop,
                j_prob_crop=j_crop,
                line_thr=line_thr,
                band=max(2, int(self.cfg.flexible_band_px)),
            )
            if s > best_s:
                best_s = s
                best_x = x_local
                best_y = y_local
                best_thr = line_thr

        best_x, best_y = self._snap_outer_lines(
            best_x,
            best_y,
            v_prob_crop=v_crop,
            h_prob_crop=h_crop,
            line_thr=best_thr,
            v_line_prob_crop=v_line_crop,
        )
        best_y = self._refine_horizontal_lines_local(
            x_lines=best_x,
            y_lines=best_y,
            v_prob_crop=v_crop,
            h_prob_crop=h_crop,
            j_prob_crop=j_crop,
            h_color_crop=h_color_crop,
            line_thr=best_thr,
        )
        best_x = self._refine_vertical_lines_local(
            x_lines=best_x,
            y_lines=best_y,
            v_prob_crop=v_crop,
            j_prob_crop=j_crop,
            line_thr=best_thr,
            v_line_crop=v_line_crop,
        )
        best_x = self._snap_vertical_lines_to_line_evidence(
            x_lines=best_x,
            y_lines=best_y,
            v_prob_crop=v_crop,
            j_prob_crop=j_crop,
            line_thr=best_thr,
            v_line_crop=v_line_crop,
        )
        best_x, best_y = self._snap_all_lines_local(
            best_x,
            best_y,
            v_prob_crop=v_crop,
            h_prob_crop=h_crop,
            j_prob_crop=j_crop,
            line_thr=best_thr,
            v_line_prob_crop=v_line_crop,
        )
        best_x = self._refine_vertical_lines_local(
            x_lines=best_x,
            y_lines=best_y,
            v_prob_crop=v_crop,
            j_prob_crop=j_crop,
            line_thr=best_thr,
            v_line_crop=v_line_crop,
        )
        best_x = self._snap_vertical_lines_to_line_evidence(
            x_lines=best_x,
            y_lines=best_y,
            v_prob_crop=v_crop,
            j_prob_crop=j_crop,
            line_thr=best_thr,
            v_line_crop=v_line_crop,
        )
        best_x = self._resnap_vertical_lines_by_segments(
            x_lines=best_x,
            y_lines=best_y,
            v_prob_crop=v_crop,
            line_thr=best_thr,
            v_line_crop=v_line_crop,
        )

        x_abs = [x0e + int(v) for v in best_x]
        y_abs = [y0e + int(v) for v in best_y]
        if not x_abs:
            x_abs = [x0e, x1e]
        if not y_abs:
            y_abs = [y0e, y1e]

        x_abs = _prune_near(sorted(set(x_abs)), max(2, int(round(0.60 * max(1, self.cfg.min_gap_px)))))
        y_abs = _prune_near(sorted(set(y_abs)), max(2, int(round(0.60 * max(1, self.cfg.min_gap_px)))))
        if len(x_abs) < 2:
            x_abs = [x0, x1]
        if len(y_abs) < 2:
            y_abs = [y0, y1]
        return x_abs, y_abs, best_thr, best_s

    def _vertical_presence(self, v_prob: np.ndarray, x_lines: list[int], y_lines: list[int], line_thr: float) -> np.ndarray:
        return self._vertical_presence_with_color(
            v_prob,
            x_lines,
            y_lines,
            line_thr=line_thr,
            v_color_prob=None,
            v_line_prob=None,
        )

    def _vertical_presence_with_color(
        self,
        v_prob: np.ndarray,
        x_lines: list[int],
        y_lines: list[int],
        line_thr: float,
        v_color_prob: Optional[np.ndarray] = None,
        v_line_prob: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        rows = max(0, len(y_lines) - 1)
        nvl = len(x_lines)
        out = np.zeros((rows, nvl), dtype=np.uint8)
        band = max(2, int(self.cfg.flexible_band_px))
        thr = max(0.20, float(line_thr) * 0.92)
        row_heights = np.diff(np.array(y_lines, dtype=np.int32)).astype(np.float32)
        med_h = float(np.median(row_heights)) if row_heights.size else 1.0
        med_h = max(1.0, med_h)
        row_cov = np.zeros((rows, nvl), dtype=np.float32)
        row_run = np.zeros((rows, nvl), dtype=np.float32)
        row_ccov = np.zeros((rows, nvl), dtype=np.float32)
        row_crun = np.zeros((rows, nvl), dtype=np.float32)
        row_lcov = np.zeros((rows, nvl), dtype=np.float32)
        row_lrun = np.zeros((rows, nvl), dtype=np.float32)
        cband = max(1, int(round(0.45 * float(band))))
        cthr = max(0.20, float(self.cfg.h_color_peak_min) - 0.06)
        for r in range(rows):
            yy0, yy1 = int(y_lines[r]), int(y_lines[r + 1])
            if yy1 <= yy0:
                continue
            row_h = float(yy1 - yy0)
            min_cov = max(float(self.cfg.min_line_cov), float(self.cfg.merge_presence_min_cov))
            min_run = float(self.cfg.presence_min_run_ratio)
            if row_h < 0.78 * med_h:
                min_run = max(min_run, 0.62)
            if r == (rows - 1):
                min_cov = max(min_cov, 0.34)
                min_run = max(min_run, 0.78)
            for i, x in enumerate(x_lines):
                search_dx = max(1, int(round(0.22 * float(band))))
                lx = max(0, int(x) - search_dx)
                rx = min(v_prob.shape[1] - 1, int(x) + search_dx)

                best_cov = 0.0
                best_run = 0.0
                best_s = -1e18
                for px in range(lx, rx + 1):
                    covp = self._flex_cov_vertical(v_prob, x=px, band=band, thr=thr, y0=yy0, y1=yy1)
                    xx0 = max(0, int(px) - band)
                    xx1 = min(v_prob.shape[1] - 1, int(px) + band)
                    seg = (np.max(v_prob[yy0:yy1, xx0 : xx1 + 1], axis=1) >= thr) if (xx1 >= xx0 and yy1 > yy0) else np.zeros((0,), dtype=np.uint8)
                    runp = _max_true_run_ratio(seg)
                    sp = float(0.68 * covp + 0.32 * runp - 0.0015 * abs(px - int(x)))
                    if sp > best_s:
                        best_s = sp
                        best_cov = float(covp)
                        best_run = float(runp)
                cov = float(best_cov)
                run_ratio = float(best_run)
                row_cov[r, i] = float(cov)
                row_run[r, i] = float(run_ratio)
                ccov = 0.0
                crun = 0.0
                lcov = 0.0
                lrun = 0.0
                if v_color_prob is not None and v_color_prob.shape[:2] == v_prob.shape[:2]:
                    cbest_cov = 0.0
                    cbest_run = 0.0
                    cbest_s = -1e18
                    for px in range(lx, rx + 1):
                        covc = self._flex_cov_vertical(v_color_prob, x=px, band=cband, thr=cthr, y0=yy0, y1=yy1)
                        cxx0 = max(0, int(px) - cband)
                        cxx1 = min(v_color_prob.shape[1] - 1, int(px) + cband)
                        cseg = (
                            np.max(v_color_prob[yy0:yy1, cxx0 : cxx1 + 1], axis=1) >= cthr
                        ) if (cxx1 >= cxx0 and yy1 > yy0) else np.zeros((0,), dtype=np.uint8)
                        runc = _max_true_run_ratio(cseg)
                        sc = float(0.66 * covc + 0.34 * runc - 0.0012 * abs(px - int(x)))
                        if sc > cbest_s:
                            cbest_s = sc
                            cbest_cov = float(covc)
                            cbest_run = float(runc)
                    ccov = float(cbest_cov)
                    crun = float(cbest_run)
                if v_line_prob is not None and v_line_prob.shape[:2] == v_prob.shape[:2]:
                    lbest_cov = 0.0
                    lbest_run = 0.0
                    lbest_s = -1e18
                    for px in range(lx, rx + 1):
                        covl = self._flex_cov_vertical(v_line_prob, x=px, band=cband, thr=0.35, y0=yy0, y1=yy1)
                        lxx0 = max(0, int(px) - cband)
                        lxx1 = min(v_line_prob.shape[1] - 1, int(px) + cband)
                        lseg = (
                            np.max(v_line_prob[yy0:yy1, lxx0 : lxx1 + 1], axis=1) >= 0.35
                        ) if (lxx1 >= lxx0 and yy1 > yy0) else np.zeros((0,), dtype=np.uint8)
                        runl = _max_true_run_ratio(lseg)
                        sl = float(0.68 * covl + 0.32 * runl - 0.0012 * abs(px - int(x)))
                        if sl > lbest_s:
                            lbest_s = sl
                            lbest_cov = float(covl)
                            lbest_run = float(runl)
                    lcov = float(lbest_cov)
                    lrun = float(lbest_run)
                row_lcov[r, i] = float(lcov)
                row_lrun[r, i] = float(lrun)
                row_ccov[r, i] = float(ccov)
                row_crun[r, i] = float(crun)
                if cov >= min_cov and run_ratio >= min_run:
                    out[r, i] = 1

        # Segment-level image evidence veto:
        # keep a vertical segment only if there is actual local line evidence
        # under that segment (prefer dedicated line map; fallback to color map).
        if rows >= 1 and nvl >= 3:
            has_line_map = v_line_prob is not None and v_line_prob.shape[:2] == v_prob.shape[:2]
            has_color_map = v_color_prob is not None and v_color_prob.shape[:2] == v_prob.shape[:2]
            if has_line_map or has_color_map:
                for r in range(rows):
                    row_h = float(y_lines[r + 1] - y_lines[r]) if (r + 1) < len(y_lines) else 0.0
                    short_row = row_h < 0.78 * med_h
                    for i in range(1, nvl - 1):
                        # Rescue: if model-line evidence is strong at this
                        # boundary segment, keep it even when the raw channel
                        # is slightly off-position.
                        if out[r, i] == 0 and has_line_map:
                            add_cov_thr = 0.52 if r == (rows - 1) else (0.34 if short_row else 0.28)
                            add_run_thr = 0.90 if r == (rows - 1) else (0.68 if short_row else 0.58)
                            if float(row_lcov[r, i]) >= add_cov_thr and float(row_lrun[r, i]) >= add_run_thr:
                                out[r, i] = 1

                        if out[r, i] == 0:
                            continue
                        # Prefer the true line map; fallback to color transition map.
                        ecov = float(row_lcov[r, i]) if has_line_map else float(row_ccov[r, i])
                        erun = float(row_lrun[r, i]) if has_line_map else float(row_crun[r, i])

                        if r == (rows - 1):
                            cov_thr = 0.34 if has_line_map else 0.30
                            run_thr = 0.82 if has_line_map else 0.72
                        elif short_row:
                            cov_thr = 0.20 if has_line_map else 0.20
                            run_thr = 0.58 if has_line_map else 0.44
                        else:
                            cov_thr = 0.12 if has_line_map else 0.14
                            run_thr = 0.34 if has_line_map else 0.30

                        if has_line_map and (float(row_lcov[r, i]) < 0.10 and float(row_lrun[r, i]) < 0.20):
                            out[r, i] = 0
                            continue

                        # Core rule requested: if there isn't a line under this
                        # segment, remove the segment.
                        if r == (rows - 1):
                            # Footer row: require strong local evidence so text
                            # strokes do not survive as fake dividers.
                            if ecov < cov_thr or erun < run_thr:
                                out[r, i] = 0
                        elif ecov < cov_thr and erun < run_thr:
                            out[r, i] = 0

                # Footer continuity sanity check:
                # a footer interior segment absent above must have strong local
                # evidence in footer to survive.
                if rows >= 2:
                    r = rows - 1
                    for i in range(1, nvl - 1):
                        if out[r, i] == 0:
                            continue
                        if out[r - 1, i] != 0:
                            continue
                        ecov = float(row_lcov[r, i]) if has_line_map else float(row_ccov[r, i])
                        erun = float(row_lrun[r, i]) if has_line_map else float(row_crun[r, i])
                        if ecov < 0.36 or erun < 0.84:
                            out[r, i] = 0

                # Continuity repair/veto across rows:
                # 1) Fill a single-row break when neighbors are present and local
                # evidence is adequate.
                # 2) Remove isolated one-off segments unless local evidence is strong.
                if rows >= 3:
                    for i in range(1, nvl - 1):
                        for r in range(rows):
                            up = bool(r > 0 and out[r - 1, i] > 0)
                            dn = bool((r + 1) < rows and out[r + 1, i] > 0)
                            ecov = float(row_lcov[r, i]) if has_line_map else float(row_ccov[r, i])
                            erun = float(row_lrun[r, i]) if has_line_map else float(row_crun[r, i])

                            if out[r, i] == 0 and up and dn:
                                fill_cov = 0.24 if has_line_map else 0.28
                                fill_run = 0.54 if has_line_map else 0.58
                                if ecov >= fill_cov and erun >= fill_run:
                                    out[r, i] = 1
                                continue

                            if out[r, i] == 0:
                                continue

                            if (not up) and (not dn):
                                iso_cov = 0.34 if has_line_map else 0.38
                                iso_run = 0.74 if has_line_map else 0.78
                                if r == (rows - 1):
                                    iso_cov = max(iso_cov, 0.38 if has_line_map else 0.42)
                                    iso_run = max(iso_run, 0.84 if has_line_map else 0.86)
                                if ecov < iso_cov or erun < iso_run:
                                    out[r, i] = 0
        return out

    def _horizontal_presence(
        self,
        h_prob: np.ndarray,
        x_lines: list[int],
        y_lines: list[int],
        line_thr: float,
        h_color_prob: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        nyl = len(y_lines)
        cols = max(0, len(x_lines) - 1)
        out = np.zeros((nyl, cols), dtype=np.uint8)
        band = max(2, int(self.cfg.flexible_band_px))
        thr = max(0.20, float(line_thr) * 0.92)
        row_cov = np.zeros((nyl, cols), dtype=np.float32)
        row_run = np.zeros((nyl, cols), dtype=np.float32)
        row_ccov = np.zeros((nyl, cols), dtype=np.float32)
        row_crun = np.zeros((nyl, cols), dtype=np.float32)
        cband = max(1, int(round(0.45 * float(band))))
        cthr = max(0.20, float(self.cfg.h_color_peak_min) - 0.06)
        search_dy = max(1, int(round(0.24 * float(band))))
        for j, y in enumerate(y_lines):
            for c in range(cols):
                xx0, xx1 = int(x_lines[c]), int(x_lines[c + 1])
                if xx1 <= xx0:
                    continue
                lo = max(0, int(y) - search_dy)
                hi = min(h_prob.shape[0] - 1, int(y) + search_dy)
                best_cov = 0.0
                best_run = 0.0
                best_s = -1e18
                for py in range(lo, hi + 1):
                    covp = self._flex_cov_horizontal(h_prob, y=py, band=band, thr=thr, x0=xx0, x1=xx1)
                    yy0 = max(0, int(py) - band)
                    yy1 = min(h_prob.shape[0] - 1, int(py) + band)
                    segp = (np.max(h_prob[yy0 : yy1 + 1, xx0:xx1], axis=0) >= thr) if (yy1 >= yy0 and xx1 > xx0) else np.zeros((0,), dtype=np.uint8)
                    runp = _max_true_run_ratio(segp)
                    sp = float(0.70 * covp + 0.30 * runp - 0.0015 * abs(py - int(y)))
                    if sp > best_s:
                        best_s = sp
                        best_cov = float(covp)
                        best_run = float(runp)
                cov = float(best_cov)
                min_cov = max(float(self.cfg.min_line_cov), float(self.cfg.merge_presence_min_cov))
                run_ratio = float(best_run)
                row_cov[j, c] = cov
                row_run[j, c] = run_ratio

                ccov = 0.0
                crun = 0.0
                if h_color_prob is not None and h_color_prob.shape[:2] == h_prob.shape[:2]:
                    cbest_cov = 0.0
                    cbest_run = 0.0
                    cbest_s = -1e18
                    for py in range(lo, hi + 1):
                        covc = self._flex_cov_horizontal(h_color_prob, y=py, band=cband, thr=cthr, x0=xx0, x1=xx1)
                        cy0 = max(0, int(py) - cband)
                        cy1 = min(h_color_prob.shape[0] - 1, int(py) + cband)
                        cseg = (
                            np.max(h_color_prob[cy0 : cy1 + 1, xx0:xx1], axis=0) >= cthr
                        ) if (cy1 >= cy0 and xx1 > xx0) else np.zeros((0,), dtype=np.uint8)
                        runc = _max_true_run_ratio(cseg)
                        sc = float(0.68 * covc + 0.32 * runc - 0.0012 * abs(py - int(y)))
                        if sc > cbest_s:
                            cbest_s = sc
                            cbest_cov = float(covc)
                            cbest_run = float(runc)
                    ccov = float(cbest_cov)
                    crun = float(cbest_run)
                row_ccov[j, c] = ccov
                row_crun[j, c] = crun

                line_ok = cov >= min_cov and run_ratio >= float(self.cfg.presence_min_run_ratio)
                # Allow color-boundary separators (common in scorecards) when the
                # boundary is strong and long even if no dark printed stroke exists.
                color_ok = ccov >= 0.58 and crun >= 0.72
                if line_ok or color_ok:
                    out[j, c] = 1

        # If a row boundary is mostly continuous in color/edge evidence, fill it
        # across width. This fixes missed full-width band separators.
        if h_color_prob is not None and h_color_prob.shape[:2] == h_prob.shape[:2] and cols >= 3 and nyl >= 3:
            x0 = int(min(x_lines))
            x1 = int(max(x_lines))
            for j in range(1, nyl - 1):
                frac = float(np.mean(out[j] > 0))
                if frac >= 0.98:
                    continue
                y = int(y_lines[j])
                gcov = self._flex_cov_horizontal(h_color_prob, y=y, band=cband, thr=cthr, x0=x0, x1=x1 + 1)
                gy0 = max(0, y - cband)
                gy1 = min(h_color_prob.shape[0] - 1, y + cband)
                gseg = (
                    np.max(h_color_prob[gy0 : gy1 + 1, x0 : x1 + 1], axis=0) >= cthr
                ) if (gy1 >= gy0 and x1 > x0) else np.zeros((0,), dtype=np.uint8)
                grun = _max_true_run_ratio(gseg)
                color_hit_frac = float(np.mean((row_ccov[j] >= 0.56) | (row_crun[j] >= 0.70)))
                line_hit_frac = float(np.mean((row_cov[j] >= max(0.14, 0.70 * float(self.cfg.min_line_cov))) | (row_run[j] >= max(0.42, float(self.cfg.presence_min_run_ratio) - 0.04))))
                if gcov >= 0.54 and grun >= 0.82 and (color_hit_frac >= 0.68 or line_hit_frac >= 0.74):
                    out[j, :] = 1
        return out

    def _trim_weak_outer_lines(
        self,
        v_prob: np.ndarray,
        h_prob: np.ndarray,
        x_lines: list[int],
        y_lines: list[int],
        line_thr: float,
    ) -> tuple[list[int], list[int]]:
        # Remove spurious outer borders when they have weak global support.
        # This tightens table bbox to true grid limits.
        if len(x_lines) >= 3 and len(y_lines) >= 2:
            y0 = int(min(y_lines))
            y1 = int(max(y_lines))
            min_cov = float(max(0.12, 0.75 * self.cfg.outer_line_trim_min_cov))
            band = max(2, int(self.cfg.flexible_band_px))
            x_gaps = np.diff(np.array(x_lines, dtype=np.int32))
            x_med = float(np.median(x_gaps)) if x_gaps.size else 0.0
            x_med = max(1.0, x_med)
            keep_large_outer_ratio = 1.85
            while len(x_lines) >= 3:
                left_cov = self._flex_cov_vertical(v_prob, x=int(x_lines[0]), band=band, thr=line_thr, y0=y0, y1=y1 + 1)
                right_cov = self._flex_cov_vertical(v_prob, x=int(x_lines[-1]), band=band, thr=line_thr, y0=y0, y1=y1 + 1)
                dropped = False
                left_gap = float(x_lines[1] - x_lines[0])
                right_gap = float(x_lines[-1] - x_lines[-2])
                left_trim_ok = left_gap <= keep_large_outer_ratio * x_med
                right_trim_ok = right_gap <= keep_large_outer_ratio * x_med
                if bool(self.cfg.conservative_table_bbox):
                    if left_cov < 0.50 * min_cov:
                        left_trim_ok = True
                    if right_cov < 0.50 * min_cov:
                        right_trim_ok = True
                if left_cov < min_cov and left_cov <= right_cov and left_trim_ok:
                    x_lines = x_lines[1:]
                    dropped = True
                elif right_cov < min_cov and right_trim_ok:
                    x_lines = x_lines[:-1]
                    dropped = True
                if not dropped:
                    break

        if len(y_lines) >= 3 and len(x_lines) >= 2:
            x0 = int(min(x_lines))
            x1 = int(max(x_lines))
            min_cov = float(max(0.12, 0.75 * self.cfg.outer_line_trim_min_cov))
            band = max(2, int(self.cfg.flexible_band_px))
            y_gaps = np.diff(np.array(y_lines, dtype=np.int32))
            y_med = float(np.median(y_gaps)) if y_gaps.size else 0.0
            y_med = max(1.0, y_med)
            keep_large_outer_ratio = 1.85
            while len(y_lines) >= 3:
                top_cov = self._flex_cov_horizontal(h_prob, y=int(y_lines[0]), band=band, thr=line_thr, x0=x0, x1=x1 + 1)
                bot_cov = self._flex_cov_horizontal(h_prob, y=int(y_lines[-1]), band=band, thr=line_thr, x0=x0, x1=x1 + 1)
                dropped = False
                top_gap = float(y_lines[1] - y_lines[0])
                bot_gap = float(y_lines[-1] - y_lines[-2])
                top_trim_ok = top_gap <= keep_large_outer_ratio * y_med
                bot_trim_ok = bot_gap <= keep_large_outer_ratio * y_med
                if top_cov < min_cov and top_cov <= bot_cov and top_trim_ok:
                    y_lines = y_lines[1:]
                    dropped = True
                elif bot_cov < min_cov and bot_trim_ok:
                    y_lines = y_lines[:-1]
                    dropped = True
                if not dropped:
                    break

        if len(x_lines) < 2:
            x_lines = sorted(set(x_lines))
        if len(y_lines) < 2:
            y_lines = sorted(set(y_lines))
        return x_lines, y_lines

    def decode_all(
        self,
        table_prob: np.ndarray,
        v_prob: np.ndarray,
        h_prob: np.ndarray,
        j_prob: np.ndarray,
        image_bgr: Optional[np.ndarray] = None,
    ) -> list[DecodedGrid]:
        # Raw model maps.
        v_hard = v_prob
        h_hard = h_prob
        v_col: Optional[np.ndarray] = None
        v_line: Optional[np.ndarray] = None
        h_col: Optional[np.ndarray] = None
        h_line: Optional[np.ndarray] = None
        v_canny: Optional[np.ndarray] = None
        h_canny: Optional[np.ndarray] = None
        if bool(self.cfg.color_edge_enable) and image_bgr is not None:
            if image_bgr.shape[:2] == v_prob.shape[:2]:
                v_col, h_col = self._color_transition_maps(image_bgr)
                v_line = self._image_vertical_line_evidence_map(image_bgr)
                h_line = self._image_horizontal_line_evidence_map(image_bgr)
                v_canny, h_canny = self._image_canny_axis_maps(image_bgr)

        if bool(self.cfg.legacy_decode):
            v_dec = v_hard
            h_dec = h_hard
            v_line_fused = None
            h_line_fused = None
            h_soft = None
            j_dec = j_prob
            v_box = v_dec
            h_box = h_dec
        else:
            # Evidence-fused decode maps:
            # when model logits are under-confident, keep line evidence from image
            # transitions so table/grid decoding does not collapse to zero tables.
            v_dec = v_hard
            h_dec = h_hard
            v_line_fused = v_line
            h_line_fused = h_line
            if v_canny is not None and v_canny.shape[:2] == v_hard.shape[:2]:
                v_dec = np.maximum(v_dec, 0.90 * v_canny)
                v_line_fused = v_canny if v_line_fused is None else np.maximum(v_line_fused, 0.88 * v_canny)
            if h_canny is not None and h_canny.shape[:2] == h_hard.shape[:2]:
                h_dec = np.maximum(h_dec, 0.90 * h_canny)
                h_line_fused = h_canny if h_line_fused is None else np.maximum(h_line_fused, 0.88 * h_canny)
            if v_line is not None and v_line.shape[:2] == v_hard.shape[:2]:
                v_dec = np.maximum(v_dec, 0.86 * v_line)
            if h_col is not None and h_col.shape[:2] == h_hard.shape[:2]:
                h_dec = np.maximum(h_dec, 0.84 * h_col)
            if h_line is not None and h_line.shape[:2] == h_hard.shape[:2]:
                h_dec = np.maximum(h_dec, 0.88 * h_line)
            h_soft = h_col
            if h_line_fused is not None and h_line_fused.shape[:2] == h_hard.shape[:2]:
                h_soft = h_line_fused if h_soft is None else np.maximum(h_soft, 0.82 * h_line_fused)
            j_dec = j_prob
            if v_dec.shape[:2] == h_dec.shape[:2]:
                j_synth = cv2.GaussianBlur((v_dec * h_dec).astype(np.float32), (5, 5), 0)
                j_con = self._prob_contrast(j_prob)
                if j_con < max(0.035, 0.75 * float(self.cfg.min_line_prob_contrast)):
                    j_dec = np.maximum(j_prob, j_synth)
                else:
                    j_dec = np.maximum(j_prob, 0.55 * j_synth)
            v_box = v_dec
            h_box = h_dec

        h_all, w_all = table_prob.shape[:2]
        boxes = self._decode_table_bboxes(table_prob, v_box, h_box)
        out: list[DecodedGrid] = []
        for tid, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            if x1 <= x0 or y1 <= y0:
                continue
            v_crop = v_dec[y0 : y1 + 1, x0 : x1 + 1]
            h_crop = h_dec[y0 : y1 + 1, x0 : x1 + 1]
            if max(self._prob_contrast(v_crop), self._prob_contrast(h_crop)) < self.cfg.min_line_prob_contrast:
                continue

            x_lines, y_lines, line_thr, grid_score = self._decode_best_lines(
                v_dec,
                h_dec,
                j_dec,
                box,
                h_color_prob=h_soft,
                v_line_prob=v_line_fused,
            )
            if grid_score < float(self.cfg.min_grid_score):
                continue

            # Conservative mode: use raw vertical head for outer x-border trim
            # (reduces page-wide expansion), but keep fused horizontal evidence
            # so row structure does not collapse.
            trim_v = v_hard if bool(self.cfg.conservative_table_bbox) else v_dec
            trim_h = h_dec
            x_lines, y_lines = self._trim_weak_outer_lines(trim_v, trim_h, x_lines, y_lines, line_thr=line_thr)
            if len(x_lines) < 2 or len(y_lines) < 2:
                continue
            cols_final = max(0, len(x_lines) - 1)
            if cols_final < int(self.cfg.min_keep_cols):
                continue

            # Presence drives merged-cell decisions; use raw model line maps to avoid
            # color-edge artifacts splitting cells.
            v_pres = self._vertical_presence_with_color(
                v_hard,
                x_lines,
                y_lines,
                line_thr=line_thr,
                v_color_prob=v_col,
                v_line_prob=v_line_fused,
            )
            h_pres = self._horizontal_presence(
                h_hard,
                x_lines,
                y_lines,
                line_thr=line_thr,
                h_color_prob=h_soft,
            )
            v_polys = self._fit_vertical_polylines(
                x_lines=x_lines,
                y_lines=y_lines,
                v_prob=v_dec,
                v_line_prob=v_line_fused,
            )
            h_polys = self._fit_horizontal_polylines(
                x_lines=x_lines,
                y_lines=y_lines,
                h_prob=h_dec,
                h_color_prob=h_soft,
                h_line_prob=h_line_fused,
            )
            # Tight bbox from inferred outer grid lines (not coarse component bounds).
            if x_lines and y_lines:
                pad = max(1, int(round(0.004 * max(h_all, w_all))))
                bx0 = int(max(0, min(x_lines) - pad))
                bx1 = int(min(w_all - 1, max(x_lines) + pad))
                by0 = int(max(0, min(y_lines) - pad))
                by1 = int(min(h_all - 1, max(y_lines) + pad))
                box_out = (bx0, by0, bx1, by1)
            else:
                box_out = box
            out.append(
                DecodedGrid(
                    table_id=int(tid),
                    bbox=box_out,
                    x_lines=x_lines,
                    y_lines=y_lines,
                    v_presence=v_pres,
                    h_presence=h_pres,
                    v_polylines=v_polys,
                    h_polylines=h_polys,
                )
            )
        if not out and bool(self.cfg.conservative_table_bbox):
            # Safety fallback: if conservative localization over-prunes and no
            # table survives, fall back to standard decode.
            fb_cfg = replace(
                self.cfg,
                conservative_table_bbox=False,
            )
            return GridDecoder(fb_cfg).decode_all(table_prob, v_prob, h_prob, j_prob, image_bgr=image_bgr)
        return out


# -----------------------------------------------------------------------------
# Cell extraction
# -----------------------------------------------------------------------------


def _interp_poly_x_at_y(poly: list[list[int]], y: float) -> float:
    if not poly:
        return 0.0
    if len(poly) == 1:
        return float(poly[0][0])
    pts = sorted(poly, key=lambda p: p[1])
    yy = float(y)
    if yy <= float(pts[0][1]):
        return float(pts[0][0])
    if yy >= float(pts[-1][1]):
        return float(pts[-1][0])
    for i in range(len(pts) - 1):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
        if y1 <= y0:
            continue
        if y0 <= yy <= y1:
            t = (yy - y0) / max(1e-6, (y1 - y0))
            return x0 + t * (x1 - x0)
    return float(pts[-1][0])


def _interp_poly_y_at_x(poly: list[list[int]], x: float) -> float:
    if not poly:
        return 0.0
    if len(poly) == 1:
        return float(poly[0][1])
    pts = sorted(poly, key=lambda p: p[0])
    xx = float(x)
    if xx <= float(pts[0][0]):
        return float(pts[0][1])
    if xx >= float(pts[-1][0]):
        return float(pts[-1][1])
    for i in range(len(pts) - 1):
        x0, y0 = float(pts[i][0]), float(pts[i][1])
        x1, y1 = float(pts[i + 1][0]), float(pts[i + 1][1])
        if x1 <= x0:
            continue
        if x0 <= xx <= x1:
            t = (xx - x0) / max(1e-6, (x1 - x0))
            return y0 + t * (y1 - y0)
    return float(pts[-1][1])


def _intersect_vh_polys(v_poly: list[list[int]], h_poly: list[list[int]], x0: float, y0: float) -> tuple[float, float]:
    x = float(x0)
    y = float(y0)
    for _ in range(6):
        x = _interp_poly_x_at_y(v_poly, y)
        y = _interp_poly_y_at_x(h_poly, x)
    return float(x), float(y)


def _build_intersection_grid(
    x_lines: list[int],
    y_lines: list[int],
    v_polylines: Optional[list[list[list[int]]]],
    h_polylines: Optional[list[list[list[int]]]],
) -> Optional[list[list[tuple[float, float]]]]:
    if v_polylines is None or h_polylines is None:
        return None
    if len(v_polylines) != len(x_lines) or len(h_polylines) != len(y_lines):
        return None
    out: list[list[tuple[float, float]]] = []
    for r, y in enumerate(y_lines):
        row: list[tuple[float, float]] = []
        for c, x in enumerate(x_lines):
            px, py = _intersect_vh_polys(v_polylines[c], h_polylines[r], x0=float(x), y0=float(y))
            row.append((float(px), float(py)))
        out.append(row)
    return out


def extract_cells_from_decoded(image_bgr: np.ndarray, dec: DecodedGrid, output_dir: Path, cfg: InferConfig) -> dict:
    _ensure_dir(output_dir)
    x_lines = dec.x_lines
    y_lines = dec.y_lines
    rows = max(0, len(y_lines) - 1)
    cols = max(0, len(x_lines) - 1)
    img_h, img_w = image_bgr.shape[:2]
    inset = max(0, int(cfg.cell_inset_px))
    grid_pts = _build_intersection_grid(x_lines, y_lines, dec.v_polylines, dec.h_polylines)

    def _crop_quad(
        quad_xy: list[tuple[float, float]],
        min_w: int,
        min_h: int,
    ) -> tuple[Optional[np.ndarray], list[int], list[list[float]]]:
        q = np.asarray(quad_xy, dtype=np.float32).reshape(4, 2)
        q[:, 0] = np.clip(q[:, 0], 0.0, float(max(0, img_w - 1)))
        q[:, 1] = np.clip(q[:, 1], 0.0, float(max(0, img_h - 1)))
        x0 = int(max(0, min(img_w, int(np.floor(np.min(q[:, 0]))))))
        x1 = int(max(0, min(img_w, int(np.ceil(np.max(q[:, 0])) + 1))))
        y0 = int(max(0, min(img_h, int(np.floor(np.min(q[:, 1]))))))
        y1 = int(max(0, min(img_h, int(np.ceil(np.max(q[:, 1])) + 1))))
        bbox = [x0, y0, x1, y1]
        if (x1 - x0) <= 1 or (y1 - y0) <= 1:
            return None, bbox, q.astype(np.float32).tolist()

        w_top = float(np.linalg.norm(q[1] - q[0]))
        w_bot = float(np.linalg.norm(q[2] - q[3]))
        h_left = float(np.linalg.norm(q[3] - q[0]))
        h_right = float(np.linalg.norm(q[2] - q[1]))
        tw = int(round(max(2.0, 0.5 * (w_top + w_bot))))
        th = int(round(max(2.0, 0.5 * (h_left + h_right))))
        if tw < int(max(2, min_w)) or th < int(max(2, min_h)):
            return None, bbox, q.astype(np.float32).tolist()

        dst = np.array(
            [[0.0, 0.0], [float(tw - 1), 0.0], [float(tw - 1), float(th - 1)], [0.0, float(th - 1)]],
            dtype=np.float32,
        )
        m = cv2.getPerspectiveTransform(q, dst)
        patch = cv2.warpPerspective(
            image_bgr,
            m,
            (tw, th),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        if inset > 0 and patch is not None and patch.size > 0 and patch.shape[0] > 2 * inset + 2 and patch.shape[1] > 2 * inset + 2:
            patch = patch[inset : patch.shape[0] - inset, inset : patch.shape[1] - inset]
        return patch, bbox, q.astype(np.float32).tolist()

    # Base matrix cells: always emitted row-major for OCR.
    base_cells: list[dict] = []
    matrix_base: list[list[dict]] = [[{} for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cx0, cx1 = int(x_lines[c]), int(x_lines[c + 1])
            cy0, cy1 = int(y_lines[r]), int(y_lines[r + 1])
            quad_pts: Optional[list[list[float]]] = None
            patch: Optional[np.ndarray] = None
            if grid_pts is not None:
                tl = grid_pts[r][c]
                tr = grid_pts[r][c + 1]
                br = grid_pts[r + 1][c + 1]
                bl = grid_pts[r + 1][c]
                patch, bbox, quad_pts = _crop_quad([tl, tr, br, bl], min_w=2, min_h=2)
                cx0, cy0, cx1, cy1 = bbox
            else:
                cx0 = max(0, min(img_w, cx0 + inset))
                cx1 = max(0, min(img_w, cx1 - inset))
                cy0 = max(0, min(img_h, cy0 + inset))
                cy1 = max(0, min(img_h, cy1 - inset))
                if cx1 <= cx0:
                    cx0, cx1 = max(0, min(img_w, int(x_lines[c]))), max(0, min(img_w, int(x_lines[c + 1])))
                if cy1 <= cy0:
                    cy0, cy1 = max(0, min(img_h, int(y_lines[r]))), max(0, min(img_h, int(y_lines[r + 1])))
            path = None
            if grid_pts is not None:
                if patch is not None and patch.size > 0:
                    p = output_dir / f"base_r{r:02d}_c{c:02d}.png"
                    cv2.imwrite(str(p), patch)
                    path = p.name
            else:
                if cx1 > cx0 and cy1 > cy0:
                    cell = image_bgr[cy0:cy1, cx0:cx1]
                    if cell.size > 0:
                        p = output_dir / f"base_r{r:02d}_c{c:02d}.png"
                        cv2.imwrite(str(p), cell)
                        path = p.name

            base_id = int(r * cols + c)
            rec = {
                "base_cell_id": base_id,
                "r": int(r),
                "c": int(c),
                "bbox_xyxy": [int(cx0), int(cy0), int(cx1), int(cy1)],
                "path": path,
            }
            if quad_pts is not None:
                rec["quad_xy"] = [[float(v[0]), float(v[1])] for v in quad_pts]
            base_cells.append(rec)
            matrix_base[r][c] = {
                "base_cell_id": base_id,
                "base_bbox_xyxy": [int(cx0), int(cy0), int(cx1), int(cy1)],
                "base_path": path,
            }
            if quad_pts is not None:
                matrix_base[r][c]["base_quad_xy"] = [[float(v[0]), float(v[1])] for v in quad_pts]

    parent = list(range(rows * cols))

    def idx(r: int, c: int) -> int:
        return r * cols + c

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Missing vertical boundary => merge horizontal neighbors.
    for r in range(rows):
        for c in range(cols - 1):
            if dec.v_presence[r, c + 1] == 0:
                union(idx(r, c), idx(r, c + 1))

    # Missing horizontal boundary => merge vertical neighbors.
    for r in range(rows - 1):
        for c in range(cols):
            if dec.h_presence[r + 1, c] == 0:
                union(idx(r, c), idx(r + 1, c))

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            rt = find(idx(r, c))
            groups.setdefault(rt, []).append((r, c))

    merged = []
    for _, members in groups.items():
        rs = [m[0] for m in members]
        cs = [m[1] for m in members]
        merged.append((min(rs), min(cs), max(rs), max(cs)))
    merged.sort(key=lambda t: (t[0], t[1], t[2], t[3]))

    cells = []
    for cid, (r0, c0, r1, c1) in enumerate(merged):
        x0 = int(x_lines[c0])
        x1 = int(x_lines[c1 + 1])
        y0 = int(y_lines[r0])
        y1 = int(y_lines[r1 + 1])
        cx0 = max(0, min(img_w, x0 + inset))
        cx1 = max(0, min(img_w, x1 - inset))
        cy0 = max(0, min(img_h, y0 + inset))
        cy1 = max(0, min(img_h, y1 - inset))
        cell_patch: Optional[np.ndarray] = None
        cell_quad: Optional[list[list[float]]] = None
        if grid_pts is not None:
            tl = grid_pts[r0][c0]
            tr = grid_pts[r0][c1 + 1]
            br = grid_pts[r1 + 1][c1 + 1]
            bl = grid_pts[r1 + 1][c0]
            cell_patch, bbox, cell_quad = _crop_quad([tl, tr, br, bl], min_w=int(cfg.min_cell_w), min_h=int(cfg.min_cell_h))
            cx0, cy0, cx1, cy1 = bbox

        path = None
        if grid_pts is not None:
            if cell_patch is not None and cell_patch.size > 0:
                p = output_dir / f"cell_{cid:04d}_r{r0:02d}_c{c0:02d}_rs{(r1-r0+1):02d}_cs{(c1-c0+1):02d}.png"
                cv2.imwrite(str(p), cell_patch)
                path = p.name
        elif cx1 > cx0 and cy1 > cy0 and (cx1 - cx0) >= cfg.min_cell_w and (cy1 - cy0) >= cfg.min_cell_h:
            cell = image_bgr[cy0:cy1, cx0:cx1]
            if cell.size > 0:
                p = output_dir / f"cell_{cid:04d}_r{r0:02d}_c{c0:02d}_rs{(r1-r0+1):02d}_cs{(c1-c0+1):02d}.png"
                cv2.imwrite(str(p), cell)
                path = p.name

        cmeta = {
            "cell_id": cid,
            "r0": r0,
            "c0": c0,
            "r1": r1,
            "c1": c1,
            "rowspan": r1 - r0 + 1,
            "colspan": c1 - c0 + 1,
            "path": path,
        }
        if cell_quad is not None:
            cmeta["quad_xy"] = [[float(v[0]), float(v[1])] for v in cell_quad]
        cells.append(cmeta)

    inv = {}
    for c in cells:
        for r in range(c["r0"], c["r1"] + 1):
            for cc in range(c["c0"], c["c1"] + 1):
                inv[(r, cc)] = c

    mat: list[list[dict]] = [[{} for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cell = inv[(r, c)]
            anchor = bool(r == cell["r0"] and c == cell["c0"])
            merged_info = {
                "cell_id": int(cell["cell_id"]),
                "anchor": anchor,
                "rowspan": int(cell["rowspan"]),
                "colspan": int(cell["colspan"]),
                "path": cell["path"] if anchor else None,
            }
            mat[r][c] = {
                **matrix_base[r][c],
                "cell_id": merged_info["cell_id"],
                "anchor": merged_info["anchor"],
                "rowspan": merged_info["rowspan"],
                "colspan": merged_info["colspan"],
                "path": merged_info["path"],
                "merged": merged_info,
            }

    out = {
        "table_id": int(dec.table_id),
        "bbox_xyxy": [int(v) for v in dec.bbox],
        "x_lines": [int(v) for v in dec.x_lines],
        "y_lines": [int(v) for v in dec.y_lines],
        "v_polylines": dec.v_polylines if dec.v_polylines is not None else None,
        "h_polylines": dec.h_polylines if dec.h_polylines is not None else None,
        "rows_base": rows,
        "cols_base": cols,
        "base_cells": base_cells,
        "cells": cells,
        "matrix": mat,
    }
    (output_dir / "matrix_index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# -----------------------------------------------------------------------------
# Torch inference wrappers
# -----------------------------------------------------------------------------


def _load_model(weights: Path, device: str) -> tuple[Any, Any, Any, dict]:
    torch, nn, F, _, _, _ = _import_torch()
    UNetSmall = _build_unet_small(nn)
    model = UNetSmall(in_ch=3, out_ch=4)

    payload = torch.load(str(weights), map_location=torch.device(device))
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
        payload = {}

    model.load_state_dict(state_dict)
    model = model.to(torch.device(device))
    model.eval()
    return model, torch, F, payload


def _predict_maps(model: Any, torch: Any, F: Any, image_bgr: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    img = image_bgr.astype(np.float32) / 255.0
    x = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(torch.device(device))

    _, _, h, w = x.shape
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    probs = probs[:, :, :h, :w][0].cpu().numpy().astype(np.float32)  # [4,H,W]
    table_prob = probs[0]
    v_prob = probs[1]
    h_prob = probs[2]
    j_prob = probs[3]
    return table_prob, v_prob, h_prob, j_prob


# -----------------------------------------------------------------------------
# CLI handlers
# -----------------------------------------------------------------------------


def _train_cli(args: argparse.Namespace) -> None:
    pretrain_images_dir = Path(args.pretrain_images_dir) if args.pretrain_images_dir else None
    cfg = TrainConfig(
        labels_dir=Path(args.labels_dir),
        out_weights=Path(args.out_weights),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        train_size=int(args.train_size),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
        junction_dilate_px=int(args.junction_dilate_px),
        num_workers=int(args.num_workers),
        pretrain_images_dir=pretrain_images_dir,
        pretrain_epochs=int(args.pretrain_epochs),
        pretrain_batch_size=int(args.pretrain_batch_size),
        pretrain_lr=float(args.pretrain_lr),
        pretrain_weight_decay=float(args.pretrain_weight_decay),
        pretrain_max_images=int(args.pretrain_max_images),
    )
    train_model(cfg)


def _run_one_with_model(
    model: Any,
    torch: Any,
    F: Any,
    input_image: Path,
    output_dir: Path,
    infer_cfg: InferConfig,
    device: str,
    pre_cfg: Optional[PreprocessConfig] = None,
) -> dict:
    _ensure_dir(output_dir)
    raw = cv2.imread(str(input_image), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"Failed to load image: {input_image}")

    prep = preprocess_scorecard(raw, pre_cfg or PreprocessConfig(ensure_upright=False))
    image = prep.image_bgr
    cv2.imwrite(str(output_dir / "debug_preprocessed.png"), image)

    table_prob, v_prob, h_prob, j_prob = _predict_maps(model, torch, F, image, device=device)

    cv2.imwrite(str(output_dir / "debug_table_prob.png"), np.clip(table_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "debug_v_prob.png"), np.clip(v_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "debug_h_prob.png"), np.clip(h_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "debug_junction_prob.png"), np.clip(j_prob * 255.0, 0, 255).astype(np.uint8))

    decoder = GridDecoder(infer_cfg)
    decoded = decoder.decode_all(table_prob, v_prob, h_prob, j_prob, image_bgr=image)

    overlay = image.copy()
    for dec in decoded:
        x0, y0, x1, y1 = dec.bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 0), 2)
        # Draw grid segments according to presence maps; prefer warped polyline
        # intersections when available.
        rows = max(0, len(dec.y_lines) - 1)
        cols = max(0, len(dec.x_lines) - 1)
        gpts = _build_intersection_grid(dec.x_lines, dec.y_lines, dec.v_polylines, dec.h_polylines)

        for i, x in enumerate(dec.x_lines):
            for r in range(rows):
                if dec.v_presence.shape == (rows, len(dec.x_lines)):
                    if i < dec.v_presence.shape[1] and r < dec.v_presence.shape[0] and int(dec.v_presence[r, i]) == 0:
                        continue
                if gpts is not None:
                    p0 = gpts[r][i]
                    p1 = gpts[r + 1][i]
                    cv2.line(
                        overlay,
                        (int(round(p0[0])), int(round(p0[1]))),
                        (int(round(p1[0])), int(round(p1[1]))),
                        (255, 80, 80),
                        1,
                    )
                else:
                    ya = int(dec.y_lines[r])
                    yb = int(dec.y_lines[r + 1])
                    if yb <= ya:
                        continue
                    cv2.line(overlay, (int(x), ya), (int(x), yb), (255, 80, 80), 1)

        for j, y in enumerate(dec.y_lines):
            for c in range(cols):
                if dec.h_presence.shape == (len(dec.y_lines), cols):
                    if j < dec.h_presence.shape[0] and c < dec.h_presence.shape[1] and int(dec.h_presence[j, c]) == 0:
                        continue
                if gpts is not None:
                    p0 = gpts[j][c]
                    p1 = gpts[j][c + 1]
                    cv2.line(
                        overlay,
                        (int(round(p0[0])), int(round(p0[1]))),
                        (int(round(p1[0])), int(round(p1[1]))),
                        (80, 80, 255),
                        1,
                    )
                else:
                    xa = int(dec.x_lines[c])
                    xb = int(dec.x_lines[c + 1])
                    if xb <= xa:
                        continue
                    cv2.line(overlay, (xa, int(y)), (xb, int(y)), (80, 80, 255), 1)
        cv2.putText(
            overlay,
            f"table_{dec.table_id:02d}",
            (x0 + 8, max(18, y0 + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_dir / "debug_grid_overlay.png"), overlay)

    table_results = []
    for dec in decoded:
        tdir = output_dir / f"table_{dec.table_id:02d}" / "cells"
        matrix = extract_cells_from_decoded(image, dec, tdir, infer_cfg)
        table_results.append(matrix)

    payload = {
        "image": str(input_image),
        "upright_rotation_degrees": int(prep.upright_rotation_degrees),
        "table_count": len(table_results),
        "tables": table_results,
    }
    (output_dir / "image_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_one(
    weights: Path,
    input_image: Path,
    output_dir: Path,
    infer_cfg: InferConfig,
    device: str,
    pre_cfg: Optional[PreprocessConfig] = None,
) -> dict:
    model, torch, F, _ = _load_model(weights=weights, device=device)
    return _run_one_with_model(
        model=model,
        torch=torch,
        F=F,
        input_image=input_image,
        output_dir=output_dir,
        infer_cfg=infer_cfg,
        device=device,
        pre_cfg=pre_cfg,
    )


def _infer_cli(args: argparse.Namespace) -> None:
    cfg = InferConfig(
        table_thresh=float(args.table_thresh),
        line_thresh=float(args.line_thresh),
        junction_thresh=float(args.junction_thresh),
        max_tables=int(args.max_tables),
        min_table_area_ratio=float(args.min_table_area_ratio),
        max_component_fill_ratio=float(args.max_component_fill_ratio),
        min_prob_contrast=float(args.min_prob_contrast),
        min_line_prob_contrast=float(args.min_line_prob_contrast),
        min_box_line_energy=float(args.min_box_line_energy),
        min_grid_score=float(args.min_grid_score),
        min_rows=int(args.min_rows),
        max_rows=int(args.max_rows),
        min_cols=int(args.min_cols),
        max_cols=int(args.max_cols),
        min_keep_cols=int(args.min_keep_cols),
        min_gap_px=int(args.min_gap_px),
        peak_rel_thresh=float(args.peak_rel_thresh),
        min_line_cov=float(args.min_line_cov),
        line_band_px=int(args.line_band_px),
        flexible_band_px=int(args.flexible_band_px),
        auto_tune_decode=not bool(args.no_auto_tune),
        cell_inset_px=int(args.cell_inset),
        min_cell_w=int(args.min_cell_w),
        min_cell_h=int(args.min_cell_h),
    )

    result = _run_one(
        weights=Path(args.weights),
        input_image=Path(args.input),
        output_dir=Path(args.output_dir),
        infer_cfg=cfg,
        device=str(args.device),
        pre_cfg=PreprocessConfig(ensure_upright=bool(args.upright)),
    )
    print(f"Image: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"tables={result['table_count']} rot={result['upright_rotation_degrees']}")
    for t in result["tables"]:
        print(f"  table_{int(t['table_id']):02d}: rows={int(t['rows_base'])} cols={int(t['cols_base'])} merged_cells={len(t['cells'])}")


def _batch_cli(args: argparse.Namespace) -> None:
    cfg = InferConfig(
        table_thresh=float(args.table_thresh),
        line_thresh=float(args.line_thresh),
        junction_thresh=float(args.junction_thresh),
        max_tables=int(args.max_tables),
        min_table_area_ratio=float(args.min_table_area_ratio),
        max_component_fill_ratio=float(args.max_component_fill_ratio),
        min_prob_contrast=float(args.min_prob_contrast),
        min_line_prob_contrast=float(args.min_line_prob_contrast),
        min_box_line_energy=float(args.min_box_line_energy),
        min_grid_score=float(args.min_grid_score),
        min_rows=int(args.min_rows),
        max_rows=int(args.max_rows),
        min_cols=int(args.min_cols),
        max_cols=int(args.max_cols),
        min_keep_cols=int(args.min_keep_cols),
        min_gap_px=int(args.min_gap_px),
        peak_rel_thresh=float(args.peak_rel_thresh),
        min_line_cov=float(args.min_line_cov),
        line_band_px=int(args.line_band_px),
        flexible_band_px=int(args.flexible_band_px),
        auto_tune_decode=not bool(args.no_auto_tune),
        cell_inset_px=int(args.cell_inset),
        min_cell_w=int(args.min_cell_w),
        min_cell_h=int(args.min_cell_h),
    )

    inp_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)
    _ensure_dir(out_root)

    imgs = sorted(inp_dir.glob("*.png"))
    if not imgs:
        print(f"No PNG images found in {inp_dir}")
        return

    model, torch, F, _ = _load_model(weights=Path(args.weights), device=str(args.device))
    for p in imgs:
        out = out_root / p.stem
        result = _run_one_with_model(
            model=model,
            torch=torch,
            F=F,
            input_image=p,
            output_dir=out,
            infer_cfg=cfg,
            device=str(args.device),
            pre_cfg=PreprocessConfig(ensure_upright=bool(args.upright)),
        )
        table_str = ", ".join([f"t{int(t['table_id'])}:r{int(t['rows_base'])}c{int(t['cols_base'])}" for t in result["tables"]])
        print(f"{p.name}: tables={result['table_count']} [{table_str}] rot={result['upright_rotation_degrees']}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Segmentation-first scorecard extractor (U-Net + flexible decoder)")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train 4-head segmentation model")
    tr.add_argument("--labels_dir", required=True)
    tr.add_argument("--out_weights", default="checkpoints/scorecard_seg_unet.pt")
    tr.add_argument("--epochs", type=int, default=45)
    tr.add_argument("--batch_size", type=int, default=2)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--weight_decay", type=float, default=1e-4)
    tr.add_argument("--train_size", type=int, default=768)
    tr.add_argument("--val_ratio", type=float, default=0.18)
    tr.add_argument("--seed", type=int, default=7)
    tr.add_argument("--junction_dilate_px", type=int, default=4)
    tr.add_argument("--num_workers", type=int, default=0)
    tr.add_argument("--pretrain_images_dir", default="")
    tr.add_argument("--pretrain_epochs", type=int, default=0)
    tr.add_argument("--pretrain_batch_size", type=int, default=4)
    tr.add_argument("--pretrain_lr", type=float, default=3e-4)
    tr.add_argument("--pretrain_weight_decay", type=float, default=1e-5)
    tr.add_argument("--pretrain_max_images", type=int, default=0, help="0 means all images")
    tr.set_defaults(func=_train_cli)

    inf = sub.add_parser("infer", help="Infer one image")
    inf.add_argument("--weights", required=True)
    inf.add_argument("--input", required=True)
    inf.add_argument("--output_dir", default="seg_cells")
    inf.add_argument("--device", default="cpu")
    inf.add_argument("--table_thresh", type=float, default=0.48)
    inf.add_argument("--line_thresh", type=float, default=0.47)
    inf.add_argument("--junction_thresh", type=float, default=0.38)
    inf.add_argument("--max_tables", type=int, default=2)
    inf.add_argument("--min_table_area_ratio", type=float, default=0.010)
    inf.add_argument("--max_component_fill_ratio", type=float, default=0.96)
    inf.add_argument("--min_prob_contrast", type=float, default=0.050)
    inf.add_argument("--min_line_prob_contrast", type=float, default=0.045)
    inf.add_argument("--min_box_line_energy", type=float, default=0.006)
    inf.add_argument("--min_grid_score", type=float, default=1.5)
    inf.add_argument("--min_rows", type=int, default=4)
    inf.add_argument("--max_rows", type=int, default=26)
    inf.add_argument("--min_cols", type=int, default=4)
    inf.add_argument("--max_cols", type=int, default=34)
    inf.add_argument("--min_keep_cols", type=int, default=9)
    inf.add_argument("--min_gap_px", type=int, default=8)
    inf.add_argument("--peak_rel_thresh", type=float, default=0.12)
    inf.add_argument("--min_line_cov", type=float, default=0.24)
    inf.add_argument("--line_band_px", type=int, default=4)
    inf.add_argument("--flexible_band_px", type=int, default=10)
    inf.add_argument("--cell_inset", type=int, default=2)
    inf.add_argument("--min_cell_w", type=int, default=10)
    inf.add_argument("--min_cell_h", type=int, default=10)
    inf.add_argument("--no_auto_tune", action="store_true")
    inf.add_argument("--upright", action="store_true", help="Enable 90/180 upright normalization (disabled by default).")
    inf.set_defaults(func=_infer_cli)

    bt = sub.add_parser("batch", help="Infer all PNG images in a folder")
    bt.add_argument("--weights", required=True)
    bt.add_argument("--input_dir", required=True)
    bt.add_argument("--output_dir", default="seg_batch_cells")
    bt.add_argument("--device", default="cpu")
    bt.add_argument("--table_thresh", type=float, default=0.48)
    bt.add_argument("--line_thresh", type=float, default=0.47)
    bt.add_argument("--junction_thresh", type=float, default=0.38)
    bt.add_argument("--max_tables", type=int, default=2)
    bt.add_argument("--min_table_area_ratio", type=float, default=0.010)
    bt.add_argument("--max_component_fill_ratio", type=float, default=0.96)
    bt.add_argument("--min_prob_contrast", type=float, default=0.050)
    bt.add_argument("--min_line_prob_contrast", type=float, default=0.045)
    bt.add_argument("--min_box_line_energy", type=float, default=0.006)
    bt.add_argument("--min_grid_score", type=float, default=1.5)
    bt.add_argument("--min_rows", type=int, default=4)
    bt.add_argument("--max_rows", type=int, default=26)
    bt.add_argument("--min_cols", type=int, default=4)
    bt.add_argument("--max_cols", type=int, default=34)
    bt.add_argument("--min_keep_cols", type=int, default=9)
    bt.add_argument("--min_gap_px", type=int, default=8)
    bt.add_argument("--peak_rel_thresh", type=float, default=0.12)
    bt.add_argument("--min_line_cov", type=float, default=0.24)
    bt.add_argument("--line_band_px", type=int, default=4)
    bt.add_argument("--flexible_band_px", type=int, default=10)
    bt.add_argument("--cell_inset", type=int, default=2)
    bt.add_argument("--min_cell_w", type=int, default=10)
    bt.add_argument("--min_cell_h", type=int, default=10)
    bt.add_argument("--no_auto_tune", action="store_true")
    bt.add_argument("--upright", action="store_true", help="Enable 90/180 upright normalization (disabled by default).")
    bt.set_defaults(func=_batch_cli)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
