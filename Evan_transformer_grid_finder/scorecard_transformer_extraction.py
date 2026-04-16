"""Transformer-first golf scorecard extractor.

This module provides:
1) Self-supervised pretraining on unlabeled scorecards (including PDF folders).
2) Supervised fine-tuning on labeled masks (table / vertical / horizontal / junction).
3) Inference + batch extraction to OCR-ready cell PNGs using existing grid decoding logic.

Model family:
- Pyramid Vision Transformer-style encoder (spatial-reduction attention).
- Lightweight FPN-style decoder to dense segmentation heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import argparse
import json
import random
import re
import time

import cv2
import numpy as np

from scorecard_preprocessing import (
    preprocess_scorecard,
    PreprocessConfig,
    convert_pdf_to_png,
    _ocr_readability_score,
)
from scorecard_segmentation_extraction import (
    _ensure_dir,
    _collect_label_jsons,
    _load_label_sample,
    _resize_with_pad,
    GridDecoder,
    InferConfig,
    DecodedGrid,
    extract_cells_from_decoded,
)


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def _collect_image_files(root_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    out: list[Path] = []
    for p in sorted(root_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
    return out


def _collect_pdf_files(root_dir: Path) -> list[Path]:
    return sorted([p for p in root_dir.rglob("*.pdf") if p.is_file()])


def _sanitize_stem(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return s or "scorecard"


def _prepare_unlabeled_images(
    src_dir: Path,
    cache_dir: Path,
    pdf_dpi: int,
    max_images: int,
) -> list[Path]:
    imgs = _collect_image_files(src_dir)
    pdfs = _collect_pdf_files(src_dir)

    if pdfs:
        _ensure_dir(cache_dir)
        for i, pdf in enumerate(pdfs):
            prefix = f"{i:04d}_{_sanitize_stem(pdf.stem)}"
            try:
                pages = convert_pdf_to_png(pdf, cache_dir, dpi=int(pdf_dpi), prefix=prefix)
                imgs.extend(pages)
            except Exception as exc:
                print(f"[warn] pdf convert failed: {pdf.name} ({exc})")

    uniq = sorted(set(p.resolve() for p in imgs if p.exists()))
    if max_images > 0 and len(uniq) > max_images:
        rng = random.Random(7)
        uniq = list(uniq)
        rng.shuffle(uniq)
        uniq = sorted(uniq[:max_images])
    return uniq


def _set_seed(seed: int, torch: Any = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _import_torch() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import Dataset, DataLoader
        from torch.utils.data import random_split
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for scorecard_transformer_extraction.py. "
            "Install torch/torchvision and rerun."
        ) from exc
    return torch, nn, F, Dataset, DataLoader, random_split


def _select_device(torch: Any, requested: str = "auto") -> str:
    req = str(requested).strip().lower()
    mps_ok = bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )

    if req in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if mps_ok:
            return "mps"
        return "cpu"
    if req == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("Requested --device=cuda but CUDA is not available.")
    if req == "mps":
        if mps_ok:
            return "mps"
        raise RuntimeError("Requested --device=mps but MPS is not available in this PyTorch build/runtime.")
    if req == "cpu":
        return "cpu"
    raise ValueError(f"Unknown device '{requested}'. Use auto|cpu|mps|cuda.")


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


def _build_transformer_model(nn: Any, torch: Any) -> Any:
    class DropPath(nn.Module):
        def __init__(self, drop_prob: float = 0.0):
            super().__init__()
            self.drop_prob = float(drop_prob)

        def forward(self, x: Any) -> Any:
            if self.drop_prob <= 0.0 or not self.training:
                return x
            keep_prob = 1.0 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            rnd = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            rnd.floor_()
            return x * rnd / keep_prob

    class MLP(nn.Module):
        def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
            super().__init__()
            hidden = int(round(dim * mlp_ratio))
            self.fc1 = nn.Linear(dim, hidden)
            self.act = nn.GELU()
            self.fc2 = nn.Linear(hidden, dim)
            self.drop = nn.Dropout(drop)

        def forward(self, x: Any) -> Any:
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class SpatialReductionAttention(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            sr_ratio: int = 1,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
        ):
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
            self.dim = int(dim)
            self.num_heads = int(num_heads)
            self.head_dim = self.dim // self.num_heads
            self.scale = self.head_dim ** -0.5
            self.sr_ratio = max(1, int(sr_ratio))

            self.q = nn.Linear(self.dim, self.dim, bias=True)
            self.kv = nn.Linear(self.dim, self.dim * 2, bias=True)
            self.proj = nn.Linear(self.dim, self.dim, bias=True)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj_drop = nn.Dropout(proj_drop)

            if self.sr_ratio > 1:
                self.sr = nn.Conv2d(
                    self.dim,
                    self.dim,
                    kernel_size=self.sr_ratio,
                    stride=self.sr_ratio,
                    padding=0,
                    bias=False,
                )
                self.norm = nn.LayerNorm(self.dim)
            else:
                self.sr = None
                self.norm = None

        def forward(self, x: Any, h: int, w: int) -> Any:
            b, n, c = x.shape
            q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

            if self.sr is not None:
                xr = x.transpose(1, 2).reshape(b, c, h, w)
                xr = self.sr(xr)
                xr = xr.reshape(b, c, -1).transpose(1, 2)
                xr = self.norm(xr)
            else:
                xr = x

            kv = self.kv(xr).reshape(b, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            out = (attn @ v).transpose(1, 2).reshape(b, n, c)
            out = self.proj(out)
            out = self.proj_drop(out)
            return out

    class TransformerBlock(nn.Module):
        def __init__(
            self,
            dim: int,
            num_heads: int,
            sr_ratio: int,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path: float = 0.0,
        ):
            super().__init__()
            self.norm1 = nn.LayerNorm(dim)
            self.attn = SpatialReductionAttention(
                dim=dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
            self.drop_path = DropPath(drop_path)
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = MLP(dim=dim, mlp_ratio=mlp_ratio, drop=drop)

        def forward(self, x: Any, h: int, w: int) -> Any:
            x = x + self.drop_path(self.attn(self.norm1(x), h=h, w=w))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

    class OverlapPatchEmbed(nn.Module):
        def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int):
            super().__init__()
            pad = kernel // 2
            self.proj = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel,
                stride=stride,
                padding=pad,
                bias=False,
            )
            self.norm = nn.LayerNorm(out_ch)

        def forward(self, x: Any) -> Any:
            x = self.proj(x)
            b, c, h, w = x.shape
            t = x.flatten(2).transpose(1, 2)
            t = self.norm(t)
            x = t.transpose(1, 2).reshape(b, c, h, w)
            return x

    class PyramidTransformerEncoder(nn.Module):
        def __init__(
            self,
            in_ch: int = 3,
            embed_dims: tuple[int, int, int, int] = (64, 128, 256, 384),
            depths: tuple[int, int, int, int] = (2, 2, 2, 2),
            num_heads: tuple[int, int, int, int] = (1, 2, 4, 6),
            sr_ratios: tuple[int, int, int, int] = (8, 4, 2, 1),
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            attn_drop: float = 0.0,
            drop_path_rate: float = 0.10,
        ):
            super().__init__()
            self.embed_dims = embed_dims

            self.patch_embeds = nn.ModuleList(
                [
                    OverlapPatchEmbed(in_ch, embed_dims[0], kernel=7, stride=4),
                    OverlapPatchEmbed(embed_dims[0], embed_dims[1], kernel=3, stride=2),
                    OverlapPatchEmbed(embed_dims[1], embed_dims[2], kernel=3, stride=2),
                    OverlapPatchEmbed(embed_dims[2], embed_dims[3], kernel=3, stride=2),
                ]
            )

            dpr = torch.linspace(0.0, float(drop_path_rate), int(sum(depths))).tolist()
            cur = 0
            self.stages = nn.ModuleList()
            self.norms = nn.ModuleList()
            for i in range(4):
                blocks = []
                for _ in range(depths[i]):
                    blocks.append(
                        TransformerBlock(
                            dim=embed_dims[i],
                            num_heads=num_heads[i],
                            sr_ratio=sr_ratios[i],
                            mlp_ratio=mlp_ratio,
                            drop=drop,
                            attn_drop=attn_drop,
                            drop_path=dpr[cur],
                        )
                    )
                    cur += 1
                self.stages.append(nn.ModuleList(blocks))
                self.norms.append(nn.LayerNorm(embed_dims[i]))

        def forward(self, x: Any) -> list[Any]:
            feats = []
            for i in range(4):
                x = self.patch_embeds[i](x)
                b, c, h, w = x.shape
                t = x.flatten(2).transpose(1, 2)
                for blk in self.stages[i]:
                    t = blk(t, h=h, w=w)
                t = self.norms[i](t)
                x = t.transpose(1, 2).reshape(b, c, h, w)
                feats.append(x)
            return feats

    def _gn(ch: int) -> Any:
        for g in (32, 16, 8, 4, 2):
            if ch % g == 0:
                return nn.GroupNorm(g, ch)
        return nn.GroupNorm(1, ch)

    class ScorecardTransformerSeg(nn.Module):
        def __init__(
            self,
            in_ch: int = 3,
            out_ch: int = 4,
            embed_dims: tuple[int, int, int, int] = (64, 128, 256, 384),
            decoder_dim: int = 128,
        ):
            super().__init__()
            self.encoder = PyramidTransformerEncoder(in_ch=in_ch, embed_dims=embed_dims)
            self.proj = nn.ModuleList([nn.Conv2d(c, decoder_dim, 1) for c in embed_dims])
            self.fuse = nn.Sequential(
                nn.Conv2d(decoder_dim * 4, decoder_dim, 3, padding=1, bias=False),
                _gn(decoder_dim),
                nn.GELU(),
                nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1, bias=False),
                _gn(decoder_dim),
                nn.GELU(),
            )
            self.head = nn.Conv2d(decoder_dim, out_ch, 1)
            self.out_ch = int(out_ch)

            with torch.no_grad():
                if self.out_ch == 4:
                    b = torch.tensor([1.1, -2.2, -2.2, -2.8], dtype=torch.float32)
                else:
                    b = torch.zeros((self.out_ch,), dtype=torch.float32)
                self.head.bias.copy_(b)

        def forward(self, x: Any) -> Any:
            feats = self.encoder(x)
            target_hw = feats[0].shape[2:]

            ups = []
            for i, f in enumerate(feats):
                p = self.proj[i](f)
                if p.shape[2:] != target_hw:
                    p = nn.functional.interpolate(p, size=target_hw, mode="bilinear", align_corners=False)
                ups.append(p)

            y = self.fuse(torch.cat(ups, dim=1))
            # stage-1 is at /4 spatial scale.
            y = nn.functional.interpolate(y, scale_factor=4.0, mode="bilinear", align_corners=False)
            return self.head(y)

    return ScorecardTransformerSeg


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------


@dataclass
class TrainConfig:
    labels_dir: Path
    out_weights: Path
    epochs: int = 50
    batch_size: int = 2
    lr: float = 1.5e-4
    weight_decay: float = 5e-4
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
    pretrain_pdf_dpi: int = 260
    pretrain_cache_dir: str = "_pretrain_pdf_cache"


def train_model(cfg: TrainConfig) -> None:
    torch, nn, F, Dataset, DataLoader, random_split = _import_torch()
    _set_seed(int(cfg.seed), torch=torch)

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

            if random.random() < 0.62:
                h, w = image.shape[:2]
                angle = random.uniform(-4.0, 4.0)
                scale = random.uniform(0.965, 1.035)
                mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
                mat[0, 2] += random.uniform(-0.02, 0.02) * w
                mat[1, 2] += random.uniform(-0.02, 0.02) * h
                image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                masks = [
                    cv2.warpAffine(m, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for m in masks
                ]

            if random.random() < 0.50:
                h, w = image.shape[:2]
                src = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float32)
                j = 0.035 * min(h, w)
                dst = src + np.random.uniform(-j, j, size=(4, 2)).astype(np.float32)
                dst[:, 0] = np.clip(dst[:, 0], 0.0, w - 1.0)
                dst[:, 1] = np.clip(dst[:, 1], 0.0, h - 1.0)
                pm = cv2.getPerspectiveTransform(src, dst)
                image = cv2.warpPerspective(image, pm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                masks = [
                    cv2.warpPerspective(m, pm, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                    for m in masks
                ]

            if random.random() < 0.58:
                alpha = random.uniform(0.88, 1.16)
                beta = random.uniform(-20.0, 20.0)
                image = np.clip(image.astype(np.float32) * alpha + beta, 0.0, 255.0).astype(np.uint8)
            if random.random() < 0.28:
                image = cv2.GaussianBlur(image, random.choice([(3, 3), (5, 5)]), 0)
            if random.random() < 0.30:
                noise = np.random.normal(0.0, 5.0, image.shape).astype(np.float32)
                image = np.clip(image.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
            return image, masks

        def __getitem__(self, idx: int) -> Any:
            image, table, vmask, hmask, jmask = _load_label_sample(
                self.files[idx], junction_dilate_px=int(cfg.junction_dilate_px)
            )
            masks = [table, vmask, hmask, jmask]
            image, masks = self._augment(image, masks)
            image, masks = _resize_with_pad(image, masks, size=int(self.train_size))

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
            if random.random() < 0.78:
                angle = random.uniform(-8.5, 8.5)
                scale = random.uniform(0.90, 1.10)
                mat = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), angle, scale)
                mat[0, 2] += random.uniform(-0.04, 0.04) * w
                mat[1, 2] += random.uniform(-0.04, 0.04) * h
                out = cv2.warpAffine(out, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

            if random.random() < 0.72:
                src = np.array([[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]], dtype=np.float32)
                j = 0.06 * min(h, w)
                dst = src + np.random.uniform(-j, j, size=(4, 2)).astype(np.float32)
                dst[:, 0] = np.clip(dst[:, 0], 0.0, w - 1.0)
                dst[:, 1] = np.clip(dst[:, 1], 0.0, h - 1.0)
                pm = cv2.getPerspectiveTransform(src, dst)
                out = cv2.warpPerspective(out, pm, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return out

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
            return np.clip(image.astype(np.float32) * dark[:, :, None], 0.0, 255.0).astype(np.uint8)

        @staticmethod
        def _degrade(image: np.ndarray) -> np.ndarray:
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
            if random.random() < 0.50:
                gains = np.array(
                    [random.uniform(0.80, 1.22), random.uniform(0.80, 1.22), random.uniform(0.80, 1.22)],
                    dtype=np.float32,
                )
                out = np.clip(out.astype(np.float32) * gains.reshape(1, 1, 3), 0.0, 255.0).astype(np.uint8)
            if random.random() < 0.48:
                out = cv2.GaussianBlur(out, random.choice([(3, 3), (5, 5), (7, 7)]), sigmaX=random.uniform(0.8, 2.8))
            if random.random() < 0.30:
                out = PretrainDataset._motion_blur(out, random.choice([7, 9, 11, 13]))
            if random.random() < 0.72:
                sigma = random.uniform(4.0, 18.0)
                noise = np.random.normal(0.0, sigma, out.shape).astype(np.float32)
                out = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
            if random.random() < 0.58:
                out = PretrainDataset._jpeg_roundtrip(out, quality=random.randint(24, 82))
            if random.random() < 0.55:
                out = PretrainDataset._shadow_overlay(out)
            if random.random() < 0.38:
                n_rect = random.randint(1, 3)
                for _ in range(n_rect):
                    rw = max(6, int(random.uniform(0.04, 0.20) * w))
                    rh = max(6, int(random.uniform(0.03, 0.14) * h))
                    x0 = random.randint(0, max(0, w - rw))
                    y0 = random.randint(0, max(0, h - rh))
                    col = int(random.uniform(0.0, 255.0))
                    cv2.rectangle(out, (x0, y0), (x0 + rw, y0 + rh), (col, col, col), -1, cv2.LINE_AA)
            return out

        def __getitem__(self, idx: int) -> Any:
            image = None
            p = self.files[idx]
            for k in range(min(8, len(self.files))):
                pp = self.files[(idx + k) % len(self.files)]
                image = cv2.imread(str(pp), cv2.IMREAD_COLOR)
                if image is not None:
                    p = pp
                    break
            if image is None:
                raise RuntimeError(f"Failed to load pretraining image: {p}")

            image = self._shared_geom(image)
            target = image.copy()
            inp = self._degrade(image.copy())

            inp, _ = _resize_with_pad(inp, [], size=int(self.train_size))
            target, _ = _resize_with_pad(target, [], size=int(self.train_size))

            x = inp.astype(np.float32) / 255.0
            y = target.astype(np.float32) / 255.0
            x_t = torch.from_numpy(np.transpose(x, (2, 0, 1))).float()
            y_t = torch.from_numpy(np.transpose(y, (2, 0, 1))).float()
            return x_t, y_t

    n_total = len(js_files)
    n_val = max(1, int(round(float(cfg.val_ratio) * n_total)))
    n_train = max(1, n_total - n_val)
    if n_train + n_val > n_total:
        n_val = n_total - n_train

    all_ds = TrainDataset(js_files, train_size=int(cfg.train_size), is_train=True)
    train_ds, val_ds = random_split(
        all_ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(int(cfg.seed)),
    )
    val_files = [js_files[i] for i in val_ds.indices]
    val_ds = TrainDataset(val_files, train_size=int(cfg.train_size), is_train=False)

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

    device_name = _select_device(torch, requested="auto")
    device = torch.device(device_name)
    print(f"device={device_name}")
    ModelCls = _build_transformer_model(nn, torch)
    model = ModelCls(in_ch=3, out_ch=4).to(device)
    pulse_every = 20

    # Optional self-supervised pretraining.
    if cfg.pretrain_images_dir is not None and int(cfg.pretrain_epochs) > 0:
        pre_dir = Path(cfg.pretrain_images_dir)
        cache_dir = cfg.out_weights.parent / str(cfg.pretrain_cache_dir)
        pre_files = _prepare_unlabeled_images(
            src_dir=pre_dir,
            cache_dir=cache_dir,
            pdf_dpi=int(cfg.pretrain_pdf_dpi),
            max_images=int(cfg.pretrain_max_images),
        )
        if pre_files:
            pre_bs = int(cfg.pretrain_batch_size) if int(cfg.pretrain_batch_size) > 0 else max(1, int(cfg.batch_size))
            pre_ds = PretrainDataset(pre_files, train_size=int(cfg.train_size))
            pre_loader = DataLoader(
                pre_ds,
                batch_size=max(1, pre_bs),
                shuffle=True,
                num_workers=max(0, int(cfg.num_workers)),
                pin_memory=False,
            )

            pre_model = ModelCls(in_ch=3, out_ch=3).to(device)
            pre_opt = torch.optim.AdamW(
                pre_model.parameters(),
                lr=float(cfg.pretrain_lr),
                weight_decay=float(cfg.pretrain_weight_decay),
            )
            pre_sched = torch.optim.lr_scheduler.CosineAnnealingLR(pre_opt, T_max=max(2, int(cfg.pretrain_epochs)))
            print(
                f"pretrain: samples={len(pre_files)} epochs={int(cfg.pretrain_epochs)} "
                f"bs={pre_bs} lr={float(cfg.pretrain_lr):.3e}"
            )

            for ep in range(1, int(cfg.pretrain_epochs) + 1):
                pre_model.train()
                ep_loss = 0.0
                ep_n = 0
                ep_t0 = time.time()
                for bi, (xb, yb) in enumerate(pre_loader, start=1):
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
                    if bi % pulse_every == 0:
                        avg = ep_loss / float(max(1, ep_n))
                        dt = time.time() - ep_t0
                        print(
                            f"[pulse pretrain e{ep:03d} b{bi:04d}/{len(pre_loader):04d}] "
                            f"avg_loss={avg:.5f} elapsed={dt:.1f}s",
                            flush=True,
                        )
                pre_sched.step()
                print(f"pretrain epoch {ep:03d}/{cfg.pretrain_epochs} loss={ep_loss / float(max(1, ep_n)):.5f}")

            dst_sd = model.state_dict()
            src_sd = pre_model.state_dict()
            copied = 0
            for k, v in src_sd.items():
                if k.startswith("head."):
                    continue
                if k in dst_sd and tuple(dst_sd[k].shape) == tuple(v.shape):
                    dst_sd[k] = v
                    copied += 1
            model.load_state_dict(dst_sd, strict=True)
            print(f"pretrain transfer: copied {copied} tensors into segmentation model")
        else:
            print("pretrain skipped: no images available after image/pdf scan.")

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
        ep_t0 = time.time()
        for bi, (xb, yb) in enumerate(train_loader, start=1):
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
            if bi % pulse_every == 0:
                avg = tr_loss / float(max(1, tr_n))
                dt = time.time() - ep_t0
                print(
                    f"[pulse train e{ep:03d} b{bi:04d}/{len(train_loader):04d}] "
                    f"avg_loss={avg:.5f} elapsed={dt:.1f}s",
                    flush=True,
                )

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
                "model": "scorecard_pvt_seg",
                "out_channels": 4,
                "train_size": int(cfg.train_size),
                "channels": ["table", "v", "h", "junction"],
                "seed": int(cfg.seed),
            }
            torch.save(payload, str(cfg.out_weights))

    print(f"Saved best weights to {cfg.out_weights}")


# -----------------------------------------------------------------------------
# Inference wrappers
# -----------------------------------------------------------------------------


def _load_model(weights: Path, device: str) -> tuple[Any, Any, Any, dict]:
    torch, nn, F, _, _, _ = _import_torch()
    device_name = _select_device(torch, requested=device)
    payload = torch.load(str(weights), map_location=torch.device(device_name))
    if isinstance(payload, dict) and "state_dict" in payload:
        state_dict = payload["state_dict"]
    else:
        state_dict = payload
        payload = {}

    out_channels = int(payload.get("out_channels", 4))
    ModelCls = _build_transformer_model(nn, torch)
    model = ModelCls(in_ch=3, out_ch=out_channels)
    model.load_state_dict(state_dict, strict=True)
    model = model.to(torch.device(device_name))
    model.eval()
    payload["_resolved_device"] = device_name
    return model, torch, F, payload


def _predict_maps(model: Any, torch: Any, F: Any, image_bgr: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    device_name = _select_device(torch, requested=device)
    img = image_bgr.astype(np.float32) / 255.0
    x = torch.from_numpy(np.transpose(img, (2, 0, 1))).float().unsqueeze(0).to(torch.device(device_name))

    _, _, h, w = x.shape
    pad_h = (32 - (h % 32)) % 32
    pad_w = (32 - (w % 32)) % 32
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    probs = probs[:, :, :h, :w][0].cpu().numpy().astype(np.float32)
    return probs[0], probs[1], probs[2], probs[3]


def _run_one_with_model(
    model: Any,
    torch: Any,
    F: Any,
    input_image: Path,
    output_dir: Path,
    infer_cfg: InferConfig,
    device: str,
    pre_cfg: Optional[PreprocessConfig] = None,
    force_output_rotate_180: bool = False,
) -> dict:
    _ensure_dir(output_dir)
    raw = cv2.imread(str(input_image), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"Failed to load image: {input_image}")

    pre_cfg_eff = pre_cfg or PreprocessConfig(ensure_upright=False)
    prep = preprocess_scorecard(raw, pre_cfg_eff)
    image0 = prep.image_bgr
    allow_upright = bool(pre_cfg_eff.ensure_upright)

    decoder = GridDecoder(infer_cfg)

    def _rotate_decoded_180(decoded_in: list[DecodedGrid], h: int, w: int) -> list[DecodedGrid]:
        out: list[DecodedGrid] = []
        for dec in decoded_in:
            x_lines = sorted([(w - 1) - int(x) for x in dec.x_lines])
            y_lines = sorted([(h - 1) - int(y) for y in dec.y_lines])
            bx0, by0, bx1, by1 = dec.bbox
            nb = (
                int((w - 1) - bx1),
                int((h - 1) - by1),
                int((w - 1) - bx0),
                int((h - 1) - by0),
            )
            v_pres = np.asarray(dec.v_presence)
            h_pres = np.asarray(dec.h_presence)
            if v_pres.ndim == 2:
                v_pres = v_pres[::-1, ::-1].copy()
            if h_pres.ndim == 2:
                h_pres = h_pres[::-1, ::-1].copy()
            out.append(
                DecodedGrid(
                    table_id=int(dec.table_id),
                    bbox=nb,
                    x_lines=[int(v) for v in x_lines],
                    y_lines=[int(v) for v in y_lines],
                    v_presence=v_pres,
                    h_presence=h_pres,
                )
            )
        return out

    def _cand_score(
        image_bgr: np.ndarray,
        table_prob: np.ndarray,
        v_prob: np.ndarray,
        h_prob: np.ndarray,
        j_prob: np.ndarray,
        decoded_local: list[Any],
    ) -> float:
        h, w = table_prob.shape[:2]
        area = float(max(1, h * w))
        v_str = float((v_prob >= max(0.52, infer_cfg.line_thresh + 0.06)).mean())
        h_str = float((h_prob >= max(0.52, infer_cfg.line_thresh + 0.06)).mean())
        j_str = float((j_prob >= max(0.45, infer_cfg.junction_thresh + 0.07)).mean())
        s = 7.5 * float(len(decoded_local)) + 26.0 * (v_str + h_str) + 7.0 * j_str
        if decoded_local:
            best = max(
                decoded_local,
                key=lambda d: (d.bbox[2] - d.bbox[0] + 1) * (d.bbox[3] - d.bbox[1] + 1),
            )
            x0, y0, x1, y1 = best.bbox
            ar = float(max(1, (x1 - x0 + 1) * (y1 - y0 + 1))) / area
            if 0.06 <= ar <= 0.92:
                s += 5.0
            s += 0.8 * max(0, len(best.x_lines) - 2)
            s += 0.8 * max(0, len(best.y_lines) - 2)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        ocr = _ocr_readability_score(gray)
        if ocr is not None:
            s += 0.45 * float(ocr)
        return float(s)

    # Orientation chooser: disabled by default. Enable with --upright.
    if not allow_upright:
        cand_imgs = [(0, image0)]
    else:
        cand_imgs = [(0, image0), (180, cv2.rotate(image0, cv2.ROTATE_180))]
    cand_out: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Any]]] = []
    for add_rot, cand_img in cand_imgs:
        tp, vp, hp, jp = _predict_maps(model, torch, F, cand_img, device=device)
        dec = decoder.decode_all(tp, vp, hp, jp, image_bgr=cand_img)
        sc = _cand_score(cand_img, tp, vp, hp, jp, dec)
        cand_out.append((sc, add_rot, cand_img, tp, vp, hp, jp, dec))
    cand_out.sort(key=lambda t: t[0], reverse=True)
    # 180-degree tie/near-tie guard: avoid unnecessary flips only when scores
    # are truly indistinguishable. Keep margin small so a clearly better
    # candidate is not overridden.
    if len(cand_out) >= 2:
        by_rot = {int(t[1]): t for t in cand_out}
        if 0 in by_rot and 180 in by_rot:
            s0 = float(by_rot[0][0])
            s180 = float(by_rot[180][0])
            # Prefer non-rotated result unless 180 has a clearly better score.
            if s180 <= s0 + 0.45:
                cand_out[0] = by_rot[0]
    _, add_rot, image, table_prob, v_prob, h_prob, j_prob, decoded = cand_out[0]
    upright_rot = int((int(prep.upright_rotation_degrees) + int(add_rot)) % 360)

    if allow_upright:
        # Final orientation correction for output artifacts:
        # if OCR readability strongly favors the 180-flipped view, flip outputs.
        g0 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        g1_img = cv2.rotate(image, cv2.ROTATE_180)
        g1 = cv2.cvtColor(g1_img, cv2.COLOR_BGR2GRAY)
        s0 = _ocr_readability_score(g0)
        s1 = _ocr_readability_score(g1)
        if s0 is not None and s1 is not None and float(s1) > float(s0) + 1.10:
            ih, iw = image.shape[:2]
            image = g1_img
            table_prob = cv2.rotate(table_prob, cv2.ROTATE_180)
            v_prob = cv2.rotate(v_prob, cv2.ROTATE_180)
            h_prob = cv2.rotate(h_prob, cv2.ROTATE_180)
            j_prob = cv2.rotate(j_prob, cv2.ROTATE_180)
            decoded = _rotate_decoded_180(decoded, h=ih, w=iw)
            upright_rot = int((upright_rot + 180) % 360)

        # Normalize final artifacts to non-upside-down orientation.
        # This keeps extracted cell matrices upright for downstream OCR.
        if int(upright_rot) == 180:
            ih, iw = image.shape[:2]
            image = cv2.rotate(image, cv2.ROTATE_180)
            table_prob = cv2.rotate(table_prob, cv2.ROTATE_180)
            v_prob = cv2.rotate(v_prob, cv2.ROTATE_180)
            h_prob = cv2.rotate(h_prob, cv2.ROTATE_180)
            j_prob = cv2.rotate(j_prob, cv2.ROTATE_180)
            decoded = _rotate_decoded_180(decoded, h=ih, w=iw)
            upright_rot = 0

    if bool(force_output_rotate_180):
        ih, iw = image.shape[:2]
        image = cv2.rotate(image, cv2.ROTATE_180)
        table_prob = cv2.rotate(table_prob, cv2.ROTATE_180)
        v_prob = cv2.rotate(v_prob, cv2.ROTATE_180)
        h_prob = cv2.rotate(h_prob, cv2.ROTATE_180)
        j_prob = cv2.rotate(j_prob, cv2.ROTATE_180)
        decoded = _rotate_decoded_180(decoded, h=ih, w=iw)
        upright_rot = int((upright_rot + 180) % 360)

    cv2.imwrite(str(output_dir / "debug_preprocessed.png"), image)
    cv2.imwrite(str(output_dir / "debug_table_prob.png"), np.clip(table_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "debug_v_prob.png"), np.clip(v_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "debug_h_prob.png"), np.clip(h_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(output_dir / "debug_junction_prob.png"), np.clip(j_prob * 255.0, 0, 255).astype(np.uint8))

    overlay = image.copy()
    for dec in decoded:
        x0, y0, x1, y1 = dec.bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 0), 2)
        # Draw segments according to presence maps so merged cells are visualized
        # correctly; full-height/width lines are misleading.
        rows = max(0, len(dec.y_lines) - 1)
        cols = max(0, len(dec.x_lines) - 1)
        for i, x in enumerate(dec.x_lines):
            for r in range(rows):
                if dec.v_presence.shape == (rows, len(dec.x_lines)):
                    if i < dec.v_presence.shape[1] and r < dec.v_presence.shape[0] and int(dec.v_presence[r, i]) == 0:
                        continue
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
        "upright_rotation_degrees": int(upright_rot),
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
    force_output_rotate_180: bool = False,
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
        pre_cfg=pre_cfg or PreprocessConfig(),
        force_output_rotate_180=bool(force_output_rotate_180),
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _train_cli(args: argparse.Namespace) -> None:
    pre_dir = Path(args.pretrain_images_dir) if args.pretrain_images_dir else None
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
        pretrain_images_dir=pre_dir,
        pretrain_epochs=int(args.pretrain_epochs),
        pretrain_batch_size=int(args.pretrain_batch_size),
        pretrain_lr=float(args.pretrain_lr),
        pretrain_weight_decay=float(args.pretrain_weight_decay),
        pretrain_max_images=int(args.pretrain_max_images),
        pretrain_pdf_dpi=int(args.pretrain_pdf_dpi),
        pretrain_cache_dir=str(args.pretrain_cache_dir),
    )
    train_model(cfg)


def _infer_cfg_from_args(args: argparse.Namespace) -> InferConfig:
    return InferConfig(
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


def _infer_cli(args: argparse.Namespace) -> None:
    cfg = _infer_cfg_from_args(args)
    pre_cfg = PreprocessConfig(
        ensure_upright=bool(args.upright and not args.no_upright),
        upright_use_osd=False,
        upright_allow_180_without_osd=False,
        upright_180_margin=2.6,
    )
    result = _run_one(
        weights=Path(args.weights),
        input_image=Path(args.input),
        output_dir=Path(args.output_dir),
        infer_cfg=cfg,
        device=str(args.device),
        pre_cfg=pre_cfg,
        force_output_rotate_180=bool(args.force_output_rotate_180),
    )
    print(f"Image: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"tables={result['table_count']} rot={result['upright_rotation_degrees']}")
    for t in result["tables"]:
        print(
            f"  table_{int(t['table_id']):02d}: rows={int(t['rows_base'])} "
            f"cols={int(t['cols_base'])} merged_cells={len(t['cells'])}"
        )


def _batch_cli(args: argparse.Namespace) -> None:
    cfg = _infer_cfg_from_args(args)
    pre_cfg = PreprocessConfig(
        ensure_upright=bool(args.upright and not args.no_upright),
        upright_use_osd=False,
        upright_allow_180_without_osd=False,
        upright_180_margin=2.6,
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
            pre_cfg=pre_cfg,
            force_output_rotate_180=bool(args.force_output_rotate_180),
        )
        table_str = ", ".join([f"t{int(t['table_id'])}:r{int(t['rows_base'])}c{int(t['cols_base'])}" for t in result["tables"]])
        print(f"{p.name}: tables={result['table_count']} [{table_str}] rot={result['upright_rotation_degrees']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Transformer-first scorecard extractor")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Pretrain on unlabeled + fine-tune on labeled masks")
    tr.add_argument("--labels_dir", required=True)
    tr.add_argument("--out_weights", default="checkpoints/scorecard_transformer_seg.pt")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch_size", type=int, default=2)
    tr.add_argument("--lr", type=float, default=1.5e-4)
    tr.add_argument("--weight_decay", type=float, default=5e-4)
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
    tr.add_argument("--pretrain_max_images", type=int, default=0, help="0 means all available")
    tr.add_argument("--pretrain_pdf_dpi", type=int, default=260)
    tr.add_argument("--pretrain_cache_dir", default="_pretrain_pdf_cache")
    tr.set_defaults(func=_train_cli)

    inf = sub.add_parser("infer", help="Infer one image")
    inf.add_argument("--weights", required=True)
    inf.add_argument("--input", required=True)
    inf.add_argument("--output_dir", default="transformer_seg_cells")
    inf.add_argument("--device", default="auto")
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
    inf.add_argument("--no_upright", action="store_true", help=argparse.SUPPRESS)
    inf.add_argument("--force_output_rotate_180", action="store_true")
    inf.set_defaults(func=_infer_cli)

    bt = sub.add_parser("batch", help="Infer all PNG images in a folder")
    bt.add_argument("--weights", required=True)
    bt.add_argument("--input_dir", required=True)
    bt.add_argument("--output_dir", default="transformer_seg_batch_cells")
    bt.add_argument("--device", default="auto")
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
    bt.add_argument("--no_upright", action="store_true", help=argparse.SUPPRESS)
    bt.add_argument("--force_output_rotate_180", action="store_true")
    bt.set_defaults(func=_batch_cli)
    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
