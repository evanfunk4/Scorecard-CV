"""ML-based golf scorecard extraction using Random Forest pixel classifiers.

This module trains 3 pixel-wise classifiers:
- table region mask
- vertical line mask
- horizontal line mask

Inference then applies golf-scorecard priors to decode a stable grid:
- detect up to 2 table regions
- keep only long axis-aligned line evidence
- auto-tune line decoding strictness per table
- reconstruct merged cells and write a matrix index
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import argparse
import json

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from scorecard_preprocessing import preprocess_scorecard, PreprocessConfig


# -----------------------------
# Utilities
# -----------------------------


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_path(base: Path, p: str | Path) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return (base / pp).resolve()


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


def _compute_feature_volume(image_bgr: np.ndarray) -> np.ndarray:
    """Return HxWxF feature tensor for pixel classifier."""

    h, w = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)

    g_blur = cv2.GaussianBlur(clahe, (5, 5), 0)
    local_contrast = cv2.absdiff(clahe, g_blur)

    sobx = cv2.Sobel(clahe, cv2.CV_32F, 1, 0, ksize=3)
    soby = cv2.Sobel(clahe, cv2.CV_32F, 0, 1, ksize=3)
    sobx = np.abs(sobx)
    soby = np.abs(soby)
    lap = np.abs(cv2.Laplacian(clahe, cv2.CV_32F, ksize=3))

    canny = cv2.Canny(clahe, 60, 170)

    hat_k = max(7, int(round(0.008 * min(h, w))) | 1)
    hat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hat_k, hat_k))
    tophat = cv2.morphologyEx(clahe, cv2.MORPH_TOPHAT, hat_kernel)
    blackhat = cv2.morphologyEx(clahe, cv2.MORPH_BLACKHAT, hat_kernel)

    yy, xx = np.indices((h, w), dtype=np.float32)
    x_norm = xx / max(1.0, float(w - 1))
    y_norm = yy / max(1.0, float(h - 1))

    def n01_u8(a: np.ndarray) -> np.ndarray:
        return a.astype(np.float32) / 255.0

    def n01_f32(a: np.ndarray) -> np.ndarray:
        mx = float(np.percentile(a, 99.5))
        if mx <= 1e-6:
            return np.zeros_like(a, dtype=np.float32)
        return np.clip(a / mx, 0.0, 1.0).astype(np.float32)

    feats = np.stack(
        [
            n01_u8(gray),
            n01_u8(clahe),
            n01_u8(local_contrast),
            n01_u8(canny),
            n01_u8(tophat),
            n01_u8(blackhat),
            n01_f32(sobx),
            n01_f32(soby),
            n01_f32(lap),
            x_norm.astype(np.float32),
            y_norm.astype(np.float32),
        ],
        axis=2,
    )
    return feats


# -----------------------------
# Label loading
# -----------------------------


def _load_label_record(json_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rec = json.loads(json_path.read_text(encoding="utf-8"))
    image = cv2.imread(str(_resolve_path(json_path.parent, rec["image"])), cv2.IMREAD_COLOR)
    table = cv2.imread(str(_resolve_path(json_path.parent, rec["table_mask"])), cv2.IMREAD_GRAYSCALE)
    vmask = cv2.imread(str(_resolve_path(json_path.parent, rec["v_mask"])), cv2.IMREAD_GRAYSCALE)
    hmask = cv2.imread(str(_resolve_path(json_path.parent, rec["h_mask"])), cv2.IMREAD_GRAYSCALE)
    if image is None or table is None or vmask is None or hmask is None:
        raise RuntimeError(f"Failed to load sample from {json_path}")
    if image.shape[:2] != table.shape[:2] or image.shape[:2] != vmask.shape[:2] or image.shape[:2] != hmask.shape[:2]:
        raise RuntimeError(f"Shape mismatch in {json_path}")
    return image, (table > 127).astype(np.uint8), (vmask > 127).astype(np.uint8), (hmask > 127).astype(np.uint8)


# -----------------------------
# Model
# -----------------------------


@dataclass
class RFModelConfig:
    n_estimators: int = 220
    max_depth: int = 24
    min_samples_leaf: int = 2
    random_state: int = 7
    n_jobs: int = -1


class RFScorecardModel:
    def __init__(self, cfg: Optional[RFModelConfig] = None):
        self.cfg = cfg or RFModelConfig()
        self.models: dict[str, RandomForestClassifier] = {}

    def fit(self, X: np.ndarray, y_table: np.ndarray, y_v: np.ndarray, y_h: np.ndarray) -> None:
        common = dict(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_leaf=self.cfg.min_samples_leaf,
            random_state=self.cfg.random_state,
            n_jobs=self.cfg.n_jobs,
            class_weight="balanced_subsample",
        )
        m_table = RandomForestClassifier(**common)
        m_v = RandomForestClassifier(**common)
        m_h = RandomForestClassifier(**common)

        print("Training table model...")
        m_table.fit(X, y_table)
        print("Training vertical-line model...")
        m_v.fit(X, y_v)
        print("Training horizontal-line model...")
        m_h.fit(X, y_h)

        self.models = {"table": m_table, "v": m_v, "h": m_h}

    def predict_proba_maps(self, image_bgr: np.ndarray, chunk_size: int = 400_000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("Model is not loaded/trained.")
        feats = _compute_feature_volume(image_bgr)
        h, w, f = feats.shape
        X = feats.reshape(-1, f)

        probs = {}
        for k in ("table", "v", "h"):
            model = self.models[k]
            out = np.zeros((X.shape[0],), dtype=np.float32)
            for s in range(0, X.shape[0], chunk_size):
                e = min(X.shape[0], s + chunk_size)
                p = model.predict_proba(X[s:e])[:, 1]
                out[s:e] = p.astype(np.float32)
            probs[k] = out.reshape(h, w)

        return probs["table"], probs["v"], probs["h"]

    def save(self, path: Path) -> None:
        _ensure_dir(path.parent)
        payload = {
            "cfg": self.cfg.__dict__,
            "models": self.models,
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "RFScorecardModel":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Weights not found: {p}")
        payload = joblib.load(p)
        cfg = RFModelConfig(**payload.get("cfg", {}))
        obj = cls(cfg)
        obj.models = payload["models"]
        return obj


# -----------------------------
# Train data sampler
# -----------------------------


@dataclass
class TrainConfig:
    labels_dir: Path
    out_weights: Path
    sample_pos_per_image: int = 25000
    neg_to_pos_ratio: float = 1.8
    n_estimators: int = 220
    max_depth: int = 24
    min_samples_leaf: int = 2
    random_state: int = 7


def _build_train_matrix(cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(cfg.random_state)
    js_files = sorted(Path(cfg.labels_dir).glob("*.json"))
    if not js_files:
        raise RuntimeError(f"No label JSON files in {cfg.labels_dir}")

    X_parts: list[np.ndarray] = []
    yt_parts: list[np.ndarray] = []
    yv_parts: list[np.ndarray] = []
    yh_parts: list[np.ndarray] = []

    for js in js_files:
        try:
            image, table, vmask, hmask = _load_label_record(js)
        except Exception:
            # Ignore non-record JSON artifacts (for example queue files).
            continue
        feats = _compute_feature_volume(image).reshape(-1, 11)
        yt = table.reshape(-1)
        yv = vmask.reshape(-1)
        yh = hmask.reshape(-1)
        pos = (yt > 0) | (yv > 0) | (yh > 0)
        pos_idx = np.where(pos)[0]
        neg_idx = np.where(~pos)[0]
        if pos_idx.size == 0:
            continue

        n_pos = min(int(cfg.sample_pos_per_image), int(pos_idx.size))
        pos_sel = rng.choice(pos_idx, size=n_pos, replace=False)
        n_neg = min(int(round(cfg.neg_to_pos_ratio * n_pos)), int(neg_idx.size))
        neg_sel = rng.choice(neg_idx, size=n_neg, replace=False) if n_neg > 0 else np.empty((0,), dtype=np.int64)

        sel = np.concatenate([pos_sel, neg_sel], axis=0)
        rng.shuffle(sel)

        X_parts.append(feats[sel].astype(np.float32))
        yt_parts.append((yt[sel] > 0).astype(np.uint8))
        yv_parts.append((yv[sel] > 0).astype(np.uint8))
        yh_parts.append((yh[sel] > 0).astype(np.uint8))

        print(f"sampled {js.name}: pos={n_pos} neg={n_neg}")

    if not X_parts:
        raise RuntimeError("No training samples were collected.")

    X = np.concatenate(X_parts, axis=0)
    y_table = np.concatenate(yt_parts, axis=0)
    y_v = np.concatenate(yv_parts, axis=0)
    y_h = np.concatenate(yh_parts, axis=0)
    print(f"train matrix: X={X.shape}, table_pos={y_table.mean():.4f}, v_pos={y_v.mean():.4f}, h_pos={y_h.mean():.4f}")
    return X, y_table, y_v, y_h


def train_model(cfg: TrainConfig) -> None:
    X, y_table, y_v, y_h = _build_train_matrix(cfg)
    model = RFScorecardModel(
        RFModelConfig(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            random_state=cfg.random_state,
            n_jobs=-1,
        )
    )
    model.fit(X, y_table, y_v, y_h)
    model.save(cfg.out_weights)
    print(f"Saved weights to {cfg.out_weights}")


# -----------------------------
# Decoder
# -----------------------------


@dataclass
class InferConfig:
    table_thresh: float = 0.50
    line_thresh: float = 0.50
    min_line_cov: float = 0.20
    min_gap_px: int = 8
    peak_rel_thresh: float = 0.16
    line_band_px: int = 2
    cell_inset_px: int = 2
    min_cell_w: int = 10
    min_cell_h: int = 10
    max_tables: int = 2
    min_table_area_ratio: float = 0.010
    min_rows: int = 4
    max_rows: int = 22
    min_cols: int = 4
    max_cols: int = 28
    auto_tune_decode: bool = True


@dataclass
class DecodedGrid:
    table_id: int
    bbox: tuple[int, int, int, int]  # x0,y0,x1,y1
    x_lines: list[int]
    y_lines: list[int]
    v_presence: np.ndarray
    h_presence: np.ndarray


class GridDecoder:
    def __init__(self, cfg: Optional[InferConfig] = None):
        self.cfg = cfg or InferConfig()

    def decode_all(self, table_prob: np.ndarray, v_prob: np.ndarray, h_prob: np.ndarray) -> list[DecodedGrid]:
        bboxes = self._decode_table_bboxes(table_prob, v_prob, h_prob)
        out: list[DecodedGrid] = []
        for tid, box in enumerate(bboxes):
            x_lines, y_lines, line_thr = self._decode_best_lines(v_prob, h_prob, box)
            v_bin = (v_prob >= line_thr).astype(np.uint8) * 255
            h_bin = (h_prob >= line_thr).astype(np.uint8) * 255
            v_presence = self._vertical_presence(v_bin, x_lines, y_lines)
            h_presence = self._horizontal_presence(h_bin, x_lines, y_lines)
            out.append(
                DecodedGrid(
                    table_id=tid,
                    bbox=box,
                    x_lines=x_lines,
                    y_lines=y_lines,
                    v_presence=v_presence,
                    h_presence=h_presence,
                )
            )
        return out

    def _decode_table_bboxes(self, table_prob: np.ndarray, v_prob: np.ndarray, h_prob: np.ndarray) -> list[tuple[int, int, int, int]]:
        h, w = table_prob.shape[:2]
        t = (table_prob >= self.cfg.table_thresh).astype(np.uint8)
        lv = (v_prob >= max(0.30, self.cfg.line_thresh * 0.75)).astype(np.uint8)
        lh = (h_prob >= max(0.30, self.cfg.line_thresh * 0.75)).astype(np.uint8)
        mask = ((t | lv | lh) * 255).astype(np.uint8)

        k = max(5, int(round(0.01 * max(h, w))) | 1)
        kk = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kk, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

        n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
        min_area = float(self.cfg.min_table_area_ratio) * float(h * w)

        cands: list[tuple[float, tuple[int, int, int, int]]] = []
        for i in range(1, n):
            x, y, bw, bh, area = stats[i]
            if area < min_area:
                continue
            if bw < max(60, int(0.14 * w)):
                continue
            if bh < max(60, int(0.08 * h)):
                continue

            x0 = int(max(0, x - round(0.01 * w)))
            y0 = int(max(0, y - round(0.01 * h)))
            x1 = int(min(w - 1, x + bw - 1 + round(0.01 * w)))
            y1 = int(min(h - 1, y + bh - 1 + round(0.01 * h)))

            crop_v = v_prob[y0 : y1 + 1, x0 : x1 + 1]
            crop_h = h_prob[y0 : y1 + 1, x0 : x1 + 1]
            line_energy = float(crop_v.mean() + crop_h.mean())
            score = float(area) * (1.0 + 0.8 * line_energy)
            cands.append((score, (x0, y0, x1, y1)))

        if not cands:
            return [(0, 0, w - 1, h - 1)]

        cands.sort(key=lambda t: t[0], reverse=True)
        chosen: list[tuple[int, int, int, int]] = []
        for _, box in cands:
            if any(_bbox_iou_xyxy(box, c) > 0.45 for c in chosen):
                continue
            chosen.append(box)
            if len(chosen) >= int(self.cfg.max_tables):
                break

        if not chosen:
            chosen = [cands[0][1]]

        chosen.sort(key=lambda b: (b[0], b[1]))
        return chosen

    def _axis_open_mask(self, crop_prob: np.ndarray, axis: str, t_hi: float, t_lo: float) -> np.ndarray:
        h, w = crop_prob.shape[:2]
        hi = (crop_prob >= t_hi).astype(np.uint8) * 255
        lo = (crop_prob >= t_lo).astype(np.uint8) * 255

        if axis == "x":
            # Keep long vertical structures.
            k1 = max(7, int(round(0.07 * h)) | 1)
            k2 = max(11, int(round(0.12 * h)) | 1)
            o1 = cv2.morphologyEx(hi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k1)), iterations=1)
            o2 = cv2.morphologyEx(lo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, k2)), iterations=1)
            m = cv2.bitwise_or(o1, o2)
        else:
            # Keep long horizontal structures.
            k1 = max(7, int(round(0.07 * w)) | 1)
            k2 = max(11, int(round(0.12 * w)) | 1)
            o1 = cv2.morphologyEx(hi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k1, 1)), iterations=1)
            o2 = cv2.morphologyEx(lo, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k2, 1)), iterations=1)
            m = cv2.bitwise_or(o1, o2)

        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
        return m

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
                if span < max(12, int(0.24 * h)):
                    continue
                if thick > max(10, int(0.12 * w)):
                    continue
                pos = int(round(cx))
                score = float(span) * (0.6 + float(area) / float(max(1, bw * bh)))
                out.append((pos, score))
            else:
                span = bw
                thick = bh
                if span < max(12, int(0.24 * w)):
                    continue
                if thick > max(10, int(0.12 * h)):
                    continue
                pos = int(round(cy))
                score = float(span) * (0.6 + float(area) / float(max(1, bw * bh)))
                out.append((pos, score))

        return out

    def _cluster_candidates(self, cands: list[tuple[int, float]], cluster_gap: int) -> list[tuple[int, float]]:
        if not cands:
            return []
        vals = sorted(cands, key=lambda t: t[0])
        groups: list[list[tuple[int, float]]] = []
        for p, s in vals:
            if not groups:
                groups.append([(p, s)])
                continue
            if abs(p - groups[-1][-1][0]) <= cluster_gap:
                groups[-1].append((p, s))
            else:
                groups.append([(p, s)])

        out: list[tuple[int, float]] = []
        for g in groups:
            ws = np.array([max(1e-3, x[1]) for x in g], dtype=np.float32)
            ps = np.array([x[0] for x in g], dtype=np.float32)
            p = int(round(float(np.sum(ps * ws) / np.sum(ws))))
            s = float(np.max(ws))
            out.append((p, s))
        return out

    def _regularize_axis(
        self,
        cands: list[tuple[int, float]],
        proj: np.ndarray,
        low: int,
        high: int,
        min_gap: int,
        max_lines: int,
    ) -> list[int]:
        if low >= high:
            return [low, high]

        items = [(int(np.clip(p, low, high)), float(s)) for p, s in cands]
        items.append((low, 1e9))
        items.append((high, 1e9))

        cl = self._cluster_candidates(items, max(1, min_gap // 2))
        cl = sorted(cl, key=lambda t: t[0])

        # Deduplicate near lines by keeping stronger one.
        dedup: list[tuple[int, float]] = []
        for p, s in cl:
            if not dedup:
                dedup.append((p, s))
                continue
            q, qs = dedup[-1]
            if p - q < min_gap and p not in (low, high) and q not in (low, high):
                if s > qs:
                    dedup[-1] = (p, s)
            else:
                dedup.append((p, s))

        vals = [x[0] for x in dedup]
        vals = sorted(set(vals))
        if low not in vals:
            vals.insert(0, low)
        if high not in vals:
            vals.append(high)

        # Fill very large gaps from spacing prior.
        if len(vals) >= 3:
            gaps = np.diff(np.array(vals, dtype=np.int32))
            base = float(np.median(gaps)) if gaps.size else float(min_gap)
            base = max(float(min_gap), base)
            inserts: list[int] = []
            for i, g in enumerate(gaps):
                if g <= 2.3 * base:
                    continue
                n_new = int(round(float(g) / base)) - 1
                n_new = max(1, min(3, n_new))
                a = vals[i]
                for k in range(n_new):
                    v = int(round(a + (k + 1) * g / float(n_new + 1)))
                    if low < v < high:
                        inserts.append(v)
            if inserts:
                vals = sorted(set(vals + inserts))
                vals = _prune_near(vals, max(1, int(round(0.70 * min_gap))))

        # Hard cap line count by score/projection strength.
        if len(vals) > max_lines:
            keep_n = max(2, int(max_lines))
            interior = [v for v in vals if v not in (low, high)]
            scored = []
            for v in interior:
                lo = max(low, v - 1)
                hi = min(high, v + 1)
                sv = float(np.mean(proj[lo : hi + 1])) if 0 <= lo <= hi < proj.size else 0.0
                scored.append((sv, v))
            scored.sort(key=lambda t: t[0], reverse=True)
            keep_inner = sorted([v for _, v in scored[: max(0, keep_n - 2)]])
            vals = [low] + keep_inner + [high]

        vals = _prune_near(vals, max(1, int(round(0.65 * min_gap))))
        if len(vals) < 2:
            vals = [low, high]
        if vals[0] != low:
            vals[0] = low
        if vals[-1] != high:
            vals[-1] = high
        return sorted(vals)

    def _decode_axis_lines(self, crop_prob: np.ndarray, axis: str, line_thr: float, cov_thr: float, peak_thr: float, min_gap: int, max_lines: int) -> list[int]:
        h, w = crop_prob.shape[:2]
        if h < 2 or w < 2:
            return [0, max(1, (w - 1) if axis == "x" else (h - 1))]

        low_thr = max(0.20, line_thr - 0.12)
        axis_mask = self._axis_open_mask(crop_prob, axis=axis, t_hi=line_thr, t_lo=low_thr)

        if axis == "x":
            proj_prob = crop_prob.mean(axis=0)
            proj_mask = (axis_mask > 0).mean(axis=0).astype(np.float32)
            n = proj_prob.size
            window = max(7, int(round(0.02 * n)) | 1)
            proj = 0.68 * _smooth_1d(proj_prob.astype(np.float32), window) + 0.32 * _smooth_1d(proj_mask, window)

            dyn_gap = max(min_gap, int(round(0.028 * w)))
            peaks = _projection_peaks(proj, rel_thresh=peak_thr, min_gap=dyn_gap)
            cands = self._axis_component_candidates(axis_mask, axis="x")

            for p in peaks:
                band = max(1, int(self.cfg.line_band_px))
                x0 = max(0, p - band)
                x1 = min(w - 1, p + band)
                sl = axis_mask[:, x0 : x1 + 1]
                cov = float((sl > 0).mean()) if sl.size else 0.0
                if cov >= max(0.08, cov_thr * 0.55):
                    s = float(proj[p]) * (0.6 + cov)
                    cands.append((int(p), s))

            vals = self._regularize_axis(
                cands,
                proj=proj,
                low=0,
                high=w - 1,
                min_gap=dyn_gap,
                max_lines=max_lines,
            )
            return vals

        proj_prob = crop_prob.mean(axis=1)
        proj_mask = (axis_mask > 0).mean(axis=1).astype(np.float32)
        n = proj_prob.size
        window = max(7, int(round(0.02 * n)) | 1)
        proj = 0.68 * _smooth_1d(proj_prob.astype(np.float32), window) + 0.32 * _smooth_1d(proj_mask, window)

        dyn_gap = max(min_gap, int(round(0.028 * h)))
        peaks = _projection_peaks(proj, rel_thresh=peak_thr, min_gap=dyn_gap)
        cands = self._axis_component_candidates(axis_mask, axis="y")

        for p in peaks:
            band = max(1, int(self.cfg.line_band_px))
            y0 = max(0, p - band)
            y1 = min(h - 1, p + band)
            sl = axis_mask[y0 : y1 + 1, :]
            cov = float((sl > 0).mean()) if sl.size else 0.0
            if cov >= max(0.08, cov_thr * 0.55):
                s = float(proj[p]) * (0.6 + cov)
                cands.append((int(p), s))

        vals = self._regularize_axis(
            cands,
            proj=proj,
            low=0,
            high=h - 1,
            min_gap=dyn_gap,
            max_lines=max_lines,
        )
        return vals

    def _score_grid(self, x_lines: list[int], y_lines: list[int], v_prob_crop: np.ndarray, h_prob_crop: np.ndarray, line_thr: float) -> float:
        cols = max(0, len(x_lines) - 1)
        rows = max(0, len(y_lines) - 1)
        if rows <= 0 or cols <= 0:
            return -1e9

        score = 0.0

        # Row/col plausibility priors for scorecards.
        if self.cfg.min_rows <= rows <= self.cfg.max_rows:
            score += 4.0
        else:
            score -= 0.8 * abs(rows - np.clip(rows, self.cfg.min_rows, self.cfg.max_rows))

        if self.cfg.min_cols <= cols <= self.cfg.max_cols:
            score += 4.0
        else:
            score -= 0.8 * abs(cols - np.clip(cols, self.cfg.min_cols, self.cfg.max_cols))

        # Soft target counts for scorecards.
        score -= 0.35 * abs(rows - 12)
        score -= 0.22 * abs(cols - 16)

        # Penalize cap-hugging solutions that usually indicate over-splitting.
        if rows >= self.cfg.max_rows - 1:
            score -= 2.5
        if cols >= self.cfg.max_cols - 1:
            score -= 2.5

        ratio = cols / float(max(1, rows))
        if 0.35 <= ratio <= 4.2:
            score += 2.0
        else:
            score -= 2.0

        # Penalize highly irregular spacing.
        if len(x_lines) >= 4:
            gx = np.diff(np.array(x_lines, dtype=np.float32))
            m = float(np.mean(gx))
            if m > 1e-6:
                score -= 2.4 * float(np.std(gx) / m)
        if len(y_lines) >= 4:
            gy = np.diff(np.array(y_lines, dtype=np.float32))
            m = float(np.mean(gy))
            if m > 1e-6:
                score -= 2.4 * float(np.std(gy) / m)

        # Reward line evidence around decoded lines.
        vbin = (v_prob_crop >= line_thr).astype(np.uint8) * 255
        hbin = (h_prob_crop >= line_thr).astype(np.uint8) * 255
        vp = self._vertical_presence(vbin, x_lines, y_lines)
        hp = self._horizontal_presence(hbin, x_lines, y_lines)
        if vp.size:
            score += 2.0 * float(vp.mean())
        if hp.size:
            score += 2.0 * float(hp.mean())

        # Degenerate solutions are penalized.
        if rows < 3 or cols < 3:
            score -= 5.0

        return score

    def _decode_best_lines(self, v_prob: np.ndarray, h_prob: np.ndarray, bbox_xyxy: tuple[int, int, int, int]) -> tuple[list[int], list[int], float]:
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

        v_crop = v_prob[y0 : y1 + 1, x0 : x1 + 1]
        h_crop = h_prob[y0 : y1 + 1, x0 : x1 + 1]

        base = (float(self.cfg.line_thresh), float(self.cfg.min_line_cov), float(self.cfg.peak_rel_thresh), int(self.cfg.min_gap_px))
        param_set = [base]
        if self.cfg.auto_tune_decode:
            param_set.extend(
                [
                    (max(0.30, base[0] - 0.10), max(0.10, base[1] - 0.08), max(0.08, base[2] - 0.06), max(6, int(round(base[3] * 0.8)))),
                    (max(0.30, base[0] - 0.05), max(0.10, base[1] - 0.05), max(0.08, base[2] - 0.03), max(6, int(round(base[3] * 0.9)))),
                    (min(0.80, base[0] + 0.08), min(0.55, base[1] + 0.08), min(0.42, base[2] + 0.07), int(round(base[3] * 1.2))),
                    (min(0.80, base[0] + 0.12), min(0.60, base[1] + 0.12), min(0.48, base[2] + 0.12), int(round(base[3] * 1.35))),
                ]
            )

        best_score = -1e18
        best_x: list[int] = [0, max(1, v_crop.shape[1] - 1)]
        best_y: list[int] = [0, max(1, v_crop.shape[0] - 1)]
        best_thr = base[0]

        for lthr, cov_thr, peak_thr, gap in param_set:
            x_local = self._decode_axis_lines(
                v_crop,
                axis="x",
                line_thr=lthr,
                cov_thr=cov_thr,
                peak_thr=peak_thr,
                min_gap=gap,
                max_lines=max(2, self.cfg.max_cols + 1),
            )
            y_local = self._decode_axis_lines(
                h_crop,
                axis="y",
                line_thr=lthr,
                cov_thr=cov_thr,
                peak_thr=peak_thr,
                min_gap=gap,
                max_lines=max(2, self.cfg.max_rows + 1),
            )
            s = self._score_grid(x_local, y_local, v_crop, h_crop, line_thr=lthr)
            if s > best_score:
                best_score = s
                best_x = x_local
                best_y = y_local
                best_thr = lthr

        # Map local coords to absolute.
        x_abs = [x0 + int(v) for v in best_x]
        y_abs = [y0 + int(v) for v in best_y]

        # Keep bounds exact.
        if x_abs:
            x_abs[0] = x0
            x_abs[-1] = x1
        else:
            x_abs = [x0, x1]
        if y_abs:
            y_abs[0] = y0
            y_abs[-1] = y1
        else:
            y_abs = [y0, y1]

        x_abs = _prune_near(sorted(set(x_abs)), max(2, int(round(0.45 * max(1, self.cfg.min_gap_px)))))
        y_abs = _prune_near(sorted(set(y_abs)), max(2, int(round(0.45 * max(1, self.cfg.min_gap_px)))))

        if len(x_abs) < 2:
            x_abs = [x0, x1]
        if len(y_abs) < 2:
            y_abs = [y0, y1]

        return x_abs, y_abs, best_thr

    def _vertical_presence(self, v_bin: np.ndarray, x_lines: list[int], y_lines: list[int]) -> np.ndarray:
        rows = max(0, len(y_lines) - 1)
        nvl = len(x_lines)
        out = np.zeros((rows, nvl), dtype=np.uint8)
        b = max(1, int(self.cfg.line_band_px))
        for r in range(rows):
            y0, y1 = y_lines[r], y_lines[r + 1]
            if y1 <= y0:
                continue
            for i, x in enumerate(x_lines):
                xx0 = max(0, x - b)
                xx1 = min(v_bin.shape[1] - 1, x + b)
                sl = v_bin[y0:y1, xx0 : xx1 + 1]
                if sl.size == 0:
                    continue
                if float((sl > 0).mean()) >= self.cfg.min_line_cov:
                    out[r, i] = 1
        return out

    def _horizontal_presence(self, h_bin: np.ndarray, x_lines: list[int], y_lines: list[int]) -> np.ndarray:
        nyl = len(y_lines)
        cols = max(0, len(x_lines) - 1)
        out = np.zeros((nyl, cols), dtype=np.uint8)
        b = max(1, int(self.cfg.line_band_px))
        for j, y in enumerate(y_lines):
            yy0 = max(0, y - b)
            yy1 = min(h_bin.shape[0] - 1, y + b)
            for c in range(cols):
                x0, x1 = x_lines[c], x_lines[c + 1]
                if x1 <= x0:
                    continue
                sl = h_bin[yy0 : yy1 + 1, x0:x1]
                if sl.size == 0:
                    continue
                if float((sl > 0).mean()) >= self.cfg.min_line_cov:
                    out[j, c] = 1
        return out


# -----------------------------
# Cell extraction with merges
# -----------------------------


def extract_cells_from_decoded(image_bgr: np.ndarray, dec: DecodedGrid, output_dir: Path, cfg: InferConfig) -> dict:
    _ensure_dir(output_dir)
    x_lines = dec.x_lines
    y_lines = dec.y_lines
    rows = max(0, len(y_lines) - 1)
    cols = max(0, len(x_lines) - 1)

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

    for r in range(rows):
        for c in range(cols - 1):
            if dec.v_presence[r, c + 1] == 0:
                union(idx(r, c), idx(r, c + 1))

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

    img_h, img_w = image_bgr.shape[:2]
    inset = max(0, int(cfg.cell_inset_px))
    cells = []
    for cid, (r0, c0, r1, c1) in enumerate(merged):
        x0 = x_lines[c0]
        x1 = x_lines[c1 + 1]
        y0 = y_lines[r0]
        y1 = y_lines[r1 + 1]
        cx0 = max(0, min(img_w, x0 + inset))
        cx1 = max(0, min(img_w, x1 - inset))
        cy0 = max(0, min(img_h, y0 + inset))
        cy1 = max(0, min(img_h, y1 - inset))

        path = None
        if cx1 > cx0 and cy1 > cy0 and (cx1 - cx0) >= cfg.min_cell_w and (cy1 - cy0) >= cfg.min_cell_h:
            cell = image_bgr[cy0:cy1, cx0:cx1]
            if cell.size > 0:
                p = output_dir / f"cell_{cid:04d}_r{r0:02d}_c{c0:02d}_rs{(r1-r0+1):02d}_cs{(c1-c0+1):02d}.png"
                cv2.imwrite(str(p), cell)
                path = p.name

        cells.append(
            {
                "cell_id": cid,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "rowspan": r1 - r0 + 1,
                "colspan": c1 - c0 + 1,
                "path": path,
            }
        )

    inv = {}
    for c in cells:
        for r in range(c["r0"], c["r1"] + 1):
            for cc in range(c["c0"], c["c1"] + 1):
                inv[(r, cc)] = c

    mat: list[list[dict]] = [[{} for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            cell = inv[(r, c)]
            anchor = (r == cell["r0"] and c == cell["c0"])
            mat[r][c] = {
                "cell_id": int(cell["cell_id"]),
                "anchor": bool(anchor),
                "rowspan": int(cell["rowspan"]),
                "colspan": int(cell["colspan"]),
                "path": cell["path"] if anchor else None,
            }

    out = {
        "table_id": int(dec.table_id),
        "bbox_xyxy": [int(v) for v in dec.bbox],
        "x_lines": [int(v) for v in dec.x_lines],
        "y_lines": [int(v) for v in dec.y_lines],
        "rows_base": rows,
        "cols_base": cols,
        "cells": cells,
        "matrix": mat,
    }
    (output_dir / "matrix_index.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# -----------------------------
# CLI handlers
# -----------------------------


def _train_cli(args: argparse.Namespace) -> None:
    cfg = TrainConfig(
        labels_dir=Path(args.labels_dir),
        out_weights=Path(args.out_weights),
        sample_pos_per_image=args.sample_pos_per_image,
        neg_to_pos_ratio=args.neg_to_pos_ratio,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
    )
    train_model(cfg)


def _run_one(model: RFScorecardModel, img_path: Path, out_dir: Path, infer_cfg: InferConfig) -> dict:
    _ensure_dir(out_dir)
    raw = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"Failed to load image: {img_path}")

    prep = preprocess_scorecard(raw, PreprocessConfig())
    image = prep.image_bgr
    cv2.imwrite(str(out_dir / "debug_preprocessed.png"), image)

    table_prob, v_prob, h_prob = model.predict_proba_maps(image)
    cv2.imwrite(str(out_dir / "debug_table_prob.png"), np.clip(table_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "debug_v_prob.png"), np.clip(v_prob * 255.0, 0, 255).astype(np.uint8))
    cv2.imwrite(str(out_dir / "debug_h_prob.png"), np.clip(h_prob * 255.0, 0, 255).astype(np.uint8))

    decoder = GridDecoder(infer_cfg)
    decoded = decoder.decode_all(table_prob, v_prob, h_prob)

    line_mask_dbg = (((v_prob >= infer_cfg.line_thresh) | (h_prob >= infer_cfg.line_thresh)).astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "debug_line_mask.png"), line_mask_dbg)

    overlay = image.copy()
    for dec in decoded:
        x0, y0, x1, y1 = dec.bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 0), 2)
        for x in dec.x_lines:
            cv2.line(overlay, (x, y0), (x, y1), (255, 80, 80), 1)
        for y in dec.y_lines:
            cv2.line(overlay, (x0, y), (x1, y), (80, 80, 255), 1)
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
    cv2.imwrite(str(out_dir / "debug_grid_overlay.png"), overlay)

    table_results = []
    for dec in decoded:
        tdir = out_dir / f"table_{dec.table_id:02d}" / "cells"
        matrix = extract_cells_from_decoded(image, dec, tdir, infer_cfg)
        table_results.append(matrix)

    payload = {
        "image": str(img_path),
        "upright_rotation_degrees": int(prep.upright_rotation_degrees),
        "table_count": len(table_results),
        "tables": table_results,
    }
    (out_dir / "image_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _infer_cli(args: argparse.Namespace) -> None:
    model = RFScorecardModel.load(args.weights)
    cfg = InferConfig(
        table_thresh=args.table_thresh,
        line_thresh=args.line_thresh,
        min_line_cov=args.min_line_cov,
        min_gap_px=args.min_gap_px,
        peak_rel_thresh=args.peak_rel_thresh,
        line_band_px=args.line_band_px,
        cell_inset_px=args.cell_inset,
        min_cell_w=args.min_cell_w,
        min_cell_h=args.min_cell_h,
        max_tables=args.max_tables,
        min_table_area_ratio=args.min_table_area_ratio,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_cols=args.min_cols,
        max_cols=args.max_cols,
        auto_tune_decode=not args.no_auto_tune,
    )
    result = _run_one(model, Path(args.input), Path(args.output_dir), cfg)
    print(f"Image: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"tables={result['table_count']} rot={result['upright_rotation_degrees']}")
    for t in result["tables"]:
        print(
            f"  table_{int(t['table_id']):02d}: rows={int(t['rows_base'])} cols={int(t['cols_base'])} merged_cells={len(t['cells'])}"
        )


def _batch_cli(args: argparse.Namespace) -> None:
    model = RFScorecardModel.load(args.weights)
    cfg = InferConfig(
        table_thresh=args.table_thresh,
        line_thresh=args.line_thresh,
        min_line_cov=args.min_line_cov,
        min_gap_px=args.min_gap_px,
        peak_rel_thresh=args.peak_rel_thresh,
        line_band_px=args.line_band_px,
        cell_inset_px=args.cell_inset,
        min_cell_w=args.min_cell_w,
        min_cell_h=args.min_cell_h,
        max_tables=args.max_tables,
        min_table_area_ratio=args.min_table_area_ratio,
        min_rows=args.min_rows,
        max_rows=args.max_rows,
        min_cols=args.min_cols,
        max_cols=args.max_cols,
        auto_tune_decode=not args.no_auto_tune,
    )

    inp_dir = Path(args.input_dir)
    out_root = Path(args.output_dir)
    _ensure_dir(out_root)

    imgs = sorted(inp_dir.glob("*.png"))
    if not imgs:
        print(f"No PNG images found in {inp_dir}")
        return

    for p in imgs:
        out = out_root / p.stem
        result = _run_one(model, p, out, cfg)
        table_str = ", ".join(
            [
                f"t{int(t['table_id'])}:r{int(t['rows_base'])}c{int(t['cols_base'])}"
                for t in result["tables"]
            ]
        )
        print(
            f"{p.name}: tables={result['table_count']} [{table_str}] rot={result['upright_rotation_degrees']}"
        )


# -----------------------------
# CLI
# -----------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ML scorecard extractor (Random Forest + structural priors)")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Train RF model from label masks")
    tr.add_argument("--labels_dir", required=True)
    tr.add_argument("--out_weights", default="checkpoints/scorecard_grid_rf.joblib")
    tr.add_argument("--sample_pos_per_image", type=int, default=25000)
    tr.add_argument("--neg_to_pos_ratio", type=float, default=1.8)
    tr.add_argument("--n_estimators", type=int, default=220)
    tr.add_argument("--max_depth", type=int, default=24)
    tr.add_argument("--min_samples_leaf", type=int, default=2)
    tr.add_argument("--seed", type=int, default=7)
    tr.set_defaults(func=_train_cli)

    inf = sub.add_parser("infer", help="Infer one image")
    inf.add_argument("--weights", required=True)
    inf.add_argument("--input", required=True)
    inf.add_argument("--output_dir", default="ml_cells")
    inf.add_argument("--table_thresh", type=float, default=0.50)
    inf.add_argument("--line_thresh", type=float, default=0.50)
    inf.add_argument("--min_line_cov", type=float, default=0.20)
    inf.add_argument("--min_gap_px", type=int, default=8)
    inf.add_argument("--peak_rel_thresh", type=float, default=0.16)
    inf.add_argument("--line_band_px", type=int, default=2)
    inf.add_argument("--cell_inset", type=int, default=2)
    inf.add_argument("--min_cell_w", type=int, default=10)
    inf.add_argument("--min_cell_h", type=int, default=10)
    inf.add_argument("--max_tables", type=int, default=2)
    inf.add_argument("--min_table_area_ratio", type=float, default=0.010)
    inf.add_argument("--min_rows", type=int, default=4)
    inf.add_argument("--max_rows", type=int, default=22)
    inf.add_argument("--min_cols", type=int, default=4)
    inf.add_argument("--max_cols", type=int, default=28)
    inf.add_argument("--no_auto_tune", action="store_true")
    inf.set_defaults(func=_infer_cli)

    bt = sub.add_parser("batch", help="Infer all PNG images in a folder")
    bt.add_argument("--weights", required=True)
    bt.add_argument("--input_dir", required=True)
    bt.add_argument("--output_dir", default="ml_batch_cells")
    bt.add_argument("--table_thresh", type=float, default=0.50)
    bt.add_argument("--line_thresh", type=float, default=0.50)
    bt.add_argument("--min_line_cov", type=float, default=0.20)
    bt.add_argument("--min_gap_px", type=int, default=8)
    bt.add_argument("--peak_rel_thresh", type=float, default=0.16)
    bt.add_argument("--line_band_px", type=int, default=2)
    bt.add_argument("--cell_inset", type=int, default=2)
    bt.add_argument("--min_cell_w", type=int, default=10)
    bt.add_argument("--min_cell_h", type=int, default=10)
    bt.add_argument("--max_tables", type=int, default=2)
    bt.add_argument("--min_table_area_ratio", type=float, default=0.010)
    bt.add_argument("--min_rows", type=int, default=4)
    bt.add_argument("--max_rows", type=int, default=22)
    bt.add_argument("--min_cols", type=int, default=4)
    bt.add_argument("--max_cols", type=int, default=28)
    bt.add_argument("--no_auto_tune", action="store_true")
    bt.set_defaults(func=_batch_cli)

    return p


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)
