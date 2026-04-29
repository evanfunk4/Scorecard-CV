#!/usr/bin/env python3
"""
Deterministic refinement for row-separator labels.

Goal:
- Keep topology stable (table count + line counts unchanged)
- Snap existing vertical/horizontal polylines to stronger local line evidence
- Leave first N files untouched (trusted labels)

This is designed as an annotation-assist pre-pass, not a final automatic labeler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return int(default)


def _clip(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _collect_jsons(labels_dir: Path) -> list[Path]:
    return sorted(p for p in labels_dir.glob("*.json") if p.is_file())


def _as_polyline(raw: Any) -> list[list[int]]:
    out: list[list[int]] = []
    if isinstance(raw, list):
        for p in raw:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append([_safe_int(p[0]), _safe_int(p[1])])
    return out


def _line_center_x(poly: list[list[int]]) -> float:
    if not poly:
        return 0.0
    return float(sum(p[0] for p in poly)) / float(len(poly))


def _line_center_y(poly: list[list[int]]) -> float:
    if not poly:
        return 0.0
    return float(sum(p[1] for p in poly)) / float(len(poly))


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


def _build_evidence_maps(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 50, 50)

    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    ax = np.abs(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3))
    ay = np.abs(cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
    bx = np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3))
    by = np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3))

    # Morphological long-line hints for text suppression.
    bw = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        35,
        9,
    )
    v_len = max(15, int(round(h * 0.05)))
    h_len = max(15, int(round(w * 0.05)))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel)
    h_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel)

    v_grad = _normalize01(gx + 0.35 * ax + 0.35 * bx)
    h_grad = _normalize01(gy + 0.35 * ay + 0.35 * by)
    v_long = _normalize01(v_open)
    h_long = _normalize01(h_open)

    v_map = np.clip(0.68 * v_grad + 0.32 * v_long, 0.0, 1.0).astype(np.float32)
    h_map = np.clip(0.68 * h_grad + 0.32 * h_long, 0.0, 1.0).astype(np.float32)

    v_map = cv2.GaussianBlur(v_map, (5, 5), 0)
    h_map = cv2.GaussianBlur(h_map, (5, 5), 0)
    return v_map, h_map


def _sample_polyline_score(
    evidence: np.ndarray,
    poly: list[list[int]],
    orient: str,
    samples: int = 110,
    band: int = 2,
) -> float:
    if len(poly) < 2:
        return -1e9
    h, w = evidence.shape[:2]

    pts = np.asarray(poly, dtype=np.float32)
    seg_len = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
    total = float(seg_len.sum())
    if total < 1.0:
        x = _clip(_safe_int(pts[0, 0]), 0, w - 1)
        y = _clip(_safe_int(pts[0, 1]), 0, h - 1)
        return float(evidence[y, x])

    # Piecewise-linear arc sampling.
    seg_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    out_vals: list[float] = []
    for t in np.linspace(0.0, total, num=max(20, samples)):
        k = int(np.searchsorted(seg_cum, t, side="right") - 1)
        k = _clip(k, 0, len(seg_len) - 1)
        a = pts[k]
        b = pts[k + 1]
        denom = max(1e-6, float(seg_len[k]))
        u = float((t - seg_cum[k]) / denom)
        x = float(a[0] + u * (b[0] - a[0]))
        y = float(a[1] + u * (b[1] - a[1]))
        xi = _clip(_safe_int(x), 0, w - 1)
        yi = _clip(_safe_int(y), 0, h - 1)

        if orient == "v":
            x0 = _clip(xi - band, 0, w - 1)
            x1 = _clip(xi + band, 0, w - 1)
            out_vals.append(float(np.max(evidence[yi, x0 : x1 + 1])))
        else:
            y0 = _clip(yi - band, 0, h - 1)
            y1 = _clip(yi + band, 0, h - 1)
            out_vals.append(float(np.max(evidence[y0 : y1 + 1, xi])))

    if not out_vals:
        return -1e9
    # Robust center tendency so tiny spikes do not dominate.
    return float(np.percentile(np.asarray(out_vals, dtype=np.float32), 65))


def _shift_poly(poly: list[list[int]], dx: int = 0, dy: int = 0) -> list[list[int]]:
    return [[p[0] + dx, p[1] + dy] for p in poly]


def _optimize_line_shift(
    evidence: np.ndarray,
    poly: list[list[int]],
    orient: str,
    max_shift: int,
) -> int:
    best_shift = 0
    best_score = _sample_polyline_score(evidence, poly, orient=orient)
    # Strongly prefer smaller edits if evidence gain is minor.
    for d in range(-max_shift, max_shift + 1):
        cand = _shift_poly(poly, dx=d, dy=0) if orient == "v" else _shift_poly(poly, dx=0, dy=d)
        s = _sample_polyline_score(evidence, cand, orient=orient) - 0.004 * abs(d)
        if s > best_score:
            best_score = s
            best_shift = d
    return int(best_shift)


def _enforce_monotonic(xs: list[int], min_gap: int, lo: int, hi: int) -> list[int]:
    if not xs:
        return xs
    y = [int(v) for v in xs]
    y[0] = _clip(y[0], lo, hi)
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + min_gap)
    for i in range(len(y) - 2, -1, -1):
        y[i] = min(y[i], y[i + 1] - min_gap)
    # Re-clip with one more forward pass.
    y[0] = _clip(y[0], lo, hi)
    for i in range(1, len(y)):
        y[i] = _clip(max(y[i], y[i - 1] + min_gap), lo, hi)
    return y


def _smooth_1d(a: np.ndarray, k: int) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    if a.size <= 1:
        return a
    k = max(3, int(k))
    if k % 2 == 0:
        k += 1
    if k >= a.size:
        k = max(3, (a.size // 2) * 2 + 1)
    if k <= 2:
        return a
    ker = cv2.getGaussianKernel(k, sigma=max(1.0, 0.35 * k)).reshape(-1)
    ker = ker / float(np.sum(ker))
    return np.convolve(a, ker, mode="same").astype(np.float32)


def _robust_projection(
    evidence: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    orient: str,
) -> np.ndarray:
    h, w = evidence.shape[:2]
    x0 = _clip(x0, 0, w - 1)
    x1 = _clip(x1, 0, w - 1)
    y0 = _clip(y0, 0, h - 1)
    y1 = _clip(y1, 0, h - 1)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    roi = evidence[y0 : y1 + 1, x0 : x1 + 1]
    if roi.size == 0:
        n = max(1, (x1 - x0 + 1) if orient == "v" else (y1 - y0 + 1))
        return np.zeros((n,), dtype=np.float32)

    if orient == "v":
        # Per-column continuity score: mean of top-k responses over rows.
        n = roi.shape[0]
        k = max(6, int(round(0.22 * n)))
        k = min(k, n)
        vals = np.sort(roi, axis=0)[-k:, :]
        proj = vals.mean(axis=0)
        ks = max(5, int(round(0.018 * max(1, x1 - x0 + 1))))
    else:
        # Per-row continuity score: mean of top-k responses over cols.
        n = roi.shape[1]
        k = max(6, int(round(0.22 * n)))
        k = min(k, n)
        vals = np.sort(roi, axis=1)[:, -k:]
        proj = vals.mean(axis=1)
        ks = max(5, int(round(0.018 * max(1, y1 - y0 + 1))))

    proj = _normalize01(proj.astype(np.float32))
    proj = _smooth_1d(proj, ks)
    proj = _normalize01(proj)
    return proj


def _refine_table(
    table: dict[str, Any],
    img_bgr: np.ndarray,
    v_map: np.ndarray,
    h_map: np.ndarray,
    max_shift_frac: float,
) -> dict[str, Any]:
    t = json.loads(json.dumps(table))
    h, w = img_bgr.shape[:2]
    bb = t.get("table_bbox", [0, 0, w - 1, h - 1])
    if not (isinstance(bb, list) and len(bb) >= 4):
        bb = [0, 0, w - 1, h - 1]
    x0, y0, x1, y1 = [_clip(_safe_int(v), 0, max(w - 1, h - 1)) for v in bb[:4]]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    tw = max(10, x1 - x0)
    th = max(10, y1 - y0)
    max_v_shift = max(4, int(round(max_shift_frac * tw)))
    max_h_shift = max(4, int(round(max_shift_frac * th)))
    v_proj = _robust_projection(v_map, x0, y0, x1, y1, orient="v")
    h_proj = _robust_projection(h_map, x0, y0, x1, y1, orient="h")

    # Vertical lines: move in x only (blend global projection + local polyline evidence).
    v_lines_raw = t.get("v_lines", [])
    v_lines = [_as_polyline(p) for p in v_lines_raw if _as_polyline(p)]
    if v_lines:
        centers = [int(round(_line_center_x(p))) for p in v_lines]
        shifts: list[int] = []
        for p, c in zip(v_lines, centers):
            best_d = 0
            best_s = -1e9
            for d in range(-max_v_shift, max_v_shift + 1):
                cc = _clip(c + d, x0, x1)
                ps = float(v_proj[_clip(cc - x0, 0, len(v_proj) - 1)]) if len(v_proj) else 0.0
                cand = _shift_poly(p, dx=d, dy=0)
                ls = _sample_polyline_score(v_map, cand, orient="v", samples=96, band=2)
                s = 0.58 * ps + 0.42 * ls - 0.0045 * abs(d)
                if s > best_s:
                    best_s = s
                    best_d = d
            shifts.append(int(best_d))
        moved_centers = [c + d for c, d in zip(centers, shifts)]
        gap = np.diff(np.asarray(centers, dtype=np.float32))
        med_gap = int(round(float(np.median(gap)))) if gap.size else 8
        min_gap = max(3, int(round(0.35 * max(1, med_gap))))
        moved_centers = _enforce_monotonic(moved_centers, min_gap=min_gap, lo=max(0, x0 - 12), hi=min(w - 1, x1 + 12))

        new_v_lines: list[list[list[int]]] = []
        for old_poly, c_old, c_new in zip(v_lines, centers, moved_centers):
            dx = int(c_new - c_old)
            npoly = _shift_poly(old_poly, dx=dx, dy=0)
            # Clamp points to image bounds.
            npoly = [[_clip(px, 0, w - 1), _clip(py, 0, h - 1)] for px, py in npoly]
            new_v_lines.append(npoly)
        t["v_lines"] = new_v_lines

    # Horizontal separators: move in y only (blend global projection + local polyline evidence).
    seps = t.get("separators", [])
    if isinstance(seps, list) and seps:
        polys = [_as_polyline(s.get("polyline", [])) for s in seps]
        centers = [int(round(_line_center_y(p))) if p else y0 for p in polys]
        shifts: list[int] = []
        for p, c in zip(polys, centers):
            if not p:
                shifts.append(0)
                continue
            best_d = 0
            best_s = -1e9
            for d in range(-max_h_shift, max_h_shift + 1):
                cc = _clip(c + d, y0, y1)
                ps = float(h_proj[_clip(cc - y0, 0, len(h_proj) - 1)]) if len(h_proj) else 0.0
                cand = _shift_poly(p, dx=0, dy=d)
                ls = _sample_polyline_score(h_map, cand, orient="h", samples=96, band=2)
                s = 0.58 * ps + 0.42 * ls - 0.0045 * abs(d)
                if s > best_s:
                    best_s = s
                    best_d = d
            shifts.append(int(best_d))
        moved_centers = [c + d for c, d in zip(centers, shifts)]
        gap = np.diff(np.asarray(centers, dtype=np.float32))
        med_gap = int(round(float(np.median(gap)))) if gap.size else 8
        min_gap = max(3, int(round(0.35 * max(1, med_gap))))
        moved_centers = _enforce_monotonic(moved_centers, min_gap=min_gap, lo=max(0, y0 - 12), hi=min(h - 1, y1 + 12))

        new_seps: list[dict[str, Any]] = []
        for s, p_old, c_old, c_new in zip(seps, polys, centers, moved_centers):
            if not p_old:
                new_seps.append(s)
                continue
            dy = int(c_new - c_old)
            npoly = _shift_poly(p_old, dx=0, dy=dy)
            npoly = [[_clip(px, 0, w - 1), _clip(py, 0, h - 1)] for px, py in npoly]
            s2 = dict(s)
            s2["polyline"] = npoly
            new_seps.append(s2)
        t["separators"] = new_seps

    # Tighten bbox from refined lines with bounded drift from original bbox.
    xs: list[int] = []
    ys: list[int] = []
    for p in t.get("v_lines", []):
        pp = _as_polyline(p)
        xs.extend([pt[0] for pt in pp])
        ys.extend([pt[1] for pt in pp])
    for s in t.get("separators", []):
        pp = _as_polyline(s.get("polyline", []))
        xs.extend([pt[0] for pt in pp])
        ys.extend([pt[1] for pt in pp])
    if xs and ys:
        nx0 = _clip(min(xs) - 3, max(0, x0 - 14), min(w - 1, x0 + 14))
        nx1 = _clip(max(xs) + 3, max(0, x1 - 14), min(w - 1, x1 + 14))
        ny0 = _clip(min(ys) - 3, max(0, y0 - 14), min(h - 1, y0 + 14))
        ny1 = _clip(max(ys) + 3, max(0, y1 - 14), min(h - 1, y1 + 14))
        if nx1 > nx0 + 8 and ny1 > ny0 + 8:
            t["table_bbox"] = [int(nx0), int(ny0), int(nx1), int(ny1)]

    return t


def refine_labels(
    labels_dir: Path,
    out_dir: Path,
    start_index: int = 5,
    max_shift_frac: float = 0.06,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _collect_jsons(labels_dir)
    if not files:
        raise RuntimeError(f"No JSON labels found in {labels_dir}")

    for i, p in enumerate(files):
        d = json.loads(p.read_text(encoding="utf-8"))
        out_path = out_dir / p.name

        if i < start_index:
            out_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue

        image_path = Path(str(d.get("image", "")))
        if not image_path.exists():
            out_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            out_path.write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue

        v_map, h_map = _build_evidence_maps(img)
        tables = d.get("tables", [])
        if not isinstance(tables, list):
            tables = []

        new_tables: list[dict[str, Any]] = []
        for t in tables:
            if not isinstance(t, dict):
                continue
            new_tables.append(
                _refine_table(
                    table=t,
                    img_bgr=img,
                    v_map=v_map,
                    h_map=h_map,
                    max_shift_frac=max_shift_frac,
                )
            )
        d["tables"] = new_tables
        out_path.write_text(json.dumps(d, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic row-separator label refinement.")
    ap.add_argument("--labels_dir", required=True, help="Input row-separator labels dir.")
    ap.add_argument("--out_dir", required=True, help="Output labels dir.")
    ap.add_argument("--start_index", type=int, default=5, help="Leave [0:start_index) untouched.")
    ap.add_argument(
        "--max_shift_frac",
        type=float,
        default=0.06,
        help="Max per-line shift as a fraction of table width/height.",
    )
    args = ap.parse_args()

    refine_labels(
        labels_dir=Path(args.labels_dir),
        out_dir=Path(args.out_dir),
        start_index=max(0, int(args.start_index)),
        max_shift_frac=max(0.01, float(args.max_shift_frac)),
    )


if __name__ == "__main__":
    main()
