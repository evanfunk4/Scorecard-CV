#!/usr/bin/env python3
"""
Convolution-based snapping refinement for row-separator labels.

Uses oriented convolution evidence to move existing lines while preserving topology:
- same table count
- same vertical-line count per table
- same separator count per table
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


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


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


def _shift_poly(poly: list[list[int]], dx: int = 0, dy: int = 0) -> list[list[int]]:
    return [[p[0] + dx, p[1] + dy] for p in poly]


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


def _build_conv_evidence(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 55, 55)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    g = gray.astype(np.float32)

    sx_g = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    sy_g = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    sx_a = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    sy_a = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    sx_b = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    sy_b = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)

    v = np.abs(sx_g) + 0.45 * np.abs(sx_a) + 0.45 * np.abs(sx_b)
    hmap = np.abs(sy_g) + 0.45 * np.abs(sy_a) + 0.45 * np.abs(sy_b)

    # Oriented continuity via long convolutions.
    kv = max(15, int(round(0.09 * h)))
    if kv % 2 == 0:
        kv += 1
    kh = max(15, int(round(0.09 * w)))
    if kh % 2 == 0:
        kh += 1
    v = cv2.GaussianBlur(v, (1, kv), 0)
    v = cv2.GaussianBlur(v, (5, 1), 0)
    hmap = cv2.GaussianBlur(hmap, (kh, 1), 0)
    hmap = cv2.GaussianBlur(hmap, (1, 5), 0)

    # Add long-line morphology channel to suppress text effects.
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 9
    )
    v_len = max(15, int(round(0.06 * h)))
    h_len = max(15, int(round(0.06 * w)))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel).astype(np.float32)
    h_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel).astype(np.float32)

    v = _normalize01(v)
    hmap = _normalize01(hmap)
    v_open = _normalize01(v_open)
    h_open = _normalize01(h_open)
    v = _normalize01(0.78 * v + 0.22 * v_open)
    hmap = _normalize01(0.78 * hmap + 0.22 * h_open)
    return v, hmap


def _sample_polyline_score(
    evidence: np.ndarray,
    poly: list[list[int]],
    orient: str,
    samples: int = 100,
    band: int = 2,
) -> float:
    if len(poly) < 2:
        return -1e9
    h, w = evidence.shape[:2]
    pts = np.asarray(poly, dtype=np.float32)
    seg = np.sqrt(np.sum((pts[1:] - pts[:-1]) ** 2, axis=1))
    total = float(seg.sum())
    if total < 1.0:
        x = _clip(_safe_int(pts[0, 0]), 0, w - 1)
        y = _clip(_safe_int(pts[0, 1]), 0, h - 1)
        return float(evidence[y, x])

    cum = np.concatenate([[0.0], np.cumsum(seg)])
    vals: list[float] = []
    for t in np.linspace(0.0, total, num=max(24, samples)):
        k = int(np.searchsorted(cum, t, side="right") - 1)
        k = _clip(k, 0, len(seg) - 1)
        a = pts[k]
        b = pts[k + 1]
        denom = max(1e-6, float(seg[k]))
        u = float((t - cum[k]) / denom)
        x = float(a[0] + u * (b[0] - a[0]))
        y = float(a[1] + u * (b[1] - a[1]))
        xi = _clip(_safe_int(x), 0, w - 1)
        yi = _clip(_safe_int(y), 0, h - 1)
        if orient == "v":
            x0 = _clip(xi - band, 0, w - 1)
            x1 = _clip(xi + band, 0, w - 1)
            vals.append(float(np.max(evidence[yi, x0 : x1 + 1])))
        else:
            y0 = _clip(yi - band, 0, h - 1)
            y1 = _clip(yi + band, 0, h - 1)
            vals.append(float(np.max(evidence[y0 : y1 + 1, xi])))
    if not vals:
        return -1e9
    return float(np.percentile(np.asarray(vals, dtype=np.float32), 65))


def _projection(evidence: np.ndarray, x0: int, y0: int, x1: int, y1: int, orient: str) -> np.ndarray:
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
        n = roi.shape[0]
        k = max(6, int(round(0.22 * n)))
        k = min(k, n)
        vals = np.sort(roi, axis=0)[-k:, :]
        p = vals.mean(axis=0)
        ks = max(5, int(round(0.02 * max(1, x1 - x0 + 1))))
    else:
        n = roi.shape[1]
        k = max(6, int(round(0.22 * n)))
        k = min(k, n)
        vals = np.sort(roi, axis=1)[:, -k:]
        p = vals.mean(axis=1)
        ks = max(5, int(round(0.02 * max(1, y1 - y0 + 1))))
    p = _normalize01(p.astype(np.float32))
    p = _smooth_1d(p, ks)
    return _normalize01(p)


def _enforce_monotonic(xs: list[int], min_gap: int, lo: int, hi: int) -> list[int]:
    if not xs:
        return xs
    y = [int(v) for v in xs]
    y[0] = _clip(y[0], lo, hi)
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + min_gap)
    for i in range(len(y) - 2, -1, -1):
        y[i] = min(y[i], y[i + 1] - min_gap)
    y[0] = _clip(y[0], lo, hi)
    for i in range(1, len(y)):
        y[i] = _clip(max(y[i], y[i - 1] + min_gap), lo, hi)
    return y


def _optimize_centers(
    centers: list[int],
    polylines: list[list[list[int]]],
    proj: np.ndarray,
    evidence: np.ndarray,
    orient: str,
    lo: int,
    hi: int,
    max_shift: int,
) -> list[int]:
    if not centers:
        return centers
    new_centers: list[int] = []
    for c, p in zip(centers, polylines):
        best_c = c
        best_s = -1e9
        for d in range(-max_shift, max_shift + 1):
            cc = _clip(c + d, lo, hi)
            idx = _clip(cc - lo, 0, len(proj) - 1)
            ps = float(proj[idx]) if len(proj) else 0.0
            cand = _shift_poly(p, dx=d, dy=0) if orient == "v" else _shift_poly(p, dx=0, dy=d)
            ls = _sample_polyline_score(evidence, cand, orient=orient, samples=88, band=2)
            score = 0.60 * ps + 0.40 * ls - 0.005 * abs(d)
            if score > best_s:
                best_s = score
                best_c = cc
        new_centers.append(int(best_c))

    gap = np.diff(np.asarray(centers, dtype=np.float32))
    med_gap = int(round(float(np.median(gap)))) if gap.size else 8
    min_gap = max(3, int(round(0.38 * max(1, med_gap))))
    return _enforce_monotonic(new_centers, min_gap=min_gap, lo=lo, hi=hi)


def _refine_table(table: dict[str, Any], v_map: np.ndarray, h_map: np.ndarray, max_shift_frac: float) -> dict[str, Any]:
    t = json.loads(json.dumps(table))
    h, w = v_map.shape[:2]
    bb = t.get("table_bbox", [0, 0, w - 1, h - 1])
    if not (isinstance(bb, list) and len(bb) >= 4):
        bb = [0, 0, w - 1, h - 1]
    x0, y0, x1, y1 = [_safe_int(v) for v in bb[:4]]
    x0 = _clip(x0, 0, w - 1)
    x1 = _clip(x1, 0, w - 1)
    y0 = _clip(y0, 0, h - 1)
    y1 = _clip(y1, 0, h - 1)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0

    tw = max(10, x1 - x0)
    th = max(10, y1 - y0)
    max_v_shift = max(5, int(round(max_shift_frac * tw)))
    max_h_shift = max(5, int(round(max_shift_frac * th)))

    v_proj = _projection(v_map, x0, y0, x1, y1, orient="v")
    h_proj = _projection(h_map, x0, y0, x1, y1, orient="h")

    # Vertical lines.
    v_lines = [_as_polyline(p) for p in t.get("v_lines", []) if _as_polyline(p)]
    if v_lines:
        centers = [int(round(_line_center_x(p))) for p in v_lines]
        new_centers = _optimize_centers(
            centers=centers,
            polylines=v_lines,
            proj=v_proj,
            evidence=v_map,
            orient="v",
            lo=max(0, x0 - 18),
            hi=min(w - 1, x1 + 18),
            max_shift=max_v_shift,
        )
        new_v: list[list[list[int]]] = []
        for old_poly, c_old, c_new in zip(v_lines, centers, new_centers):
            dx = int(c_new - c_old)
            npoly = _shift_poly(old_poly, dx=dx, dy=0)
            npoly = [[_clip(px, 0, w - 1), _clip(py, 0, h - 1)] for px, py in npoly]
            new_v.append(npoly)
        t["v_lines"] = new_v
        kinds = t.get("v_line_kinds", [])
        if not isinstance(kinds, list):
            kinds = []
        if len(kinds) != len(new_v):
            t["v_line_kinds"] = (kinds + ["line"] * len(new_v))[: len(new_v)]

    # Horizontal separators.
    seps_raw = t.get("separators", [])
    if isinstance(seps_raw, list) and seps_raw:
        polys: list[list[list[int]]] = []
        seps_meta: list[dict[str, Any]] = []
        for s in seps_raw:
            if not isinstance(s, dict):
                continue
            p = _as_polyline(s.get("polyline", []))
            if len(p) < 2:
                continue
            polys.append(p)
            seps_meta.append(s)
        if polys:
            centers = [int(round(_line_center_y(p))) for p in polys]
            new_centers = _optimize_centers(
                centers=centers,
                polylines=polys,
                proj=h_proj,
                evidence=h_map,
                orient="h",
                lo=max(0, y0 - 18),
                hi=min(h - 1, y1 + 18),
                max_shift=max_h_shift,
            )
            new_seps: list[dict[str, Any]] = []
            for s, old_poly, c_old, c_new in zip(seps_meta, polys, centers, new_centers):
                dy = int(c_new - c_old)
                npoly = _shift_poly(old_poly, dx=0, dy=dy)
                npoly = [[_clip(px, 0, w - 1), _clip(py, 0, h - 1)] for px, py in npoly]
                s2 = dict(s)
                s2["polyline"] = npoly
                kind = str(s2.get("kind", "line")).strip().lower()
                if kind not in {"line", "boundary"}:
                    kind = "line"
                s2["kind"] = kind
                new_seps.append(s2)
            t["separators"] = new_seps

    # Bounded bbox refresh from updated lines.
    xs: list[int] = []
    ys: list[int] = []
    for p in t.get("v_lines", []):
        pp = _as_polyline(p)
        xs.extend([q[0] for q in pp])
        ys.extend([q[1] for q in pp])
    for s in t.get("separators", []):
        pp = _as_polyline(s.get("polyline", []))
        xs.extend([q[0] for q in pp])
        ys.extend([q[1] for q in pp])
    if xs and ys:
        nx0 = _clip(min(xs) - 3, max(0, x0 - 16), min(w - 1, x0 + 16))
        nx1 = _clip(max(xs) + 3, max(0, x1 - 16), min(w - 1, x1 + 16))
        ny0 = _clip(min(ys) - 3, max(0, y0 - 16), min(h - 1, y0 + 16))
        ny1 = _clip(max(ys) + 3, max(0, y1 - 16), min(h - 1, y1 + 16))
        if nx1 > nx0 + 8 and ny1 > ny0 + 8:
            t["table_bbox"] = [int(nx0), int(ny0), int(nx1), int(ny1)]
    return t


def refine(
    labels_dir: Path,
    out_dir: Path,
    start_index: int,
    max_shift_frac: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = _collect_jsons(labels_dir)
    if not files:
        raise RuntimeError(f"No json labels in {labels_dir}")

    for i, p in enumerate(files):
        d = json.loads(p.read_text(encoding="utf-8"))
        outp = out_dir / p.name
        if i < start_index:
            outp.write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue
        ip = Path(str(d.get("image", "")))
        if not ip.exists():
            outp.write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue
        img = cv2.imread(str(ip), cv2.IMREAD_COLOR)
        if img is None:
            outp.write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue
        v_map, h_map = _build_conv_evidence(img)
        tabs = d.get("tables", [])
        if not isinstance(tabs, list):
            tabs = []
        new_tabs: list[dict[str, Any]] = []
        for t in tabs:
            if not isinstance(t, dict):
                continue
            new_tabs.append(_refine_table(t, v_map=v_map, h_map=h_map, max_shift_frac=max_shift_frac))
        d["tables"] = new_tabs
        outp.write_text(json.dumps(d, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Convolution evidence snap refinement for rowsep labels.")
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start_index", type=int, default=5)
    ap.add_argument("--max_shift_frac", type=float, default=0.20)
    args = ap.parse_args()
    refine(
        labels_dir=Path(args.labels_dir),
        out_dir=Path(args.out_dir),
        start_index=max(0, int(args.start_index)),
        max_shift_frac=max(0.02, float(args.max_shift_frac)),
    )


if __name__ == "__main__":
    main()

