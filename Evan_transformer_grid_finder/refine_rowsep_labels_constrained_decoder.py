#!/usr/bin/env python3
"""
Constrained decoder refinement for row-separator labels.

Design goals:
- Keep each table count unchanged.
- Use strong oriented evidence (line + color-boundary) to snap rows/columns.
- Rebuild per-segment geometry between guide lines (handles merged cells).
- Leave first N samples untouched for trusted labels.
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


def _normalize01(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    lo = float(np.percentile(x, 1.0))
    hi = float(np.percentile(x, 99.0))
    if hi <= lo + 1e-6:
        return np.zeros_like(x, dtype=np.float32)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0).astype(np.float32)


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


def _as_polyline(raw: Any) -> list[list[int]]:
    out: list[list[int]] = []
    if isinstance(raw, list):
        for p in raw:
            if isinstance(p, (list, tuple)) and len(p) >= 2:
                out.append([_safe_int(p[0]), _safe_int(p[1])])
    return out


def _poly_x(poly: list[list[int]]) -> float:
    if not poly:
        return 0.0
    return float(sum(p[0] for p in poly)) / float(len(poly))


def _poly_y(poly: list[list[int]]) -> float:
    if not poly:
        return 0.0
    return float(sum(p[1] for p in poly)) / float(len(poly))


def _poly_minmax_x(poly: list[list[int]]) -> tuple[int, int]:
    xs = [int(p[0]) for p in poly]
    if not xs:
        return (0, 0)
    return (min(xs), max(xs))


def _poly_minmax_y(poly: list[list[int]]) -> tuple[int, int]:
    ys = [int(p[1]) for p in poly]
    if not ys:
        return (0, 0)
    return (min(ys), max(ys))


def _norm_kind(kind: Any) -> str:
    k = str(kind).strip().lower()
    return k if k in {"line", "boundary"} else "line"


def _collect_jsons(labels_dir: Path) -> list[Path]:
    return sorted(p for p in labels_dir.glob("*.json") if p.is_file())


def _build_evidence_maps(img_bgr: np.ndarray) -> dict[str, np.ndarray]:
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 55, 55)

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = lab[:, :, 1]
    b = lab[:, :, 2]
    g = gray.astype(np.float32)

    gx_g = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy_g = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    gx_a = np.abs(cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3))
    gy_a = np.abs(cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3))
    gx_b = np.abs(cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3))
    gy_b = np.abs(cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3))

    v_bound = gx_g + 0.55 * gx_a + 0.55 * gx_b
    h_bound = gy_g + 0.55 * gy_a + 0.55 * gy_b

    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 9
    )
    v_len = max(15, int(round(0.06 * h)))
    h_len = max(15, int(round(0.06 * w)))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_kernel).astype(np.float32)
    h_open = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_kernel).astype(np.float32)

    v_line = 0.70 * _normalize01(v_open) + 0.30 * _normalize01(gx_g)
    h_line = 0.70 * _normalize01(h_open) + 0.30 * _normalize01(gy_g)
    v_bound = _normalize01(v_bound)
    h_bound = _normalize01(h_bound)

    kv = max(13, int(round(0.09 * h)))
    if kv % 2 == 0:
        kv += 1
    kh = max(13, int(round(0.09 * w)))
    if kh % 2 == 0:
        kh += 1

    v_line = cv2.GaussianBlur(v_line, (1, kv), 0)
    v_line = cv2.GaussianBlur(v_line, (5, 1), 0)
    h_line = cv2.GaussianBlur(h_line, (kh, 1), 0)
    h_line = cv2.GaussianBlur(h_line, (1, 5), 0)

    v_bound = cv2.GaussianBlur(v_bound, (1, kv), 0)
    v_bound = cv2.GaussianBlur(v_bound, (3, 1), 0)
    h_bound = cv2.GaussianBlur(h_bound, (kh, 1), 0)
    h_bound = cv2.GaussianBlur(h_bound, (1, 3), 0)

    v_line = _normalize01(v_line)
    h_line = _normalize01(h_line)
    v_bound = _normalize01(v_bound)
    h_bound = _normalize01(h_bound)

    v_mix = _normalize01(0.74 * v_line + 0.26 * v_bound)
    h_mix = _normalize01(0.74 * h_line + 0.26 * h_bound)
    return {
        "v_line": v_line,
        "h_line": h_line,
        "v_bound": v_bound,
        "h_bound": h_bound,
        "v_mix": v_mix,
        "h_mix": h_mix,
    }


def _cluster_positions(vals: list[float], tol: float) -> list[list[float]]:
    if not vals:
        return []
    s = sorted(float(v) for v in vals)
    out: list[list[float]] = [[s[0]]]
    for v in s[1:]:
        if abs(v - float(np.mean(out[-1]))) <= tol:
            out[-1].append(v)
        else:
            out.append([v])
    return out


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


def _projection(
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
        n = roi.shape[0]
        k = max(6, int(round(0.23 * n)))
        k = min(k, n)
        vals = np.sort(roi, axis=0)[-k:, :]
        p = vals.mean(axis=0)
        ks = max(5, int(round(0.02 * max(1, x1 - x0 + 1))))
    else:
        n = roi.shape[1]
        k = max(6, int(round(0.23 * n)))
        k = min(k, n)
        vals = np.sort(roi, axis=1)[:, -k:]
        p = vals.mean(axis=1)
        ks = max(5, int(round(0.02 * max(1, y1 - y0 + 1))))

    p = _normalize01(p.astype(np.float32))
    p = _smooth_1d(p, ks)
    return _normalize01(p)


def _line_support_full(
    evidence: np.ndarray,
    orient: str,
    coord: int,
    a0: int,
    a1: int,
    band: int = 2,
) -> float:
    h, w = evidence.shape[:2]
    if orient == "v":
        x = _clip(coord, 0, w - 1)
        y0 = _clip(min(a0, a1), 0, h - 1)
        y1 = _clip(max(a0, a1), 0, h - 1)
        x0 = _clip(x - band, 0, w - 1)
        x1 = _clip(x + band, 0, w - 1)
        arr = evidence[y0 : y1 + 1, x0 : x1 + 1]
        if arr.size == 0:
            return 0.0
        vec = np.max(arr, axis=1)
    else:
        y = _clip(coord, 0, h - 1)
        x0 = _clip(min(a0, a1), 0, w - 1)
        x1 = _clip(max(a0, a1), 0, w - 1)
        y0 = _clip(y - band, 0, h - 1)
        y1 = _clip(y + band, 0, h - 1)
        arr = evidence[y0 : y1 + 1, x0 : x1 + 1]
        if arr.size == 0:
            return 0.0
        vec = np.max(arr, axis=0)
    return float(np.percentile(vec.astype(np.float32), 70))


def _snap_centers(
    centers: list[int],
    proj: np.ndarray,
    evidence: np.ndarray,
    orient: str,
    lo: int,
    hi: int,
    a0: int,
    a1: int,
    max_shift: int,
) -> list[int]:
    if not centers:
        return []
    new_centers: list[int] = []
    for c in centers:
        best_c = c
        best_s = -1e9
        for d in range(-max_shift, max_shift + 1):
            cc = _clip(c + d, lo, hi)
            idx = _clip(cc - lo, 0, len(proj) - 1)
            ps = float(proj[idx]) if len(proj) else 0.0
            ls = _line_support_full(evidence, orient, cc, a0, a1, band=2)
            score = 0.60 * ps + 0.40 * ls - 0.005 * abs(d)
            if score > best_s:
                best_s = score
                best_c = cc
        new_centers.append(int(best_c))

    if len(new_centers) >= 2:
        gap = np.diff(np.asarray(sorted(centers), dtype=np.float32))
        med_gap = float(np.median(gap)) if gap.size else 8.0
        min_gap = max(3, int(round(0.36 * max(1.0, med_gap))))
    else:
        min_gap = 3
    out = _enforce_monotonic(new_centers, min_gap=min_gap, lo=lo, hi=hi)
    return out


def _build_guides(lo: int, hi: int, centers: list[int], tol: int) -> list[int]:
    vals = [int(lo), int(hi)] + [int(v) for v in centers]
    vals = sorted(vals)
    if not vals:
        return [int(lo), int(hi)]
    merged: list[int] = [vals[0]]
    for v in vals[1:]:
        if abs(v - merged[-1]) <= tol:
            merged[-1] = int(round(0.5 * (merged[-1] + v)))
        else:
            merged.append(int(v))
    if merged[0] != lo:
        merged.insert(0, int(lo))
    if merged[-1] != hi:
        merged.append(int(hi))
    merged = sorted(set(merged))
    if len(merged) < 2:
        merged = [int(lo), int(hi)]
    return merged


def _mode_kind(kinds: list[str], default: str = "line") -> str:
    if not kinds:
        return default
    line_n = sum(1 for k in kinds if _norm_kind(k) == "line")
    bound_n = sum(1 for k in kinds if _norm_kind(k) == "boundary")
    return "boundary" if bound_n > line_n else "line"


def _rebuild_table(
    table: dict[str, Any],
    maps: dict[str, np.ndarray],
    max_shift_frac: float,
) -> dict[str, Any]:
    t = json.loads(json.dumps(table))
    h, w = maps["v_mix"].shape[:2]
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
    max_v_shift = max(4, int(round(max_shift_frac * tw)))
    max_h_shift = max(4, int(round(max_shift_frac * th)))

    # Collapse current geometry into row/column anchors first.
    v_lines_raw = t.get("v_lines", [])
    v_kinds_raw = t.get("v_line_kinds", [])
    if not isinstance(v_lines_raw, list):
        v_lines_raw = []
    if not isinstance(v_kinds_raw, list):
        v_kinds_raw = []
    v_entries: list[dict[str, Any]] = []
    for i, ln in enumerate(v_lines_raw):
        p = _as_polyline(ln)
        if len(p) < 2:
            continue
        ys0, ys1 = _poly_minmax_y(p)
        v_entries.append(
            {
                "x": _poly_x(p),
                "y0": int(ys0),
                "y1": int(ys1),
                "kind": _norm_kind(v_kinds_raw[i] if i < len(v_kinds_raw) else "line"),
            }
        )

    seps_raw = t.get("separators", [])
    if not isinstance(seps_raw, list):
        seps_raw = []
    h_entries: list[dict[str, Any]] = []
    for s in seps_raw:
        if not isinstance(s, dict):
            continue
        p = _as_polyline(s.get("polyline", []))
        if len(p) < 2:
            continue
        xs0, xs1 = _poly_minmax_x(p)
        h_entries.append(
            {
                "y": _poly_y(p),
                "x0": int(xs0),
                "x1": int(xs1),
                "kind": _norm_kind(s.get("kind", "line")),
            }
        )

    if not v_entries or not h_entries:
        return t

    x_tol = max(4, int(round(0.010 * tw)))
    y_tol = max(4, int(round(0.008 * th)))

    x_clusters = _cluster_positions([e["x"] for e in v_entries], tol=float(x_tol))
    y_clusters = _cluster_positions([e["y"] for e in h_entries], tol=float(y_tol))
    if not x_clusters or not y_clusters:
        return t

    col_centers = [int(round(float(np.mean(c)))) for c in x_clusters]
    row_centers = [int(round(float(np.mean(c)))) for c in y_clusters]
    col_centers = sorted(col_centers)
    row_centers = sorted(row_centers)

    v_proj = _projection(maps["v_mix"], x0, y0, x1, y1, orient="v")
    h_proj = _projection(maps["h_mix"], x0, y0, x1, y1, orient="h")
    col_centers = _snap_centers(
        centers=col_centers,
        proj=v_proj,
        evidence=maps["v_mix"],
        orient="v",
        lo=x0,
        hi=x1,
        a0=y0,
        a1=y1,
        max_shift=max_v_shift,
    )
    row_centers = _snap_centers(
        centers=row_centers,
        proj=h_proj,
        evidence=maps["h_mix"],
        orient="h",
        lo=y0,
        hi=y1,
        a0=x0,
        a1=x1,
        max_shift=max_h_shift,
    )

    x_guides = _build_guides(x0, x1, col_centers, tol=max(3, x_tol // 2))
    y_guides = _build_guides(y0, y1, row_centers, tol=max(3, y_tol // 2))

    if len(x_guides) < 2 or len(y_guides) < 2:
        return t

    # Build prior coverage from existing segmented geometry.
    n_col = len(col_centers)
    n_row = len(row_centers)
    n_yband = len(y_guides) - 1
    n_xspan = len(x_guides) - 1

    v_prior_exist = np.zeros((n_col, n_yband), dtype=np.float32)
    v_prior_bound = np.zeros((n_col, n_yband), dtype=np.float32)
    h_prior_exist = np.zeros((n_row, n_xspan), dtype=np.float32)
    h_prior_bound = np.zeros((n_row, n_xspan), dtype=np.float32)

    for e in v_entries:
        if n_col <= 0:
            break
        ci = int(np.argmin(np.abs(np.asarray(col_centers, dtype=np.float32) - float(e["x"]))))
        ey0, ey1 = int(e["y0"]), int(e["y1"])
        for bj in range(n_yband):
            b0, b1 = y_guides[bj], y_guides[bj + 1]
            ov = max(0, min(ey1, b1) - max(ey0, b0))
            if ov >= 0.25 * max(1, b1 - b0):
                v_prior_exist[ci, bj] += 1.0
                if e["kind"] == "boundary":
                    v_prior_bound[ci, bj] += 1.0

    for e in h_entries:
        if n_row <= 0:
            break
        ri = int(np.argmin(np.abs(np.asarray(row_centers, dtype=np.float32) - float(e["y"]))))
        ex0, ex1 = int(e["x0"]), int(e["x1"])
        for sj in range(n_xspan):
            s0, s1 = x_guides[sj], x_guides[sj + 1]
            ov = max(0, min(ex1, s1) - max(ex0, s0))
            if ov >= 0.25 * max(1, s1 - s0):
                h_prior_exist[ri, sj] += 1.0
                if e["kind"] == "boundary":
                    h_prior_bound[ri, sj] += 1.0

    # Rebuild vertical segments band-by-band, then merge contiguous kept bands.
    v_ls = np.zeros((n_col, n_yband), dtype=np.float32)
    v_bs = np.zeros((n_col, n_yband), dtype=np.float32)
    v_score = np.zeros((n_col, n_yband), dtype=np.float32)
    v_valid = np.zeros((n_col, n_yband), dtype=bool)
    v_kind_id = np.zeros((n_col, n_yband), dtype=np.int32)  # 0=line, 1=boundary
    v_score_list: list[float] = []
    for ci, x in enumerate(col_centers):
        for bj in range(n_yband):
            yb0, yb1 = y_guides[bj], y_guides[bj + 1]
            if yb1 - yb0 < 6:
                continue
            ls = _line_support_full(maps["v_line"], "v", x, yb0, yb1, band=2)
            bs = _line_support_full(maps["v_bound"], "v", x, yb0, yb1, band=2)
            prior_on = 1.0 if v_prior_exist[ci, bj] > 0.0 else 0.0
            sc = max(ls, 0.90 * bs) + 0.08 * prior_on
            v_ls[ci, bj] = float(ls)
            v_bs[ci, bj] = float(bs)
            v_score[ci, bj] = float(sc)
            v_valid[ci, bj] = True
            v_score_list.append(float(sc))
            k_prior_boundary = v_prior_bound[ci, bj] > 0.0 and v_prior_exist[ci, bj] > 0.0
            if k_prior_boundary and bs >= 0.75 * ls:
                v_kind_id[ci, bj] = 1
            else:
                v_kind_id[ci, bj] = 1 if (bs > 1.12 * ls and bs >= 0.20) else 0

    if v_score_list:
        v_thr = float(np.percentile(np.asarray(v_score_list, dtype=np.float32), 35))
    else:
        v_thr = 0.24
    v_thr = float(np.clip(v_thr, 0.17, 0.40))

    v_keep = np.zeros((n_col, n_yband), dtype=bool)
    for ci in range(n_col):
        for bj in range(n_yband):
            if not v_valid[ci, bj]:
                continue
            prior = bool(v_prior_exist[ci, bj] > 0.0)
            sc = float(v_score[ci, bj])
            keep = (sc >= v_thr) or (prior and sc >= (v_thr - 0.08))
            v_keep[ci, bj] = keep
        if np.any(v_valid[ci, :]) and not np.any(v_keep[ci, :]):
            # Guarantee one segment per column.
            bj_best = int(np.argmax(np.where(v_valid[ci, :], v_score[ci, :], -1e9)))
            if v_valid[ci, bj_best]:
                v_keep[ci, bj_best] = True

    new_v_lines: list[list[list[int]]] = []
    new_v_kinds: list[str] = []
    for ci, x in enumerate(col_centers):
        run_start = -1
        run_kind = 0
        for bj in range(n_yband + 1):
            active = False
            kind_id = 0
            if bj < n_yband and v_keep[ci, bj]:
                active = True
                kind_id = int(v_kind_id[ci, bj])
            if run_start < 0 and active:
                run_start = bj
                run_kind = kind_id
                continue
            if run_start >= 0 and active and kind_id == run_kind:
                continue
            if run_start >= 0:
                yb0 = int(y_guides[run_start])
                yb1 = int(y_guides[bj])
                if yb1 - yb0 >= 4:
                    ym = int(round(0.5 * (yb0 + yb1)))
                    new_v_lines.append([[int(x), yb0], [int(x), ym], [int(x), yb1]])
                    new_v_kinds.append("boundary" if run_kind == 1 else "line")
                run_start = -1
            if active:
                run_start = bj
                run_kind = kind_id

    # Rebuild horizontal segments span-by-span, then merge contiguous kept spans.
    h_ls = np.zeros((n_row, n_xspan), dtype=np.float32)
    h_bs = np.zeros((n_row, n_xspan), dtype=np.float32)
    h_score = np.zeros((n_row, n_xspan), dtype=np.float32)
    h_valid = np.zeros((n_row, n_xspan), dtype=bool)
    h_kind_id = np.zeros((n_row, n_xspan), dtype=np.int32)  # 0=line, 1=boundary
    h_score_list: list[float] = []
    for ri, y in enumerate(row_centers):
        for sj in range(n_xspan):
            xb0, xb1 = x_guides[sj], x_guides[sj + 1]
            if xb1 - xb0 < 6:
                continue
            ls = _line_support_full(maps["h_line"], "h", y, xb0, xb1, band=2)
            bs = _line_support_full(maps["h_bound"], "h", y, xb0, xb1, band=2)
            prior_on = 1.0 if h_prior_exist[ri, sj] > 0.0 else 0.0
            sc = max(ls, 0.92 * bs) + 0.08 * prior_on
            h_ls[ri, sj] = float(ls)
            h_bs[ri, sj] = float(bs)
            h_score[ri, sj] = float(sc)
            h_valid[ri, sj] = True
            h_score_list.append(float(sc))
            k_prior_boundary = h_prior_bound[ri, sj] > 0.0 and h_prior_exist[ri, sj] > 0.0
            if k_prior_boundary and bs >= 0.75 * ls:
                h_kind_id[ri, sj] = 1
            else:
                h_kind_id[ri, sj] = 1 if (bs > 1.10 * ls and bs >= 0.20) else 0

    if h_score_list:
        h_thr = float(np.percentile(np.asarray(h_score_list, dtype=np.float32), 33))
    else:
        h_thr = 0.22
    h_thr = float(np.clip(h_thr, 0.16, 0.38))

    h_keep = np.zeros((n_row, n_xspan), dtype=bool)
    for ri in range(n_row):
        for sj in range(n_xspan):
            if not h_valid[ri, sj]:
                continue
            prior = bool(h_prior_exist[ri, sj] > 0.0)
            sc = float(h_score[ri, sj])
            keep = (sc >= h_thr) or (prior and sc >= (h_thr - 0.08))
            h_keep[ri, sj] = keep
        if np.any(h_valid[ri, :]) and not np.any(h_keep[ri, :]):
            # Guarantee one segment per row.
            sj_best = int(np.argmax(np.where(h_valid[ri, :], h_score[ri, :], -1e9)))
            if h_valid[ri, sj_best]:
                h_keep[ri, sj_best] = True

    new_seps: list[dict[str, Any]] = []
    for ri, y in enumerate(row_centers):
        run_start = -1
        run_kind = 0
        for sj in range(n_xspan + 1):
            active = False
            kind_id = 0
            if sj < n_xspan and h_keep[ri, sj]:
                active = True
                kind_id = int(h_kind_id[ri, sj])
            if run_start < 0 and active:
                run_start = sj
                run_kind = kind_id
                continue
            if run_start >= 0 and active and kind_id == run_kind:
                continue
            if run_start >= 0:
                xb0 = int(x_guides[run_start])
                xb1 = int(x_guides[sj])
                if xb1 - xb0 >= 4:
                    xm = int(round(0.5 * (xb0 + xb1)))
                    new_seps.append(
                        {
                            "kind": "boundary" if run_kind == 1 else "line",
                            "polyline": [[xb0, int(y)], [xm, int(y)], [xb1, int(y)]],
                        }
                    )
                run_start = -1
            if active:
                run_start = sj
                run_kind = kind_id

    # Final ordering and sanity.
    vz = list(zip(new_v_lines, new_v_kinds))
    vz.sort(key=lambda it: (float(np.mean([p[0] for p in it[0]])), float(np.mean([p[1] for p in it[0]]))))
    new_v_lines = [it[0] for it in vz]
    new_v_kinds = [it[1] for it in vz]
    new_seps.sort(key=lambda s: float(np.mean([p[1] for p in s["polyline"]])))

    if new_v_lines:
        t["v_lines"] = new_v_lines
        t["v_line_kinds"] = new_v_kinds
    if new_seps:
        t["separators"] = new_seps
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
        raise RuntimeError(f"No JSON files in {labels_dir}")

    for i, js in enumerate(files):
        d = json.loads(js.read_text(encoding="utf-8"))
        outp = out_dir / js.name

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
        maps = _build_evidence_maps(img)

        tabs = d.get("tables", [])
        if not isinstance(tabs, list):
            tabs = []
        new_tabs: list[dict[str, Any]] = []
        for t in tabs:
            if not isinstance(t, dict):
                continue
            new_tabs.append(_rebuild_table(t, maps=maps, max_shift_frac=max_shift_frac))
        d["tables"] = new_tabs
        outp.write_text(json.dumps(d, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Constrained decoder refinement for row-separator labels.")
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
