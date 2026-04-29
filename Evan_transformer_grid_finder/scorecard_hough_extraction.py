"""Hough-line-based golf scorecard grid extraction.

Pipeline:
1) Preprocess image to enhance long grid lines.
2) Detect horizontal/vertical Hough line segments.
3) Merge segments into canonical table lines.
4) Find up to 2 table candidates and apply golf-specific priors.
5) Slice table cells into a matrix of PNG images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json

import cv2
import numpy as np

from scorecard_preprocessing import (
    PreprocessConfig,
    PreprocessResult,
    convert_pdf_to_png,
    load_image,
    preprocess_scorecard,
)


@dataclass
class HoughConfig:
    """Config for Hough line detection and table assembly."""

    rho: float = 1.0
    theta_deg: float = 1.0
    threshold: int = 66
    min_line_length_ratio: float = 0.26
    max_line_gap_ratio: float = 0.025
    enable_sensitive_hough_pass: bool = True
    sensitive_threshold_scale: float = 0.58
    sensitive_min_line_length_scale: float = 0.55
    sensitive_max_line_gap_scale: float = 2.2
    angle_tolerance_deg: float = 12.0
    merge_pos_tol_px: int = 12
    merge_gap_tol_px: int = 22
    min_table_area_ratio: float = 0.015
    max_tables: int = 2
    min_v_lines_per_table: int = 4
    min_h_lines_per_table: int = 4
    min_cell_w: int = 14
    min_cell_h: int = 14
    border_inset_px: int = 0
    # Keep strictness concentrated here: accepted axis lines must span a large
    # fraction of the table extent to avoid snapping to characters.
    min_axis_line_coverage: float = 0.44
    proj_peak_rel_thresh: float = 0.14
    proj_peak_window: int = 9
    proj_min_gap_px: int = 8
    split_gap_ratio: float = 1.50
    tiny_gap_ratio: float = 0.42
    cell_inset_px: int = 2
    min_refined_axis_coverage: float = 0.22
    insert_min_axis_coverage: float = 0.12
    coverage_weight: float = 0.60
    min_refined_x_coverage: float = 0.22
    min_refined_y_coverage: float = 0.15
    insert_min_x_coverage: float = 0.12
    insert_min_y_coverage: float = 0.08
    support_dilate_px: int = 3
    edge_search_ratio: float = 0.14
    edge_min_coverage: float = 0.12
    edge_min_strength_rel: float = 0.04
    edge_promote_gap_ratio: float = 0.05
    min_table_width_ratio: float = 0.20
    min_table_height_ratio: float = 0.20
    min_grid_x_lines: int = 5
    min_grid_y_lines: int = 5
    min_line_continuity: float = 0.12
    min_line_intersections: int = 2
    refine_bbox_pad_px: int = 8
    top_header_band_ratio: float = 0.12
    top_header_penalty: float = 0.60
    proj_clip_percentile: float = 98.5
    proj_baseline_window_ratio: float = 0.14
    proj_rel_weight: float = 0.65
    max_line_thickness_px: int = 8
    thick_line_min_intersections: int = 6
    enable_outlier_trim: bool = True
    use_coverage_filter: bool = True


@dataclass
class CanonicalLine:
    orientation: str  # "v" or "h"
    pos: int
    start: int
    end: int
    support: int


@dataclass
class TableGrid:
    table_id: int
    bbox: tuple[int, int, int, int]  # x, y, w, h
    x_lines: list[int] = field(default_factory=list)
    y_lines: list[int] = field(default_factory=list)
    hole_mode: str = "unknown"
    hole_column_indices: list[int] = field(default_factory=list)
    separator_column_index: Optional[int] = None


@dataclass
class ExtractionResult:
    source_image: Path
    output_dir: Path
    table_grids: list[TableGrid]
    written_cells: list[Path]
    cell_matrices: dict[int, list[list[Optional[Path]]]]
    preprocess: PreprocessResult


def _detect_hough_segments(
    mask: np.ndarray,
    cfg: HoughConfig,
    threshold_scale: float = 1.0,
    min_len_scale: float = 1.0,
    max_gap_scale: float = 1.0,
) -> np.ndarray:
    h, w = mask.shape[:2]
    lines = cv2.HoughLinesP(
        mask,
        rho=cfg.rho,
        theta=np.deg2rad(cfg.theta_deg),
        threshold=max(20, int(round(cfg.threshold * threshold_scale))),
        minLineLength=max(
            12, int(round(cfg.min_line_length_ratio * min_len_scale * max(h, w)))
        ),
        maxLineGap=max(
            3, int(round(cfg.max_line_gap_ratio * max_gap_scale * max(h, w)))
        ),
    )
    if lines is None:
        return np.empty((0, 4), dtype=np.int32)
    return lines[:, 0, :].astype(np.int32)


def _classify_segments(
    segments: np.ndarray, angle_tolerance_deg: float
) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    vertical = []
    horizontal = []
    for x1, y1, x2, y2 in segments:
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        # 0 deg => horizontal, 90 deg => vertical
        if angle <= angle_tolerance_deg or abs(angle - 180.0) <= angle_tolerance_deg:
            horizontal.append((x1, y1, x2, y2))
        elif abs(angle - 90.0) <= angle_tolerance_deg:
            vertical.append((x1, y1, x2, y2))
    return vertical, horizontal


def _merge_1d_lines(
    segments: list[tuple[int, int, int, int]],
    orientation: str,
    pos_tol: int,
    gap_tol: int,
) -> list[CanonicalLine]:
    if not segments:
        return []

    # Convert each segment to (pos, start, end).
    items = []
    for x1, y1, x2, y2 in segments:
        if orientation == "v":
            pos = int(round((x1 + x2) * 0.5))
            start = int(min(y1, y2))
            end = int(max(y1, y2))
        else:
            pos = int(round((y1 + y2) * 0.5))
            start = int(min(x1, x2))
            end = int(max(x1, x2))
        if end <= start:
            continue
        items.append((pos, start, end))
    if not items:
        return []

    items.sort(key=lambda t: (t[0], t[1], t[2]))

    # Cluster by near-equal position.
    clusters: list[list[tuple[int, int, int]]] = []
    for it in items:
        if not clusters:
            clusters.append([it])
            continue
        cpos = int(round(np.mean([x[0] for x in clusters[-1]])))
        if abs(it[0] - cpos) <= pos_tol:
            clusters[-1].append(it)
        else:
            clusters.append([it])

    merged: list[CanonicalLine] = []
    for cluster in clusters:
        pos = int(round(np.mean([x[0] for x in cluster])))
        spans = sorted([(x[1], x[2]) for x in cluster], key=lambda s: s[0])
        cur_s, cur_e = spans[0]
        support = 0
        for s, e in spans[1:]:
            if s <= cur_e + gap_tol:
                cur_e = max(cur_e, e)
            else:
                merged.append(
                    CanonicalLine(
                        orientation=orientation,
                        pos=pos,
                        start=cur_s,
                        end=cur_e,
                        support=cur_e - cur_s + support,
                    )
                )
                support = 0
                cur_s, cur_e = s, e
        merged.append(
            CanonicalLine(
                orientation=orientation,
                pos=pos,
                start=cur_s,
                end=cur_e,
                support=cur_e - cur_s + support,
            )
        )

    # Keep strongest lines if duplicates remain near same position.
    merged.sort(key=lambda ln: (ln.pos, -(ln.end - ln.start)))
    dedup: list[CanonicalLine] = []
    for ln in merged:
        if dedup and abs(ln.pos - dedup[-1].pos) <= max(1, pos_tol // 2):
            if (ln.end - ln.start) > (dedup[-1].end - dedup[-1].start):
                dedup[-1] = ln
        else:
            dedup.append(ln)
    return dedup


def _draw_lines(shape: tuple[int, int], v_lines: list[CanonicalLine], h_lines: list[CanonicalLine]) -> np.ndarray:
    h, w = shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    for ln in v_lines:
        cv2.line(canvas, (ln.pos, ln.start), (ln.pos, ln.end), 255, 2)
    for ln in h_lines:
        cv2.line(canvas, (ln.start, ln.pos), (ln.end, ln.pos), 255, 2)
    return canvas


def _filter_frame_like_lines(
    shape: tuple[int, int],
    v_lines: list[CanonicalLine],
    h_lines: list[CanonicalLine],
) -> tuple[list[CanonicalLine], list[CanonicalLine]]:
    """Remove near-page-border frame lines that can merge unrelated regions."""

    h, w = shape
    out_v: list[CanonicalLine] = []
    out_h: list[CanonicalLine] = []
    for ln in v_lines:
        length = ln.end - ln.start
        near_edge = ln.pos < int(0.03 * w) or ln.pos > int(0.97 * w)
        if near_edge and length >= int(0.90 * h):
            continue
        out_v.append(ln)
    for ln in h_lines:
        length = ln.end - ln.start
        near_edge = ln.pos < int(0.05 * h) or ln.pos > int(0.95 * h)
        if near_edge and length >= int(0.90 * w):
            continue
        out_h.append(ln)
    return out_v, out_h


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / float(max(1, union))


def _dominant_subbbox_from_h_coverage(
    bbox: tuple[int, int, int, int],
    v_lines: list[CanonicalLine],
    h_lines: list[CanonicalLine],
    image_w: int,
) -> Optional[tuple[int, int, int, int]]:
    """Split a very wide merged component and keep the stronger grid side."""

    x, y, bw, bh = bbox
    if bw < int(0.60 * image_w) or bw < 260:
        return None

    cov_h = np.zeros(max(1, bw), dtype=np.float32)
    cov_v = np.zeros(max(1, bw), dtype=np.float32)
    x_end = x + bw
    y_end = y + bh
    for ln in h_lines:
        if ln.pos < y or ln.pos > y_end:
            continue
        lo = max(x, ln.start)
        hi = min(x_end, ln.end)
        if hi <= lo:
            continue
        # Weight by coverage ratio so huge top/bottom bars are less dominant.
        w = float(max(1, hi - lo)) / float(max(1, bw))
        cov_h[lo - x : hi - x + 1] += w
    for ln in v_lines:
        if ln.pos < x or ln.pos > x_end:
            continue
        oy0 = max(y, ln.start)
        oy1 = min(y_end, ln.end)
        span = max(0, oy1 - oy0)
        if span <= 0:
            continue
        ratio = float(span) / float(max(1, bh))
        if ratio < 0.20:
            continue
        idx = int(np.clip(ln.pos - x, 0, max(0, bw - 1)))
        lo = max(0, idx - 2)
        hi = min(bw, idx + 3)
        cov_v[lo:hi] += ratio

    if cov_h.size < 40:
        return None
    if float(cov_h.max()) <= 0.0 and float(cov_v.max()) <= 0.0:
        return None
    h_norm = cov_h / float(max(1e-6, float(cov_h.max())))
    v_norm = cov_v / float(max(1e-6, float(cov_v.max()))) if float(cov_v.max()) > 0 else cov_v
    cov = 0.30 * h_norm + 0.70 * v_norm

    win = max(15, int(round(0.04 * cov.size)) | 1)
    sm = _smooth_1d(cov.astype(np.float32), win)
    if float(sm.max()) <= 0.0:
        return None

    lo = int(round(0.20 * cov.size))
    hi = int(round(0.80 * cov.size))
    if hi - lo < 20:
        return None
    seg = sm[lo:hi]
    cut = int(lo + int(np.argmin(seg)))
    if cut <= int(0.18 * cov.size) or cut >= int(0.82 * cov.size):
        return None

    valley = float(sm[cut])
    peak = float(sm.max())
    if valley > 0.58 * peak:
        return None

    left_mean = float(np.mean(sm[:cut])) if cut > 0 else 0.0
    right_mean = float(np.mean(sm[cut:])) if cut < sm.size else 0.0
    mx = max(left_mean, right_mean, 1e-6)
    mn = min(left_mean, right_mean)
    # Only split when one side is clearly weaker (grid + non-grid merge case).
    if mn >= 0.62 * mx:
        return None

    pad = max(8, int(round(0.01 * bw)))
    left_bbox = (x, y, max(1, (x + cut - pad) - x), bh)
    right_bbox = (x + cut + pad, y, max(1, (x + bw) - (x + cut + pad)), bh)
    if left_bbox[2] < max(140, int(round(0.22 * bw))) or right_bbox[2] < max(
        140, int(round(0.22 * bw))
    ):
        return None

    def _side_metrics(sb: tuple[int, int, int, int]) -> tuple[float, int, int, int]:
        sx, sy, sw, sh = sb
        sx1 = sx + sw
        sy1 = sy + sh
        min_v_span = int(round(0.40 * sh))
        min_h_span = int(round(0.40 * sw))
        v_sel = [
            ln
            for ln in v_lines
            if sx <= ln.pos <= sx1
            and not (ln.end < sy or ln.start > sy1)
            and (ln.end - ln.start) >= min_v_span
        ]
        h_sel = []
        for ln in h_lines:
            if not (sy <= ln.pos <= sy1):
                continue
            ov = max(0, min(sx1, ln.end) - max(sx, ln.start))
            if ov >= min_h_span:
                h_sel.append(ln)
        inter = 0
        for vl in v_sel:
            for hl in h_sel:
                if hl.start <= vl.pos <= hl.end and vl.start <= hl.pos <= vl.end:
                    inter += 1
        score = 2.0 * len(v_sel) + 1.5 * len(h_sel) + 0.08 * float(inter)
        return score, len(v_sel), len(h_sel), inter

    left_score, lv, lh, li = _side_metrics(left_bbox)
    right_score, rv, rh, ri = _side_metrics(right_bbox)
    if max(left_score, right_score) < 12.0:
        return None
    if lv >= 8 and lh >= 6 and rv >= 8 and rh >= 6:
        return None
    # If both sides are comparably strong, this is likely a legitimate
    # two-block table (e.g., 18-hole with center separator), so keep whole bbox.
    if min(left_score, right_score) >= 0.70 * max(left_score, right_score) and min(
        left_score, right_score
    ) >= 18.0:
        return None
    split_ratio = 1.55
    if left_score >= split_ratio * right_score:
        return left_bbox
    if right_score >= split_ratio * left_score:
        return right_bbox
    return None


def _select_table_candidates(
    line_canvas: np.ndarray,
    v_lines: list[CanonicalLine],
    h_lines: list[CanonicalLine],
    cfg: HoughConfig,
) -> list[tuple[int, int, int, int]]:
    h, w = line_canvas.shape[:2]
    img_area = float(h * w)

    # Connect nearby lines into components.
    k = max(3, int(round(0.008 * max(h, w))) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    conn = cv2.morphologyEx(line_canvas, cv2.MORPH_CLOSE, kernel, iterations=1)
    conn = cv2.dilate(conn, kernel, iterations=1)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(conn, connectivity=8)
    cands: list[tuple[int, int, int, int, int]] = []
    for i in range(1, n_labels):
        x, y, bw, bh, area = stats[i]
        if area < cfg.min_table_area_ratio * img_area:
            continue
        if bw < 0.15 * w or bh < 0.10 * h:
            continue
        if bw < cfg.min_table_width_ratio * w or bh < cfg.min_table_height_ratio * h:
            continue

        cand_bbox = (x, y, bw, bh)
        for _ in range(2):
            dominant = _dominant_subbbox_from_h_coverage(cand_bbox, v_lines, h_lines, w)
            if dominant is None:
                break
            # Stop if split is too small to matter.
            if dominant[2] >= int(0.94 * cand_bbox[2]):
                break
            cand_bbox = dominant
        x, y, bw, bh = cand_bbox

        xv = sum(
            1
            for ln in v_lines
            if x <= ln.pos <= x + bw and not (ln.end < y or ln.start > y + bh)
        )
        yh = sum(
            1
            for ln in h_lines
            if y <= ln.pos <= y + bh and not (ln.end < x or ln.start > x + bw)
        )
        if xv < cfg.min_v_lines_per_table or yh < cfg.min_h_lines_per_table:
            continue
        cands.append((x, y, bw, bh, area))

    if not cands:
        return []

    cands.sort(key=lambda t: t[4], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for x, y, bw, bh, _ in cands:
        b = (x, y, bw, bh)
        if any(_bbox_iou(b, s) > 0.4 for s in selected):
            continue
        selected.append(b)
        if len(selected) >= cfg.max_tables:
            break
    return selected


def _ensure_border_lines(values: list[int], low: int, high: int, tol: int = 8) -> list[int]:
    out = sorted(set(int(v) for v in values))
    if not out:
        return [low, high]
    if abs(out[0] - low) > tol:
        out.insert(0, low)
    else:
        out[0] = low
    if abs(out[-1] - high) > tol:
        out.append(high)
    else:
        out[-1] = high
    return out


def _prune_near_duplicate_positions(values: list[int], min_gap: int) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    out = [values[0]]
    for v in values[1:]:
        if v - out[-1] >= min_gap:
            out.append(v)
    return out


def _smooth_1d(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    if signal.size < window:
        return signal.astype(np.float32, copy=True)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def _projection_peaks(
    signal: np.ndarray,
    rel_thresh: float,
    min_gap: int,
    coverage: Optional[np.ndarray] = None,
    min_coverage: float = 0.0,
    coverage_weight: float = 0.0,
) -> list[int]:
    if signal.size < 3:
        return []
    mx = float(np.max(signal))
    if mx <= 0:
        return []
    thr = rel_thresh * mx
    cands: list[tuple[float, int]] = []
    for i in range(1, signal.size - 1):
        s = float(signal[i])
        if s < thr:
            continue
        cov = 1.0
        if coverage is not None and 0 <= i < coverage.size:
            cov = float(coverage[i])
            if cov < min_coverage:
                continue
        if s >= float(signal[i - 1]) and s >= float(signal[i + 1]):
            score = s * (1.0 + coverage_weight * cov)
            cands.append((score, i))
    if not cands:
        return []
    cands.sort(key=lambda t: t[0], reverse=True)
    chosen: list[int] = []
    for _, idx in cands:
        if all(abs(idx - c) >= min_gap for c in chosen):
            chosen.append(idx)
    return sorted(chosen)


def _prune_by_strength(
    positions: list[int],
    low: int,
    high: int,
    strength: np.ndarray,
    min_gap: int,
    coverage: Optional[np.ndarray] = None,
    coverage_weight: float = 0.0,
) -> list[int]:
    """Prune near-duplicate lines by keeping the stronger projection response."""

    if not positions:
        return [low, high]

    vals = sorted(set(int(v) for v in positions if low <= v <= high))
    if low not in vals:
        vals.insert(0, low)
    if high not in vals:
        vals.append(high)

    kept = [vals[0]]
    for v in vals[1:]:
        prev = kept[-1]
        if v - prev < min_gap and v not in (low, high) and prev not in (low, high):
            s_prev = float(strength[prev]) if 0 <= prev < strength.size else 0.0
            s_v = float(strength[v]) if 0 <= v < strength.size else 0.0
            if coverage is not None:
                c_prev = float(coverage[prev]) if 0 <= prev < coverage.size else 0.0
                c_v = float(coverage[v]) if 0 <= v < coverage.size else 0.0
                s_prev *= 1.0 + coverage_weight * c_prev
                s_v *= 1.0 + coverage_weight * c_v
            if s_v > s_prev:
                kept[-1] = v
        else:
            kept.append(v)
    if kept[0] != low:
        kept[0] = low
    if kept[-1] != high:
        kept[-1] = high
    return kept


def _fill_missing_lines(
    positions: list[int],
    strength: np.ndarray,
    coverage: Optional[np.ndarray],
    low: int,
    high: int,
    min_gap: int,
    split_gap_ratio: float,
    coverage_weight: float,
    min_insert_coverage: float,
) -> list[int]:
    """Insert likely missing lines when a gap is much larger than typical."""

    if len(positions) < 2:
        return [low, high]

    out = sorted(set(int(v) for v in positions if low <= v <= high))
    if out[0] != low:
        out.insert(0, low)
    if out[-1] != high:
        out.append(high)

    widths = np.diff(np.array(out, dtype=np.int32))
    if widths.size == 0:
        return out
    med = float(np.median(widths))
    if med <= 1.0:
        return out

    inserts: list[int] = []
    for a, b in zip(out[:-1], out[1:]):
        gap = b - a
        if gap <= split_gap_ratio * med:
            continue

        target_new = max(1, int(round(gap / med)) - 1)
        lo = max(a + min_gap, low)
        hi = min(b - min_gap, high)
        if hi <= lo:
            continue
        local = strength[lo : hi + 1]
        local_cov = coverage[lo : hi + 1] if coverage is not None else None
        local_peaks = _projection_peaks(
            local,
            rel_thresh=0.45,
            min_gap=min_gap,
            coverage=local_cov,
            min_coverage=min_insert_coverage,
            coverage_weight=coverage_weight,
        )
        local_peaks = sorted(
            local_peaks,
            key=lambda i: float(local[i])
            * (
                1.0
                + coverage_weight
                * (float(local_cov[i]) if local_cov is not None and 0 <= i < local_cov.size else 0.0)
            ),
            reverse=True,
        )
        chosen = [lo + i for i in local_peaks[:target_new]]

        if not chosen:
            step = gap / float(target_new + 1)
            chosen = [int(round(a + step * (k + 1))) for k in range(target_new)]

        for c in chosen:
            if lo <= c <= hi:
                inserts.append(c)

    if inserts:
        out.extend(inserts)
        out = sorted(set(out))
    return out


def _axis_coverage_profile(binary_crop: np.ndarray, axis: str) -> np.ndarray:
    """Coverage ratio per axis position (fraction of rows/cols containing line ink)."""

    if binary_crop.size == 0:
        return np.zeros(0, dtype=np.float32)
    if axis == "x":
        dil = cv2.dilate(
            binary_crop,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)),
            iterations=1,
        )
        return (dil > 0).mean(axis=0).astype(np.float32)
    if axis == "y":
        dil = cv2.dilate(
            binary_crop,
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)),
            iterations=1,
        )
        return (dil > 0).mean(axis=1).astype(np.float32)
    raise ValueError("axis must be 'x' or 'y'")


def _axis_support_profile_from_lines(
    lines: list[CanonicalLine],
    low: int,
    high: int,
    span_norm: int,
    dilate_px: int = 0,
) -> np.ndarray:
    """Per-axis profile from Hough line span length (uses explicit line length)."""

    n = max(0, high - low + 1)
    prof = np.zeros(n, dtype=np.float32)
    if n == 0 or span_norm <= 0:
        return prof
    d = max(0, int(dilate_px))
    denom = float(max(1, span_norm))
    for ln in lines:
        idx = int(ln.pos) - low
        if idx < -d or idx >= n + d:
            continue
        ratio = float(max(0, ln.end - ln.start)) / denom
        ratio = float(np.clip(ratio, 0.0, 1.0))
        lo = max(0, idx - d)
        hi = min(n, idx + d + 1)
        if lo < hi:
            prof[lo:hi] = np.maximum(prof[lo:hi], ratio)
    return prof


def _line_continuity(
    line_mask: np.ndarray,
    line: CanonicalLine,
    band_px: int = 1,
) -> float:
    """Estimate continuity of a canonical line directly from the binary mask."""

    h, w = line_mask.shape[:2]
    b = max(0, int(band_px))
    if line.orientation == "v":
        x0 = max(0, line.pos - b)
        x1 = min(w, line.pos + b + 1)
        y0 = max(0, line.start)
        y1 = min(h, line.end + 1)
    else:
        y0 = max(0, line.pos - b)
        y1 = min(h, line.pos + b + 1)
        x0 = max(0, line.start)
        x1 = min(w, line.end + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    sl = line_mask[y0:y1, x0:x1]
    if sl.size == 0:
        return 0.0
    return float((sl > 0).mean())


def _line_thickness_estimate(
    line_mask: np.ndarray,
    line: CanonicalLine,
    n_samples: int = 17,
) -> float:
    """Estimate stroke thickness orthogonal to the line direction."""

    h, w = line_mask.shape[:2]
    n = max(5, int(n_samples))
    thicknesses: list[int] = []
    if line.orientation == "h":
        y = int(np.clip(line.pos, 0, h - 1))
        x0 = int(np.clip(line.start, 0, w - 1))
        x1 = int(np.clip(line.end, 0, w - 1))
        if x1 < x0:
            x0, x1 = x1, x0
        xs = np.linspace(x0, x1, n).astype(np.int32)
        for x in xs:
            if line_mask[y, x] == 0:
                continue
            y0 = y
            y1 = y
            while y0 > 0 and line_mask[y0 - 1, x] > 0:
                y0 -= 1
            while y1 + 1 < h and line_mask[y1 + 1, x] > 0:
                y1 += 1
            thicknesses.append(y1 - y0 + 1)
    else:
        x = int(np.clip(line.pos, 0, w - 1))
        y0 = int(np.clip(line.start, 0, h - 1))
        y1 = int(np.clip(line.end, 0, h - 1))
        if y1 < y0:
            y0, y1 = y1, y0
        ys = np.linspace(y0, y1, n).astype(np.int32)
        for y in ys:
            if line_mask[y, x] == 0:
                continue
            x0 = x
            x1 = x
            while x0 > 0 and line_mask[y, x0 - 1] > 0:
                x0 -= 1
            while x1 + 1 < w and line_mask[y, x1 + 1] > 0:
                x1 += 1
            thicknesses.append(x1 - x0 + 1)

    if not thicknesses:
        return 0.0
    return float(np.median(np.asarray(thicknesses, dtype=np.float32)))


def _intersection_counts(
    v_lines: list[CanonicalLine],
    h_lines: list[CanonicalLine],
) -> tuple[list[int], list[int]]:
    v_hits = [0 for _ in v_lines]
    h_hits = [0 for _ in h_lines]
    for i, vl in enumerate(v_lines):
        for j, hl in enumerate(h_lines):
            if hl.start <= vl.pos <= hl.end and vl.start <= hl.pos <= vl.end:
                v_hits[i] += 1
                h_hits[j] += 1
    return v_hits, h_hits


def _robust_projection_strength(
    strength: np.ndarray,
    clip_percentile: float,
    baseline_window_ratio: float,
    rel_weight: float,
) -> np.ndarray:
    """Suppress dominant outliers so weak but regular grid lines remain visible."""

    s = np.asarray(strength, dtype=np.float32)
    if s.size == 0:
        return s
    clip_p = float(np.clip(clip_percentile, 90.0, 100.0))
    clip_v = float(np.percentile(s, clip_p))
    if clip_v > 0:
        s = np.minimum(s, clip_v)

    n = int(s.size)
    win = max(11, int(round(float(np.clip(baseline_window_ratio, 0.04, 0.40)) * n)) | 1)
    base = _smooth_1d(s, win)
    eps = 1e-6
    rel = np.maximum(0.0, (s / np.maximum(base, eps)) - 1.0)
    absn = s / float(max(eps, float(np.max(s))))
    w_rel = float(np.clip(rel_weight, 0.0, 1.0))
    out = (w_rel * rel) + ((1.0 - w_rel) * absn)
    return out.astype(np.float32, copy=False)


def _draw_orientation_maps_for_bbox(
    shape: tuple[int, int],
    bx0: int,
    by0: int,
    bx1: int,
    by1: int,
    v_lines: list[CanonicalLine],
    h_lines: list[CanonicalLine],
) -> tuple[np.ndarray, np.ndarray]:
    """Render vertical-only and horizontal-only maps for a bbox."""

    ch, cw = shape
    v_map = np.zeros((ch, cw), dtype=np.uint8)
    h_map = np.zeros((ch, cw), dtype=np.uint8)

    for ln in v_lines:
        if ln.orientation != "v":
            continue
        x = int(ln.pos) - bx0
        if x < 0 or x >= cw:
            continue
        y0 = max(by0, int(ln.start)) - by0
        y1 = min(by1, int(ln.end)) - by0
        if y1 < y0:
            continue
        cv2.line(v_map, (x, y0), (x, y1), 255, 1)

    for ln in h_lines:
        if ln.orientation != "h":
            continue
        y = int(ln.pos) - by0
        if y < 0 or y >= ch:
            continue
        x0 = max(bx0, int(ln.start)) - bx0
        x1 = min(bx1, int(ln.end)) - bx0
        if x1 < x0:
            continue
        cv2.line(h_map, (x0, y), (x1, y), 255, 1)

    return v_map, h_map


def _edge_border_candidate(
    strength: np.ndarray,
    coverage: np.ndarray,
    side: str,
    cfg: HoughConfig,
) -> Optional[int]:
    n = int(strength.size)
    if n <= 0:
        return None
    win = max(8, int(round(cfg.edge_search_ratio * n)))
    win = min(n, win)
    if side == "left":
        idxs = range(0, win)
    elif side == "right":
        idxs = range(max(0, n - win), n)
    else:
        raise ValueError("side must be 'left' or 'right'")

    max_s = float(np.max(strength))
    if max_s <= 0:
        return None
    best_idx = None
    best_score = -1e9
    for i in idxs:
        cov = float(coverage[i]) if 0 <= i < coverage.size else 0.0
        if cov < cfg.edge_min_coverage:
            continue
        s_rel = float(strength[i]) / max_s
        if s_rel < cfg.edge_min_strength_rel:
            continue
        if side == "left":
            edge_bonus = (win - i) / float(max(1, win))
        else:
            edge_bonus = (i - (n - win)) / float(max(1, win))
        score = s_rel * (1.0 + cfg.coverage_weight * cov) + 0.20 * edge_bonus
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _repair_outer_borders(
    lines: list[int],
    strength: np.ndarray,
    coverage: np.ndarray,
    low: int,
    high: int,
    cfg: HoughConfig,
) -> list[int]:
    """Promote strong edge candidates as borders when current borders are missing."""

    vals = sorted(set(int(v) for v in lines if low <= v <= high))
    if not vals:
        vals = [low, high]
    left = vals[0]
    right = vals[-1]

    cand_l = _edge_border_candidate(strength, coverage, side="left", cfg=cfg)
    cand_r = _edge_border_candidate(strength, coverage, side="right", cfg=cfg)
    promote_gap = max(3, int(round(cfg.edge_promote_gap_ratio * max(1, high - low + 1))))

    if cand_l is not None and cand_l < left - promote_gap:
        left = cand_l
    if cand_r is not None and cand_r > right + promote_gap:
        right = cand_r
    if right <= left:
        left, right = low, high

    out = [v for v in vals if left <= v <= right]
    out = sorted(set(out + [left, right]))
    return out


def _trim_axis_outlier_clusters(
    lines: list[int],
    low: int,
    high: int,
) -> list[int]:
    """Trim tiny outlier clusters when one dominant grid cluster exists."""

    vals = sorted(set(int(v) for v in lines if low <= v <= high))
    if len(vals) < 4:
        return vals
    gaps = np.diff(np.array(vals, dtype=np.int32))
    if gaps.size == 0:
        return vals
    med = float(np.median(gaps))
    if med <= 1.0:
        return vals
    break_thr = max(int(round(2.6 * med)), int(round(0.12 * max(1, high - low + 1))))
    clusters: list[list[int]] = [[vals[0]]]
    for v, g in zip(vals[1:], gaps):
        if int(g) > break_thr:
            clusters.append([v])
        else:
            clusters[-1].append(v)
    if len(clusters) <= 1:
        return vals

    big_clusters = [c for c in clusters if len(c) >= 4]
    if len(big_clusters) >= 2:
        # Likely true multi-block structure; keep everything.
        return vals

    clusters_sorted = sorted(clusters, key=lambda c: len(c), reverse=True)
    top = clusters_sorted[0]
    second_len = len(clusters_sorted[1]) if len(clusters_sorted) > 1 else 0
    # Keep tiny stray clusters trimmed, but preserve meaningful secondary
    # row groups (common in scorecards with lower summary sections).
    if len(top) >= max(5, 3 * second_len) and second_len <= 2:
        return sorted(set(top))
    return vals


def _filter_by_coverage(
    positions: list[int],
    low: int,
    high: int,
    coverage: np.ndarray,
    min_coverage: float,
    keep: Optional[set[int]] = None,
) -> list[int]:
    keep = keep or set()
    vals = sorted(set(int(v) for v in positions if low <= v <= high))
    if not vals:
        return [low, high]
    if vals[0] != low:
        vals.insert(0, low)
    if vals[-1] != high:
        vals.append(high)

    out = [vals[0]]
    for v in vals[1:-1]:
        if v in keep:
            out.append(v)
            continue
        cov = float(coverage[v]) if 0 <= v < coverage.size else 0.0
        if cov >= min_coverage:
            out.append(v)
    out.append(vals[-1])
    return sorted(set(out))


def _refine_axis_lines(
    positions: list[int],
    low: int,
    high: int,
    strength_1d: np.ndarray,
    coverage_1d: np.ndarray,
    min_refined_coverage: float,
    min_insert_coverage: float,
    cfg: HoughConfig,
) -> list[int]:
    min_gap = max(cfg.proj_min_gap_px, cfg.merge_pos_tol_px // 2)
    anchor = set(int(v) for v in positions if low <= v <= high)

    sm = _smooth_1d(strength_1d, cfg.proj_peak_window)
    peak_idx = _projection_peaks(
        sm,
        cfg.proj_peak_rel_thresh,
        min_gap,
        coverage=coverage_1d,
        min_coverage=min_refined_coverage,
        coverage_weight=cfg.coverage_weight,
    )
    peaks = [low + int(i) for i in peak_idx]

    lines = sorted(set(positions + peaks))
    lines = _ensure_border_lines(lines, low, high)
    lines = _prune_by_strength(
        lines,
        low,
        high,
        sm,
        min_gap,
        coverage=coverage_1d,
        coverage_weight=cfg.coverage_weight,
    )
    lines = _fill_missing_lines(
        lines,
        sm,
        coverage_1d,
        low,
        high,
        min_gap=min_gap,
        split_gap_ratio=cfg.split_gap_ratio,
        coverage_weight=cfg.coverage_weight,
        min_insert_coverage=min_insert_coverage,
    )
    if cfg.use_coverage_filter:
        lines = _filter_by_coverage(
            lines,
            low,
            high,
            coverage_1d,
            min_coverage=min_refined_coverage,
            keep=anchor,
        )

    # Final duplicate pruning with dynamic gap based on median spacing.
    if len(lines) >= 3:
        widths = np.diff(np.array(lines, dtype=np.int32))
        med = float(np.median(widths))
        dynamic_gap = max(min_gap, int(round(cfg.tiny_gap_ratio * med)))
        lines = _prune_by_strength(
            lines,
            low,
            high,
            sm,
            dynamic_gap,
            coverage=coverage_1d,
            coverage_weight=cfg.coverage_weight,
        )
        if cfg.use_coverage_filter:
            lines = _filter_by_coverage(
                lines,
                low,
                high,
                coverage_1d,
                min_coverage=min_refined_coverage,
                keep=anchor,
            )
    lines = sorted(set(lines))
    lines = _ensure_border_lines(lines, low, high)
    return lines


def _infer_golf_structure(x_lines: list[int], table_count: int) -> tuple[str, list[int], Optional[int]]:
    """Infer hole layout without OCR using golf scorecard priors."""

    n_cols = max(0, len(x_lines) - 1)
    if n_cols == 0:
        return "unknown", [], None

    # If two distinct tables exist, usually each corresponds to a 9-hole block.
    if table_count == 2:
        hole_cols = [i for i in range(1, min(n_cols, 10))]
        return "two_9hole_tables", hole_cols, None

    # One table case: infer 9-hole vs 18-hole variants.
    if n_cols <= 13:
        hole_cols = [i for i in range(1, min(n_cols, 10))]
        return "single_9hole_table", hole_cols, None

    # 9-hole with trailing non-grid/info columns can appear as a wider table.
    widths = np.diff(np.array(x_lines, dtype=np.int32))
    if widths.size > 0:
        p9 = _estimate_axis_pitch(widths)
        if p9 is not None:
            run_s, run_e, run_len = _longest_regular_gap_run(widths, p9)
            trailing = n_cols - (run_e + 1) - 1 if run_e >= 0 else 0
            if 8 <= run_len <= 12 and run_s <= 2 and trailing >= 3:
                hole_cols = [i for i in range(1, min(n_cols, 10))]
                return "single_9hole_table", hole_cols, None

    # Candidate 18-hole single table.
    if widths.size == 0:
        return "single_18hole_table", [], None

    med = float(np.median(widths))
    sep_idx = None
    center = n_cols // 2
    for i, w in enumerate(widths):
        # Initials separator often appears near the center and deviates in width.
        if abs(i - center) <= 2 and (w > 1.45 * med or w < 0.6 * med):
            sep_idx = i
            break

    if sep_idx is not None:
        # Hole columns are first 9 after label column and last 9 after separator.
        front = [i for i in range(1, min(sep_idx, 10))]
        back_start = sep_idx + 1
        back = list(range(back_start, min(back_start + 9, n_cols)))
        return "single_18hole_with_separator", front + back, sep_idx

    # Continuous 18-hole table, likely label + 18 hole columns (+ summary cols).
    hole_cols = [i for i in range(1, min(n_cols, 19))]
    return "single_18hole_continuous", hole_cols, None


def _estimate_axis_pitch(gaps: np.ndarray) -> Optional[float]:
    vals = np.asarray(gaps, dtype=np.float32)
    vals = vals[vals > 0]
    if vals.size < 3:
        return None
    vals = np.sort(vals)
    take = max(3, int(round(0.60 * vals.size)))
    base = vals[:take]
    p = float(np.median(base))
    if p <= 1.0:
        return None
    return p


def _longest_regular_gap_run(
    gaps: np.ndarray,
    pitch: float,
    lo_ratio: float = 0.72,
    hi_ratio: float = 1.36,
) -> tuple[int, int, int]:
    """Return (start_gap_idx, end_gap_idx, length) for the longest regular run."""

    vals = np.asarray(gaps, dtype=np.float32)
    if vals.size == 0 or pitch <= 1.0:
        return -1, -1, 0
    lo = float(lo_ratio) * float(pitch)
    hi = float(hi_ratio) * float(pitch)
    ok = (vals >= lo) & (vals <= hi)
    best_s = -1
    best_e = -1
    best_len = 0
    cur_s = -1
    for i, good in enumerate(ok.tolist()):
        if good:
            if cur_s < 0:
                cur_s = i
        else:
            if cur_s >= 0:
                ln = i - cur_s
                if ln > best_len:
                    best_s, best_e, best_len = cur_s, i - 1, ln
                cur_s = -1
    if cur_s >= 0:
        ln = len(ok) - cur_s
        if ln > best_len:
            best_s, best_e, best_len = cur_s, len(ok) - 1, ln
    return best_s, best_e, best_len


def _repair_9hole_x_lines(xs: list[int]) -> list[int]:
    vals = sorted(set(int(v) for v in xs))
    if len(vals) < 3:
        return vals

    gaps = np.diff(np.array(vals, dtype=np.int32))
    p = _estimate_axis_pitch(gaps)
    if p is None:
        p = float(np.median(gaps)) if gaps.size > 0 else None
    if p is None or p <= 1.0:
        return vals

    # Remove a likely spurious split in the left label column.
    if gaps.size >= 2 and gaps[0] <= 1.4 * p and gaps[1] >= 2.2 * p and len(vals) >= 8:
        vals.pop(1)
        gaps = np.diff(np.array(vals, dtype=np.int32))
        p2 = _estimate_axis_pitch(gaps)
        if p2 is not None:
            p = p2

    # After label + tee columns, enforce hole-column regularity.
    inserts: list[int] = []
    gaps = np.diff(np.array(vals, dtype=np.int32))
    start_idx = 0 if len(vals) <= 8 else 2
    for i, g in enumerate(gaps):
        if i < start_idx:
            continue
        if g <= 1.75 * p:
            continue
        n_new = max(1, int(round(float(g) / p)) - 1)
        n_new = min(4, n_new)
        step = g / float(n_new + 1)
        a = vals[i]
        for k in range(n_new):
            inserts.append(int(round(a + step * (k + 1))))

    if inserts:
        vals.extend(inserts)
        vals = sorted(set(vals))
        gaps = np.diff(np.array(vals, dtype=np.int32))
        p2 = _estimate_axis_pitch(gaps)
        if p2 is not None:
            p = p2

    # If still severely under-detected, keep splitting the largest gaps.
    target_lines = 13
    if len(vals) < target_lines:
        for _ in range(24):
            if len(vals) >= target_lines:
                break
            gaps = np.diff(np.array(vals, dtype=np.int32))
            if gaps.size == 0:
                break
            i = int(np.argmax(gaps))
            g = int(gaps[i])
            if g < max(18, int(round(0.95 * p))):
                break
            mid = int(round((vals[i] + vals[i + 1]) * 0.5))
            if mid <= vals[i] or mid >= vals[i + 1]:
                break
            vals.insert(i + 1, mid)
            vals = sorted(set(vals))

    min_gap = max(12, int(round(0.45 * p)))
    vals = _prune_near_duplicate_positions(vals, min_gap)

    # Trim trailing non-grid columns when a dominant 9-hole run is present.
    if len(vals) > 13:
        gaps = np.diff(np.array(vals, dtype=np.int32))
        p3 = _estimate_axis_pitch(gaps)
        if p3 is not None:
            run_s, run_e, run_len = _longest_regular_gap_run(gaps, p3)
            if run_len >= 8 and run_s <= 2:
                # Gap run [run_s..run_e] maps to lines [run_s..run_e+1].
                start_line = max(0, run_s - 2)
                end_line = min(len(vals) - 1, run_e + 2)
                sub = vals[start_line : end_line + 1]
                if vals[0] not in sub:
                    sub = [vals[0]] + sub
                # Keep a compact 9-hole-style structure.
                if len(sub) > 13:
                    sub = sub[:13]
                if len(sub) >= 10:
                    vals = sorted(set(sub))
    return vals


def _repair_9hole_y_lines(ys: list[int]) -> list[int]:
    vals = sorted(set(int(v) for v in ys))
    if len(vals) < 8:
        return vals

    # Remove tiny-gap outliers that commonly come from text baselines.
    changed = True
    while changed and len(vals) >= 6:
        changed = False
        gaps = np.diff(np.array(vals, dtype=np.int32))
        p = _estimate_axis_pitch(gaps)
        if p is None:
            break
        tiny = 0.60 * p
        huge = 1.35 * p

        # tiny then huge => middle line is usually spurious (e.g., through text row).
        for i in range(gaps.size - 1):
            if gaps[i] < tiny and gaps[i + 1] > huge:
                vals.pop(i + 1)
                changed = True
                break
        if changed:
            continue

        # two consecutive tiny gaps => drop first interior line in the cluster.
        for i in range(gaps.size - 1):
            if gaps[i] < tiny and gaps[i + 1] < tiny:
                vals.pop(i + 1)
                changed = True
                break

    gaps = np.diff(np.array(vals, dtype=np.int32))
    p = _estimate_axis_pitch(gaps)
    if p is None:
        return vals

    inserts: list[int] = []
    gaps = np.diff(np.array(vals, dtype=np.int32))
    for i, g in enumerate(gaps):
        if g <= 1.75 * p:
            continue
        n_new = max(1, int(round(float(g) / p)) - 1)
        n_new = min(3, n_new)
        step = g / float(n_new + 1)
        a = vals[i]
        for k in range(n_new):
            inserts.append(int(round(a + step * (k + 1))))

    if inserts:
        vals.extend(inserts)
        vals = sorted(set(vals))
    min_gap = max(10, int(round(0.45 * p)))
    vals = _prune_near_duplicate_positions(vals, min_gap)
    return vals


def _apply_9hole_priors(xs: list[int], ys: list[int]) -> tuple[list[int], list[int]]:
    return _repair_9hole_x_lines(xs), _repair_9hole_y_lines(ys)


def _lines_for_bbox(
    bbox: tuple[int, int, int, int],
    v_lines: list[CanonicalLine],
    h_lines: list[CanonicalLine],
    line_mask: np.ndarray,
    cfg: HoughConfig,
) -> tuple[list[int], list[int]]:
    x, y, w, h = bbox
    H, W = line_mask.shape[:2]
    bx0 = x + cfg.border_inset_px
    by0 = y + cfg.border_inset_px
    bx1 = x + w - cfg.border_inset_px
    by1 = y + h - cfg.border_inset_px
    bx0 = max(0, min(W - 1, bx0))
    bx1 = max(0, min(W - 1, bx1))
    by0 = max(0, min(H - 1, by0))
    by1 = max(0, min(H - 1, by1))
    if bx1 <= bx0:
        bx0, bx1 = 0, max(0, W - 1)
    if by1 <= by0:
        by0, by1 = 0, max(0, H - 1)

    raw_v_lines = [
        ln
        for ln in v_lines
        if bx0 <= ln.pos <= bx1 and not (ln.end < by0 or ln.start > by1)
    ]
    raw_h_lines = [
        ln
        for ln in h_lines
        if by0 <= ln.pos <= by1 and not (ln.end < bx0 or ln.start > bx1)
    ]
    span_h = max(1, by1 - by0 + 1)
    span_w = max(1, bx1 - bx0 + 1)
    anchor_cov = max(0.24, 0.72 * float(cfg.min_axis_line_coverage))
    long_v_lines = [
        ln
        for ln in raw_v_lines
        if (ln.end - ln.start) >= int(round(anchor_cov * span_h))
    ]
    long_h_lines = [
        ln
        for ln in raw_h_lines
        if (ln.end - ln.start) >= int(round(anchor_cov * span_w))
    ]
    if not long_v_lines:
        long_v_lines = list(raw_v_lines)
    if not long_h_lines:
        long_h_lines = list(raw_h_lines)

    # Remove long spans that are mostly disconnected character fragments.
    long_v_core = [
        ln
        for ln in long_v_lines
        if _line_continuity(line_mask, ln, band_px=1) >= cfg.min_line_continuity
    ]
    long_h_core = [
        ln
        for ln in long_h_lines
        if _line_continuity(line_mask, ln, band_px=1) >= cfg.min_line_continuity
    ]
    if len(long_v_core) < cfg.min_v_lines_per_table:
        long_v_core = list(long_v_lines)
    if len(long_h_core) < cfg.min_h_lines_per_table:
        long_h_core = list(long_h_lines)

    # Thickness prior: suppress thick band artifacts unless heavily supported
    # by orthogonal intersections.
    if long_v_core and long_h_core:
        v_hits0, h_hits0 = _intersection_counts(long_v_core, long_h_core)
    else:
        v_hits0 = [0 for _ in long_v_core]
        h_hits0 = [0 for _ in long_h_core]
    v_keep_thin = []
    for ln, hits in zip(long_v_core, v_hits0):
        thick = _line_thickness_estimate(line_mask, ln)
        if thick <= cfg.max_line_thickness_px or hits >= cfg.thick_line_min_intersections:
            v_keep_thin.append(ln)
    h_keep_thin = []
    for ln, hits in zip(long_h_core, h_hits0):
        thick = _line_thickness_estimate(line_mask, ln)
        if thick <= cfg.max_line_thickness_px or hits >= cfg.thick_line_min_intersections:
            h_keep_thin.append(ln)
    if len(v_keep_thin) >= cfg.min_v_lines_per_table:
        long_v_core = v_keep_thin
    if len(h_keep_thin) >= cfg.min_h_lines_per_table:
        long_h_core = h_keep_thin

    # Require orthogonal support so text baselines do not become grid lines.
    if long_v_core and long_h_core:
        v_hits, h_hits = _intersection_counts(long_v_core, long_h_core)
        v_keep = [
            ln
            for ln, hits in zip(long_v_core, v_hits)
            if hits >= cfg.min_line_intersections
        ]
        h_keep = [
            ln
            for ln, hits in zip(long_h_core, h_hits)
            if hits >= cfg.min_line_intersections
        ]
        if len(v_keep) >= cfg.min_v_lines_per_table and len(h_keep) >= cfg.min_h_lines_per_table:
            long_v_core = v_keep
            long_h_core = h_keep

    raw_xs = [ln.pos for ln in long_v_core]
    raw_ys = [ln.pos for ln in long_h_core]

    # Use full table candidate bbox for refinement so weak border lines can be recovered.
    crop = line_mask[by0 : by1 + 1, bx0 : bx1 + 1]
    if crop.size == 0:
        return _ensure_border_lines(raw_xs, bx0, bx1), _ensure_border_lines(raw_ys, by0, by1)

    v_map, h_map = _draw_orientation_maps_for_bbox(
        crop.shape[:2],
        bx0,
        by0,
        bx1,
        by1,
        raw_v_lines,
        raw_h_lines,
    )
    if not np.any(v_map):
        v_map = crop
    if not np.any(h_map):
        h_map = crop

    # Use orientation-pure evidence so tiny vertical strokes cannot create
    # horizontal divider rows (and vice versa).
    x_strength = v_map.sum(axis=0).astype(np.float32)
    y_strength = h_map.sum(axis=1).astype(np.float32)
    x_coverage = _axis_coverage_profile(v_map, axis="x")
    y_coverage = _axis_coverage_profile(h_map, axis="y")
    x_support = _axis_support_profile_from_lines(
        long_v_core,
        low=bx0,
        high=bx1,
        span_norm=span_h,
        dilate_px=cfg.support_dilate_px,
    )
    y_support = _axis_support_profile_from_lines(
        long_h_core,
        low=by0,
        high=by1,
        span_norm=span_w,
        dilate_px=cfg.support_dilate_px,
    )
    # Keep detection permissive, but emphasize long-line support over raw ink coverage.
    x_cov_combined = np.maximum(0.50 * x_coverage, x_support)
    y_cov_combined = np.maximum(0.50 * y_coverage, y_support)

    # Top header rows contain dense text; down-weight them for row-divider scoring.
    head = int(round(cfg.top_header_band_ratio * crop.shape[0]))
    if head > 0:
        pen = float(np.clip(cfg.top_header_penalty, 0.0, 1.0))
        y_strength[:head] *= pen
        y_cov_combined[:head] *= pen

    # Robust axis normalization: clip dominant rows/columns and emphasize
    # relative local peaks so thin grid lines are not drowned out.
    x_strength = _robust_projection_strength(
        x_strength,
        clip_percentile=cfg.proj_clip_percentile,
        baseline_window_ratio=cfg.proj_baseline_window_ratio,
        rel_weight=cfg.proj_rel_weight,
    )
    y_strength = _robust_projection_strength(
        y_strength,
        clip_percentile=cfg.proj_clip_percentile,
        baseline_window_ratio=cfg.proj_baseline_window_ratio,
        rel_weight=cfg.proj_rel_weight,
    )

    # Map absolute coordinates to local coordinates for refinement.
    x_local = [v - bx0 for v in raw_xs if bx0 <= v <= bx1]
    y_local = [v - by0 for v in raw_ys if by0 <= v <= by1]
    x_pad = max(0, int(cfg.refine_bbox_pad_px), int(round(0.10 * crop.shape[1])))
    y_pad = max(0, int(cfg.refine_bbox_pad_px), int(round(0.10 * crop.shape[0])))
    if x_local:
        x_low = max(0, min(x_local) - x_pad)
        x_high = min(max(0, crop.shape[1] - 1), max(x_local) + x_pad)
    else:
        x_low, x_high = 0, max(0, crop.shape[1] - 1)
    if y_local:
        y_low = max(0, min(y_local) - y_pad)
        y_high = min(max(0, crop.shape[0] - 1), max(y_local) + y_pad)
    else:
        y_low, y_high = 0, max(0, crop.shape[0] - 1)
    if x_high <= x_low:
        x_low, x_high = 0, max(0, crop.shape[1] - 1)
    if y_high <= y_low:
        y_low, y_high = 0, max(0, crop.shape[0] - 1)

    x_refined_local = _refine_axis_lines(
        x_local,
        low=x_low,
        high=x_high,
        strength_1d=x_strength,
        coverage_1d=x_cov_combined,
        min_refined_coverage=cfg.min_refined_x_coverage,
        min_insert_coverage=cfg.insert_min_x_coverage,
        cfg=cfg,
    )
    y_refined_local = _refine_axis_lines(
        y_local,
        low=y_low,
        high=y_high,
        strength_1d=y_strength,
        coverage_1d=y_cov_combined,
        min_refined_coverage=cfg.min_refined_y_coverage,
        min_insert_coverage=cfg.insert_min_y_coverage,
        cfg=cfg,
    )

    # If an axis is clearly under-detected, relax coverage thresholds and retry.
    if len(x_refined_local) <= 7:
        x_refined_local = _refine_axis_lines(
            x_refined_local,
            low=x_low,
            high=x_high,
            strength_1d=x_strength,
            coverage_1d=x_cov_combined,
            min_refined_coverage=min(cfg.min_refined_x_coverage, 0.20),
            min_insert_coverage=min(cfg.insert_min_x_coverage, 0.10),
            cfg=cfg,
        )
    if len(y_refined_local) <= 6:
        y_refined_local = _refine_axis_lines(
            y_refined_local,
            low=y_low,
            high=y_high,
            strength_1d=y_strength,
            coverage_1d=y_cov_combined,
            min_refined_coverage=min(cfg.min_refined_y_coverage, 0.18),
            min_insert_coverage=min(cfg.insert_min_y_coverage, 0.08),
            cfg=cfg,
        )

    if cfg.enable_outlier_trim:
        x_refined_local = _trim_axis_outlier_clusters(x_refined_local, x_low, x_high)
        y_refined_local = _trim_axis_outlier_clusters(y_refined_local, y_low, y_high)

    # Border repair: if outer borders are weak/missing, promote strong edge candidates.
    x_refined_local = _repair_outer_borders(
        x_refined_local,
        x_strength,
        x_cov_combined,
        x_low,
        x_high,
        cfg=cfg,
    )
    y_refined_local = _repair_outer_borders(
        y_refined_local,
        y_strength,
        y_cov_combined,
        y_low,
        y_high,
        cfg=cfg,
    )

    # One extra structure pass after border promotion to fill easy missing lines.
    x_refined_local = _refine_axis_lines(
        x_refined_local,
        low=x_low,
        high=x_high,
        strength_1d=x_strength,
        coverage_1d=x_cov_combined,
        min_refined_coverage=cfg.min_refined_x_coverage,
        min_insert_coverage=cfg.insert_min_x_coverage,
        cfg=cfg,
    )
    y_refined_local = _refine_axis_lines(
        y_refined_local,
        low=y_low,
        high=y_high,
        strength_1d=y_strength,
        coverage_1d=y_cov_combined,
        min_refined_coverage=cfg.min_refined_y_coverage,
        min_insert_coverage=cfg.insert_min_y_coverage,
        cfg=cfg,
    )

    xs = [bx0 + v for v in x_refined_local]
    ys = [by0 + v for v in y_refined_local]
    xs = _prune_near_duplicate_positions(xs, max(12, cfg.proj_min_gap_px + 4))
    ys = _prune_near_duplicate_positions(ys, max(12, cfg.proj_min_gap_px + 4))
    x_low_abs = min(xs) if len(xs) >= 2 else bx0
    x_high_abs = max(xs) if len(xs) >= 2 else bx1
    y_low_abs = min(ys) if len(ys) >= 2 else by0
    y_high_abs = max(ys) if len(ys) >= 2 else by1
    xs = _ensure_border_lines(xs, x_low_abs, x_high_abs)
    ys = _ensure_border_lines(ys, y_low_abs, y_high_abs)
    return xs, ys


def _fallback_bbox_from_lines(
    shape: tuple[int, int],
    v_lines: list[CanonicalLine],
    h_lines: list[CanonicalLine],
) -> list[tuple[int, int, int, int]]:
    h, w = shape
    if not v_lines or not h_lines:
        return [(0, 0, w, h)]
    xs = [ln.pos for ln in v_lines]
    ys = [ln.pos for ln in h_lines]
    x0, x1 = max(0, min(xs) - 8), min(w - 1, max(xs) + 8)
    y0, y1 = max(0, min(ys) - 8), min(h - 1, max(ys) + 8)
    return [(x0, y0, max(1, x1 - x0), max(1, y1 - y0))]


def _crop_cells(
    image: np.ndarray,
    grid: TableGrid,
    output_dir: Path,
    min_w: int,
    min_h: int,
    inset_px: int,
) -> tuple[list[Path], list[list[Optional[Path]]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    matrix: list[list[Optional[Path]]] = []
    img_h, img_w = image.shape[:2]
    for r in range(len(grid.y_lines) - 1):
        y0 = grid.y_lines[r]
        y1 = grid.y_lines[r + 1]
        row_paths: list[Optional[Path]] = []
        for c in range(len(grid.x_lines) - 1):
            x0 = grid.x_lines[c]
            x1 = grid.x_lines[c + 1]

            cx0 = max(0, min(img_w, x0 + inset_px))
            cx1 = max(0, min(img_w, x1 - inset_px))
            cy0 = max(0, min(img_h, y0 + inset_px))
            cy1 = max(0, min(img_h, y1 - inset_px))

            if cx1 <= cx0 or cy1 <= cy0:
                row_paths.append(None)
                continue
            if (cx1 - cx0) < min_w or (cy1 - cy0) < min_h:
                row_paths.append(None)
                continue

            cell = image[cy0:cy1, cx0:cx1]
            if cell.size == 0:
                row_paths.append(None)
                continue
            path = output_dir / f"table_{grid.table_id:02d}_r{r:02d}_c{c:02d}.png"
            cv2.imwrite(str(path), cell)
            written.append(path)
            row_paths.append(path)
        matrix.append(row_paths)
    return written, matrix


def extract_scorecard_grids(
    image_path: str | Path,
    output_dir: str | Path,
    preprocess_cfg: Optional[PreprocessConfig] = None,
    hough_cfg: Optional[HoughConfig] = None,
) -> ExtractionResult:
    p_cfg = preprocess_cfg or PreprocessConfig()
    h_cfg = hough_cfg or HoughConfig()

    src = Path(image_path)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    image = load_image(src)
    prep = preprocess_scorecard(image, p_cfg)

    segments = _detect_hough_segments(prep.line_mask, h_cfg)
    if h_cfg.enable_sensitive_hough_pass:
        seg_soft = _detect_hough_segments(
            prep.line_mask,
            h_cfg,
            threshold_scale=h_cfg.sensitive_threshold_scale,
            min_len_scale=h_cfg.sensitive_min_line_length_scale,
            max_gap_scale=h_cfg.sensitive_max_line_gap_scale,
        )
        if seg_soft.size > 0:
            if segments.size == 0:
                segments = seg_soft
            else:
                segments = np.vstack([segments, seg_soft]).astype(np.int32, copy=False)
    v_seg, h_seg = _classify_segments(segments, h_cfg.angle_tolerance_deg)

    v_lines = _merge_1d_lines(v_seg, "v", h_cfg.merge_pos_tol_px, h_cfg.merge_gap_tol_px)
    h_lines = _merge_1d_lines(h_seg, "h", h_cfg.merge_pos_tol_px, h_cfg.merge_gap_tol_px)

    cand_v_lines, cand_h_lines = _filter_frame_like_lines(
        prep.line_mask.shape[:2], v_lines, h_lines
    )
    line_canvas = _draw_lines(prep.line_mask.shape[:2], cand_v_lines, cand_h_lines)
    cands = _select_table_candidates(line_canvas, cand_v_lines, cand_h_lines, h_cfg)
    if not cands:
        cands = _fallback_bbox_from_lines(
            prep.line_mask.shape[:2], cand_v_lines, cand_h_lines
        )

    # Stable ordering for left->right (or top->bottom if stacked).
    cands.sort(key=lambda b: (b[0], b[1]))

    refined: list[tuple[tuple[int, int, int, int], list[int], list[int]]] = []
    for bbox in cands:
        xs, ys = _lines_for_bbox(bbox, v_lines, h_lines, prep.line_mask, h_cfg)
        if len(xs) >= 2 and len(ys) >= 2:
            gx0, gx1 = int(min(xs)), int(max(xs))
            gy0, gy1 = int(min(ys)), int(max(ys))
            gb = (gx0, gy0, max(1, gx1 - gx0), max(1, gy1 - gy0))
        else:
            gb = bbox
        refined.append((gb, xs, ys))

    raw = refined
    refined = [
        (bbox, xs, ys)
        for bbox, xs, ys in raw
        if len(xs) >= h_cfg.min_grid_x_lines and len(ys) >= h_cfg.min_grid_y_lines
    ]
    if not refined and raw:
        # Never return nothing: keep the strongest structured candidate.
        refined = [max(raw, key=lambda t: len(t[1]) * len(t[2]))]

    grids: list[TableGrid] = []
    for i, (bbox, xs, ys) in enumerate(refined):
        hole_mode, hole_cols, sep_col = _infer_golf_structure(xs, len(refined))
        if hole_mode in {"single_9hole_table", "two_9hole_tables"}:
            xs, ys = _apply_9hole_priors(xs, ys)
            if len(xs) >= 2 and len(ys) >= 2:
                gx0, gx1 = int(min(xs)), int(max(xs))
                gy0, gy1 = int(min(ys)), int(max(ys))
                bbox = (gx0, gy0, max(1, gx1 - gx0), max(1, gy1 - gy0))
            hole_mode, hole_cols, sep_col = _infer_golf_structure(xs, len(refined))
        grids.append(
            TableGrid(
                table_id=i,
                bbox=bbox,
                x_lines=xs,
                y_lines=ys,
                hole_mode=hole_mode,
                hole_column_indices=hole_cols,
                separator_column_index=sep_col,
            )
        )

    written_cells: list[Path] = []
    cell_matrices: dict[int, list[list[Optional[Path]]]] = {}
    for grid in grids:
        table_dir = out_root / f"table_{grid.table_id:02d}"
        table_written, table_matrix = _crop_cells(
            prep.image_bgr,
            grid,
            table_dir,
            min_w=h_cfg.min_cell_w,
            min_h=h_cfg.min_cell_h,
            inset_px=h_cfg.cell_inset_px,
        )
        written_cells.extend(table_written)
        cell_matrices[grid.table_id] = table_matrix

        matrix_index = {
            "table_id": grid.table_id,
            "rows": len(table_matrix),
            "cols": (len(table_matrix[0]) if table_matrix else 0),
            "cells": [
                [str(p.name) if p is not None else None for p in row]
                for row in table_matrix
            ],
        }
        with open(table_dir / "matrix_index.json", "w", encoding="utf-8") as f:
            json.dump(matrix_index, f, indent=2)

    # Optional visual debug.
    debug = prep.image_bgr.copy()
    for grid in grids:
        x, y, w, h = grid.bbox
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 220, 0), 2)
        for xv in grid.x_lines:
            cv2.line(debug, (xv, y), (xv, y + h), (255, 80, 80), 1)
        for yv in grid.y_lines:
            cv2.line(debug, (x, yv), (x + w, yv), (80, 80, 255), 1)
        label = grid.hole_mode
        cv2.putText(
            debug,
            label,
            (x + 8, max(20, y + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(out_root / "debug_grid_overlay.png"), debug)
    cv2.imwrite(str(out_root / "debug_line_mask.png"), prep.line_mask)

    return ExtractionResult(
        source_image=src,
        output_dir=out_root,
        table_grids=grids,
        written_cells=written_cells,
        cell_matrices=cell_matrices,
        preprocess=prep,
    )


def extract_scorecard_grids_from_pdf(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
    preprocess_cfg: Optional[PreprocessConfig] = None,
    hough_cfg: Optional[HoughConfig] = None,
) -> list[ExtractionResult]:
    """Convert PDF pages to PNG, then extract grid cells for each page."""

    out_root = Path(output_dir)
    png_dir = out_root / "pdf_pages"
    page_pngs = convert_pdf_to_png(pdf_path, png_dir, dpi=dpi)

    results: list[ExtractionResult] = []
    for i, png in enumerate(page_pngs, start=1):
        page_out = out_root / f"page_{i:03d}"
        results.append(
            extract_scorecard_grids(
                png,
                page_out,
                preprocess_cfg=preprocess_cfg,
                hough_cfg=hough_cfg,
            )
        )
    return results


def _print_summary(result: ExtractionResult) -> None:
    print(f"Image: {result.source_image}")
    print(f"Output: {result.output_dir}")
    print(f"Tables found: {len(result.table_grids)}")
    for grid in result.table_grids:
        matrix = result.cell_matrices.get(grid.table_id, [])
        rows = len(matrix)
        cols = len(matrix[0]) if matrix else 0
        filled = sum(1 for row in matrix for p in row if p is not None)
        print(
            f"  table_{grid.table_id:02d}: rows={rows}, cols={cols}, filled={filled}, "
            f"mode={grid.hole_mode}, sep_col={grid.separator_column_index}"
        )
    print(f"Cells written: {len(result.written_cells)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract golf scorecard table cells with Hough lines + priors."
    )
    parser.add_argument("--input", required=True, help="Path to scorecard image.")
    parser.add_argument(
        "--output_dir",
        default="scorecard_cells",
        help="Directory where table cell PNGs will be written.",
    )
    parser.add_argument(
        "--pdf_dpi",
        type=int,
        default=300,
        help="DPI to rasterize PDF inputs (ignored for image inputs).",
    )
    parser.add_argument(
        "--min_axis_coverage",
        type=float,
        default=0.44,
        help="Strict long-line gate: minimum fraction of table height/width a line must span.",
    )
    parser.add_argument(
        "--proj_peak_thresh",
        type=float,
        default=0.14,
        help="Relative threshold for projection peaks used to refine dividers.",
    )
    parser.add_argument(
        "--split_gap_ratio",
        type=float,
        default=1.50,
        help="Insert missing divider(s) when a gap exceeds this times median gap.",
    )
    parser.add_argument(
        "--cell_inset",
        type=int,
        default=2,
        help="Inset (px) applied inside each cell crop to avoid borders/outside bleed.",
    )
    parser.add_argument(
        "--min_refined_coverage",
        type=float,
        default=0.22,
        help="Refinement coverage threshold (kept low; long-line gate handles strictness).",
    )
    parser.add_argument(
        "--insert_min_coverage",
        type=float,
        default=0.12,
        help="Minimum axis coverage when inserting missing divider lines in wide gaps.",
    )
    parser.add_argument(
        "--coverage_weight",
        type=float,
        default=0.60,
        help="How strongly line coverage influences divider scoring.",
    )
    parser.add_argument(
        "--sensitive_threshold_scale",
        type=float,
        default=0.58,
        help="Lower-threshold secondary Hough pass scale (smaller => more sensitive).",
    )
    parser.add_argument(
        "--sensitive_min_len_scale",
        type=float,
        default=0.55,
        help="Secondary Hough minimum-length scale (smaller => shorter lines allowed).",
    )
    parser.add_argument(
        "--sensitive_gap_scale",
        type=float,
        default=2.2,
        help="Secondary Hough gap scale (larger => bridge bigger breaks).",
    )
    parser.add_argument(
        "--min_refined_y_coverage",
        type=float,
        default=0.15,
        help="Minimum refined coverage for horizontal lines (lower => recover more rows).",
    )
    parser.add_argument(
        "--edge_min_coverage",
        type=float,
        default=0.12,
        help="Minimum coverage to promote edge candidates as outer borders.",
    )
    parser.add_argument(
        "--proj_clip_percentile",
        type=float,
        default=98.5,
        help="Clip projection outliers above this percentile before peak search.",
    )
    parser.add_argument(
        "--proj_baseline_window_ratio",
        type=float,
        default=0.14,
        help="Window ratio for local projection baseline normalization.",
    )
    parser.add_argument(
        "--proj_rel_weight",
        type=float,
        default=0.65,
        help="Weight for relative-over-baseline projection score vs absolute score.",
    )
    parser.add_argument(
        "--max_line_thickness",
        type=int,
        default=8,
        help="Maximum accepted line thickness in pixels unless strongly intersecting.",
    )
    parser.add_argument(
        "--thick_line_min_intersections",
        type=int,
        default=6,
        help="Keep thick lines only if they intersect this many orthogonal lines.",
    )
    args = parser.parse_args()

    h_cfg = HoughConfig(
        min_axis_line_coverage=args.min_axis_coverage,
        proj_peak_rel_thresh=args.proj_peak_thresh,
        split_gap_ratio=args.split_gap_ratio,
        cell_inset_px=max(0, args.cell_inset),
        min_refined_axis_coverage=args.min_refined_coverage,
        insert_min_axis_coverage=args.insert_min_coverage,
        coverage_weight=args.coverage_weight,
        sensitive_threshold_scale=args.sensitive_threshold_scale,
        sensitive_min_line_length_scale=args.sensitive_min_len_scale,
        sensitive_max_line_gap_scale=args.sensitive_gap_scale,
        min_refined_x_coverage=args.min_refined_coverage,
        min_refined_y_coverage=args.min_refined_y_coverage,
        insert_min_x_coverage=args.insert_min_coverage,
        insert_min_y_coverage=max(0.04, 0.75 * args.insert_min_coverage),
        edge_min_coverage=args.edge_min_coverage,
        proj_clip_percentile=args.proj_clip_percentile,
        proj_baseline_window_ratio=args.proj_baseline_window_ratio,
        proj_rel_weight=args.proj_rel_weight,
        max_line_thickness_px=max(2, args.max_line_thickness),
        thick_line_min_intersections=max(1, args.thick_line_min_intersections),
    )

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".pdf":
        results = extract_scorecard_grids_from_pdf(
            input_path,
            args.output_dir,
            dpi=args.pdf_dpi,
            hough_cfg=h_cfg,
        )
        print(f"PDF pages processed: {len(results)}")
        for res in results:
            _print_summary(res)
    else:
        result = extract_scorecard_grids(input_path, args.output_dir, hough_cfg=h_cfg)
        _print_summary(result)
