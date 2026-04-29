"""Grid-structure evaluator for scorecard extraction JSON outputs.

Compares a predicted `image_index.json` (or compatible schema) against a
ground-truth JSON and reports a single accuracy score plus metric breakdown.

Primary design goal:
- Avoid brittle index-shift collapse when one separator line is missed.
  We use order-preserving (monotonic) line correspondence with tolerance.

Supported table schema per table (minimum):
{
  "x_lines": [...],
  "y_lines": [...],
  "cells": [{"r0":..,"c0":..,"r1":..,"c1":..}, ...]   # optional
}

Top-level may be:
- {"tables":[...]}   (preferred, same as image_index.json)
- or a single table dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import argparse
import json

import numpy as np


def _f1(tp: int, fp: int, fn: int) -> float:
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    den = (2 * tp + fp + fn)
    return float((2 * tp) / den) if den > 0 else 0.0


def _precision(tp: int, fp: int) -> float:
    den = tp + fp
    if den <= 0:
        return 1.0
    return float(tp / den)


def _recall(tp: int, fn: int) -> float:
    den = tp + fn
    if den <= 0:
        return 1.0
    return float(tp / den)


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    bb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    den = aa + bb - inter
    if den <= 0.0:
        return 0.0
    return float(inter / den)


def _unique_sorted_ints(vals: list[Any]) -> list[int]:
    out: list[int] = []
    for v in vals:
        try:
            out.append(int(round(float(v))))
        except Exception:
            continue
    return sorted(set(out))


def _cluster_sorted_ints(vals: list[int], tol: int = 3) -> list[int]:
    if not vals:
        return []
    tol = int(max(0, tol))
    s = sorted(int(v) for v in vals)
    groups: list[list[int]] = [[s[0]]]
    for v in s[1:]:
        gm = int(round(float(sum(groups[-1])) / float(len(groups[-1]))))
        if abs(int(v) - gm) <= tol:
            groups[-1].append(int(v))
        else:
            groups.append([int(v)])
    out = [int(round(float(sum(g)) / float(len(g)))) for g in groups]
    return sorted(set(out))


def _clip_span(r0: int, c0: int, r1: int, c1: int, rows: int, cols: int) -> Optional[tuple[int, int, int, int]]:
    if rows <= 0 or cols <= 0:
        return None
    r0 = int(np.clip(r0, 0, rows - 1))
    r1 = int(np.clip(r1, 0, rows - 1))
    c0 = int(np.clip(c0, 0, cols - 1))
    c1 = int(np.clip(c1, 0, cols - 1))
    if r1 < r0:
        r0, r1 = r1, r0
    if c1 < c0:
        c0, c1 = c1, c0
    return (r0, c0, r1, c1)


def _extract_spans(table: dict[str, Any], rows: int, cols: int) -> list[tuple[int, int, int, int]]:
    spans: list[tuple[int, int, int, int]] = []

    cells = table.get("cells", None)
    if isinstance(cells, list):
        for c in cells:
            if not isinstance(c, dict):
                continue
            if not all(k in c for k in ("r0", "c0", "r1", "c1")):
                continue
            sp = _clip_span(int(c["r0"]), int(c["c0"]), int(c["r1"]), int(c["c1"]), rows, cols)
            if sp is not None:
                spans.append(sp)

    if not spans:
        mat = table.get("matrix", None)
        if isinstance(mat, list):
            for r, row in enumerate(mat):
                if not isinstance(row, list):
                    continue
                for c, item in enumerate(row):
                    if not isinstance(item, dict):
                        continue
                    merged = item.get("merged", item)
                    if not isinstance(merged, dict):
                        continue
                    anchor = bool(merged.get("anchor", item.get("anchor", False)))
                    if not anchor:
                        continue
                    rs = int(merged.get("rowspan", item.get("rowspan", 1)))
                    cs = int(merged.get("colspan", item.get("colspan", 1)))
                    sp = _clip_span(r, c, r + rs - 1, c + cs - 1, rows, cols)
                    if sp is not None:
                        spans.append(sp)

    # Fallback: atomic cells.
    if not spans:
        for r in range(rows):
            for c in range(cols):
                spans.append((r, c, r, c))

    spans = sorted(set(spans))
    return spans


def _presence_from_spans(rows: int, cols: int, spans: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """Derive boundary segment presence from merged-cell spans.

    v_presence shape: [rows, cols+1]
    h_presence shape: [rows+1, cols]
    """
    if rows <= 0 or cols <= 0:
        return np.zeros((0, 0), dtype=np.uint8), np.zeros((0, 0), dtype=np.uint8)

    ids = -np.ones((rows, cols), dtype=np.int32)
    for cid, (r0, c0, r1, c1) in enumerate(spans):
        ids[r0 : r1 + 1, c0 : c1 + 1] = int(cid)

    # Any uncovered slot becomes a unique atomic id (robustness).
    next_id = int(max(-1, ids.max())) + 1
    for r in range(rows):
        for c in range(cols):
            if ids[r, c] < 0:
                ids[r, c] = next_id
                next_id += 1

    v = np.zeros((rows, cols + 1), dtype=np.uint8)
    h = np.zeros((rows + 1, cols), dtype=np.uint8)

    # Outer border boundaries always present.
    v[:, 0] = 1
    v[:, cols] = 1
    h[0, :] = 1
    h[rows, :] = 1

    # Interior boundaries present when adjacent atomic slots belong to different merged cells.
    for r in range(rows):
        for c in range(1, cols):
            v[r, c] = 1 if ids[r, c - 1] != ids[r, c] else 0
    for r in range(1, rows):
        for c in range(cols):
            h[r, c] = 1 if ids[r - 1, c] != ids[r, c] else 0

    return v, h


def _spans_from_presence(rows: int, cols: int, v_presence: np.ndarray, h_presence: np.ndarray) -> list[tuple[int, int, int, int]]:
    if rows <= 0 or cols <= 0:
        return []
    n = rows * cols
    parent = list(range(n))

    def idx(r: int, c: int) -> int:
        return int(r * cols + c)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Missing vertical boundary merges horizontal neighbors.
    for r in range(rows):
        for c in range(cols - 1):
            if int(v_presence[r, c + 1]) == 0:
                union(idx(r, c), idx(r, c + 1))

    # Missing horizontal boundary merges vertical neighbors.
    for r in range(rows - 1):
        for c in range(cols):
            if int(h_presence[r + 1, c]) == 0:
                union(idx(r, c), idx(r + 1, c))

    groups: dict[int, list[tuple[int, int]]] = {}
    for r in range(rows):
        for c in range(cols):
            rt = find(idx(r, c))
            groups.setdefault(rt, []).append((r, c))

    spans: list[tuple[int, int, int, int]] = []
    for members in groups.values():
        rs = [m[0] for m in members]
        cs = [m[1] for m in members]
        spans.append((int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))))
    spans = sorted(set(spans))
    return spans


@dataclass
class TableStruct:
    bbox: tuple[float, float, float, float]
    x_lines: list[int]
    y_lines: list[int]
    spans: list[tuple[int, int, int, int]]
    v_presence: np.ndarray
    h_presence: np.ndarray

    @property
    def rows(self) -> int:
        return max(0, len(self.y_lines) - 1)

    @property
    def cols(self) -> int:
        return max(0, len(self.x_lines) - 1)


def _table_from_raw(tbl: dict[str, Any]) -> Optional[TableStruct]:
    # Row-separator format support:
    # {
    #   "table_bbox": [x0,y0,x1,y1],
    #   "v_lines": [ [[x,y],[x,y],[x,y]], ... ],
    #   "separators": [ {"polyline":[...], "kind":"line|boundary"}, ... ]
    # }
    if "table_bbox" in tbl and ("v_lines" in tbl or "separators" in tbl):
        try:
            bb = tbl.get("table_bbox", [0, 0, 0, 0])
            x0, y0, x1, y1 = [int(round(float(bb[i]))) for i in range(4)]
        except Exception:
            return None
        if x1 <= x0 or y1 <= y0:
            return None

        v_lines_raw = tbl.get("v_lines", [])
        seps_raw = tbl.get("separators", [])
        if not isinstance(v_lines_raw, list):
            v_lines_raw = []
        if not isinstance(seps_raw, list):
            seps_raw = []

        xv: list[int] = [int(x0), int(x1)]
        for ln in v_lines_raw:
            if not isinstance(ln, list) or len(ln) < 2:
                continue
            try:
                xs = [float(p[0]) for p in ln]
            except Exception:
                continue
            xv.append(int(round(float(np.mean(xs)))))
        x_lines = _cluster_sorted_ints(xv, tol=3)

        yh: list[int] = [int(y0), int(y1)]
        for s in seps_raw:
            if not isinstance(s, dict):
                continue
            ln = s.get("polyline", [])
            if not isinstance(ln, list) or len(ln) < 2:
                continue
            try:
                ys = [float(p[1]) for p in ln]
            except Exception:
                continue
            yh.append(int(round(float(np.mean(ys)))))
        y_lines = _cluster_sorted_ints(yh, tol=3)

        if len(x_lines) < 2 or len(y_lines) < 2:
            return None
        rows = len(y_lines) - 1
        cols = len(x_lines) - 1

        v_presence = np.zeros((rows, cols + 1), dtype=np.uint8)
        h_presence = np.zeros((rows + 1, cols), dtype=np.uint8)
        v_presence[:, 0] = 1
        v_presence[:, cols] = 1
        h_presence[0, :] = 1
        h_presence[rows, :] = 1

        tolx = max(3.0, 0.010 * float(max(1, x1 - x0)))
        toly = max(3.0, 0.010 * float(max(1, y1 - y0)))

        def _x_at_y(line: list[list[int]], yq: float) -> float:
            pts = sorted([[float(p[0]), float(p[1])] for p in line], key=lambda p: p[1])
            if len(pts) < 2:
                return float(pts[0][0]) if pts else 0.0
            if len(pts) == 2:
                pts = [pts[0], [(pts[0][0] + pts[1][0]) * 0.5, (pts[0][1] + pts[1][1]) * 0.5], pts[1]]
            p0, p1, p2 = pts[0], pts[1], pts[2]

            def _lerp_x(a: list[float], b: list[float], yy: float) -> float:
                ax, ay = a
                bx, by = b
                dy = by - ay
                if abs(dy) < 1e-6:
                    return 0.5 * (ax + bx)
                t = max(0.0, min(1.0, (yy - ay) / dy))
                return ax + t * (bx - ax)

            if yq <= p1[1]:
                return _lerp_x(p0, p1, yq)
            return _lerp_x(p1, p2, yq)

        def _y_at_x(line: list[list[int]], xq: float) -> float:
            pts = sorted([[float(p[0]), float(p[1])] for p in line], key=lambda p: p[0])
            if len(pts) < 2:
                return float(pts[0][1]) if pts else 0.0
            if len(pts) == 2:
                pts = [pts[0], [(pts[0][0] + pts[1][0]) * 0.5, (pts[0][1] + pts[1][1]) * 0.5], pts[1]]
            p0, p1, p2 = pts[0], pts[1], pts[2]

            def _lerp_y(a: list[float], b: list[float], xx: float) -> float:
                ax, ay = a
                bx, by = b
                dx = bx - ax
                if abs(dx) < 1e-6:
                    return 0.5 * (ay + by)
                t = max(0.0, min(1.0, (xx - ax) / dx))
                return ay + t * (by - ay)

            if xq <= p1[0]:
                return _lerp_y(p0, p1, xq)
            return _lerp_y(p1, p2, xq)

        # Mark vertical segment presence by row-band midpoints.
        for ln in v_lines_raw:
            if not isinstance(ln, list) or len(ln) < 2:
                continue
            try:
                xavg = float(np.mean([float(p[0]) for p in ln]))
                yvals = [float(p[1]) for p in ln]
            except Exception:
                continue
            j = int(np.argmin([abs(float(xv_) - xavg) for xv_ in x_lines]))
            if j < 0 or j >= len(x_lines):
                continue
            if abs(float(x_lines[j]) - xavg) > tolx:
                continue
            ylo = float(min(yvals)) - toly
            yhi = float(max(yvals)) + toly
            for r in range(rows):
                ym = 0.5 * (float(y_lines[r]) + float(y_lines[r + 1]))
                if ym < ylo or ym > yhi:
                    continue
                xx = _x_at_y(ln, ym)
                if abs(xx - float(x_lines[j])) <= tolx:
                    v_presence[r, j] = 1

        # Mark horizontal segment presence by col-band midpoints.
        for s in seps_raw:
            if not isinstance(s, dict):
                continue
            ln = s.get("polyline", [])
            if not isinstance(ln, list) or len(ln) < 2:
                continue
            try:
                yavg = float(np.mean([float(p[1]) for p in ln]))
                xvals = [float(p[0]) for p in ln]
            except Exception:
                continue
            i = int(np.argmin([abs(float(yv_) - yavg) for yv_ in y_lines]))
            if i < 0 or i >= len(y_lines):
                continue
            if abs(float(y_lines[i]) - yavg) > toly:
                continue
            xlo = float(min(xvals)) - tolx
            xhi = float(max(xvals)) + tolx
            for c in range(cols):
                xm = 0.5 * (float(x_lines[c]) + float(x_lines[c + 1]))
                if xm < xlo or xm > xhi:
                    continue
                yy = _y_at_x(ln, xm)
                if abs(yy - float(y_lines[i])) <= toly:
                    h_presence[i, c] = 1

        spans = _spans_from_presence(rows=rows, cols=cols, v_presence=v_presence, h_presence=h_presence)
        return TableStruct(
            bbox=(float(x0), float(y0), float(x1), float(y1)),
            x_lines=x_lines,
            y_lines=y_lines,
            spans=spans,
            v_presence=v_presence,
            h_presence=h_presence,
        )

    x_lines = _unique_sorted_ints(tbl.get("x_lines", []))
    y_lines = _unique_sorted_ints(tbl.get("y_lines", []))
    if len(x_lines) < 2 or len(y_lines) < 2:
        return None
    rows = len(y_lines) - 1
    cols = len(x_lines) - 1

    bbox_raw = tbl.get("bbox_xyxy", None)
    if isinstance(bbox_raw, list) and len(bbox_raw) >= 4:
        try:
            bx0, by0, bx1, by1 = [float(bbox_raw[i]) for i in range(4)]
        except Exception:
            bx0, by0, bx1, by1 = float(min(x_lines)), float(min(y_lines)), float(max(x_lines)), float(max(y_lines))
    else:
        bx0, by0, bx1, by1 = float(min(x_lines)), float(min(y_lines)), float(max(x_lines)), float(max(y_lines))

    spans = _extract_spans(tbl, rows=rows, cols=cols)
    v_presence, h_presence = _presence_from_spans(rows=rows, cols=cols, spans=spans)
    return TableStruct(
        bbox=(bx0, by0, bx1, by1),
        x_lines=x_lines,
        y_lines=y_lines,
        spans=spans,
        v_presence=v_presence,
        h_presence=h_presence,
    )


def _load_tables(path: Path) -> list[TableStruct]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw_tables: list[dict[str, Any]] = []
    if isinstance(data, dict) and isinstance(data.get("tables"), list):
        raw_tables = [t for t in data["tables"] if isinstance(t, dict)]
    elif isinstance(data, dict) and ("x_lines" in data and "y_lines" in data):
        raw_tables = [data]
    elif isinstance(data, list):
        raw_tables = [t for t in data if isinstance(t, dict)]

    out: list[TableStruct] = []
    for t in raw_tables:
        ts = _table_from_raw(t)
        if ts is not None:
            out.append(ts)
    return out


def _best_table_matching_exact(
    gt_tables: list[TableStruct],
    pr_tables: list[TableStruct],
    iou_thr: float,
) -> tuple[list[tuple[int, int]], set[int], set[int], list[float]]:
    n = len(gt_tables)
    m = len(pr_tables)
    iou = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            iou[i, j] = _bbox_iou(gt_tables[i].bbox, pr_tables[j].bbox)

    best_pairs: list[tuple[int, int]] = []
    best_tp = -1
    best_sum = -1.0

    used = [False] * m
    cur_pairs: list[tuple[int, int]] = []

    def _dfs(i: int, cur_tp: int, cur_sum: float) -> None:
        nonlocal best_pairs, best_tp, best_sum
        if i >= n:
            if cur_tp > best_tp or (cur_tp == best_tp and cur_sum > best_sum):
                best_tp = cur_tp
                best_sum = cur_sum
                best_pairs = list(cur_pairs)
            return

        # Option 1: skip this GT table.
        _dfs(i + 1, cur_tp, cur_sum)

        # Option 2: match with one unused pred table above threshold.
        for j in range(m):
            if used[j]:
                continue
            s = float(iou[i, j])
            if s < float(iou_thr):
                continue
            used[j] = True
            cur_pairs.append((i, j))
            _dfs(i + 1, cur_tp + 1, cur_sum + s)
            cur_pairs.pop()
            used[j] = False

    _dfs(0, 0, 0.0)

    used_gt = {i for i, _ in best_pairs}
    used_pr = {j for _, j in best_pairs}
    unmatched_gt = set(range(n)) - used_gt
    unmatched_pr = set(range(m)) - used_pr
    ious = [float(iou[i, j]) for i, j in best_pairs]
    return best_pairs, unmatched_gt, unmatched_pr, ious


def _match_lines_dp(gt_vals: list[int], pr_vals: list[int], tol: float) -> tuple[list[tuple[int, int]], list[float]]:
    """Order-preserving line matching with gap tolerance.

    Maximizes #matches first, then minimizes total absolute distance.
    """
    n = len(gt_vals)
    m = len(pr_vals)
    if n == 0 or m == 0:
        return [], []

    best_m = np.zeros((n + 1, m + 1), dtype=np.int32)
    best_d = np.zeros((n + 1, m + 1), dtype=np.float32)
    choice = np.zeros((n + 1, m + 1), dtype=np.uint8)  # 1=skip_gt,2=skip_pr,3=match

    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            # skip gt
            m1 = int(best_m[i + 1, j])
            d1 = float(best_d[i + 1, j])
            c = 1
            bm, bd = m1, d1

            # skip pred
            m2 = int(best_m[i, j + 1])
            d2 = float(best_d[i, j + 1])
            if m2 > bm or (m2 == bm and d2 < bd):
                bm, bd, c = m2, d2, 2

            # match if within tolerance
            dd = abs(float(gt_vals[i]) - float(pr_vals[j]))
            if dd <= tol:
                m3 = int(best_m[i + 1, j + 1]) + 1
                d3 = float(best_d[i + 1, j + 1]) + float(dd)
                if m3 > bm or (m3 == bm and d3 < bd):
                    bm, bd, c = m3, d3, 3

            best_m[i, j] = int(bm)
            best_d[i, j] = float(bd)
            choice[i, j] = int(c)

    matches: list[tuple[int, int]] = []
    dists: list[float] = []
    i = 0
    j = 0
    while i < n and j < m:
        c = int(choice[i, j])
        if c == 3:
            matches.append((i, j))
            dists.append(abs(float(gt_vals[i]) - float(pr_vals[j])))
            i += 1
            j += 1
        elif c == 1:
            i += 1
        else:
            j += 1
    return matches, dists


def _spans_to_boxes(table: TableStruct) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for r0, c0, r1, c1 in table.spans:
        x0 = float(table.x_lines[c0])
        x1 = float(table.x_lines[c1 + 1])
        y0 = float(table.y_lines[r0])
        y1 = float(table.y_lines[r1 + 1])
        if x1 <= x0 or y1 <= y0:
            continue
        out.append((x0, y0, x1, y1))
    return out


def _greedy_match_boxes(
    gt_boxes: list[tuple[float, float, float, float]],
    pr_boxes: list[tuple[float, float, float, float]],
    iou_thr: float,
) -> tuple[int, int, int, list[float]]:
    cands: list[tuple[float, int, int]] = []
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pr_boxes):
            s = _bbox_iou(gb, pb)
            if s >= float(iou_thr):
                cands.append((float(s), i, j))
    cands.sort(reverse=True, key=lambda t: t[0])

    used_gt: set[int] = set()
    used_pr: set[int] = set()
    ious: list[float] = []
    for s, i, j in cands:
        if i in used_gt or j in used_pr:
            continue
        used_gt.add(i)
        used_pr.add(j)
        ious.append(float(s))

    tp = len(ious)
    fp = len(pr_boxes) - tp
    fn = len(gt_boxes) - tp
    return int(tp), int(fp), int(fn), ious


def _segment_counts_for_table(table: TableStruct) -> int:
    return int(np.sum(table.v_presence)) + int(np.sum(table.h_presence))


@dataclass
class _LineSegment:
    orient: str  # "v" or "h"
    pos: float   # x for vertical, y for horizontal
    a0: float    # span start along orthogonal axis
    a1: float    # span end along orthogonal axis
    tol: float   # matching tolerance (GT-local)
    sigma: float # Gaussian sigma (for soft score)


def _collect_line_segments(table: TableStruct, with_local_tol: bool) -> list[_LineSegment]:
    rows = int(table.rows)
    cols = int(table.cols)
    out: list[_LineSegment] = []
    if rows <= 0 or cols <= 0:
        return out

    x = [int(v) for v in table.x_lines]
    y = [int(v) for v in table.y_lines]

    # Vertical segments: boundary index j in [0..cols], row band r in [0..rows-1].
    for r in range(rows):
        y0 = float(y[r])
        y1 = float(y[r + 1])
        if y1 <= y0:
            continue
        for j in range(cols + 1):
            if j >= table.v_presence.shape[1] or r >= table.v_presence.shape[0]:
                continue
            if int(table.v_presence[r, j]) == 0:
                continue
            xpos = float(x[j])
            tol = 1.0
            sigma = 0.5
            if with_local_tol:
                w_left = float(x[j] - x[j - 1]) if j > 0 else None
                w_right = float(x[j + 1] - x[j]) if j < cols else None
                if w_left is not None and w_right is not None:
                    base = 0.5 * (w_left + w_right)
                elif w_left is not None:
                    base = w_left
                elif w_right is not None:
                    base = w_right
                else:
                    base = 1.0
                base = max(1.0, float(base))
                tol = max(1.0, 0.10 * base)
                sigma = max(1e-6, 0.5 * tol)
            out.append(_LineSegment(orient="v", pos=xpos, a0=y0, a1=y1, tol=float(tol), sigma=float(sigma)))

    # Horizontal segments: boundary index i in [0..rows], col band c in [0..cols-1].
    for i in range(rows + 1):
        ypos = float(y[i])
        for c in range(cols):
            if i >= table.h_presence.shape[0] or c >= table.h_presence.shape[1]:
                continue
            if int(table.h_presence[i, c]) == 0:
                continue
            x0 = float(x[c])
            x1 = float(x[c + 1])
            if x1 <= x0:
                continue
            tol = 1.0
            sigma = 0.5
            if with_local_tol:
                h_up = float(y[i] - y[i - 1]) if i > 0 else None
                h_dn = float(y[i + 1] - y[i]) if i < rows else None
                if h_up is not None and h_dn is not None:
                    base = 0.5 * (h_up + h_dn)
                elif h_up is not None:
                    base = h_up
                elif h_dn is not None:
                    base = h_dn
                else:
                    base = 1.0
                base = max(1.0, float(base))
                tol = max(1.0, 0.10 * base)
                sigma = max(1e-6, 0.5 * tol)
            out.append(_LineSegment(orient="h", pos=ypos, a0=x0, a1=x1, tol=float(tol), sigma=float(sigma)))

    return out


def _span_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo = max(float(a0), float(b0))
    hi = min(float(a1), float(b1))
    return max(0.0, hi - lo)


def _soft_segment_match_score(
    gt_segments: list[_LineSegment],
    pr_segments: list[_LineSegment],
) -> tuple[float, int, int, int]:
    """One-to-one soft matching of predicted segments to GT segments.

    Returns:
      score_sum: summed truncated-Gaussian scores over matched pairs
      matched_gt: number of GT segments that found a match
      matched_pr: number of predicted segments used in a match
      candidates: number of viable candidate pairs considered
    """
    if not gt_segments or not pr_segments:
        return 0.0, 0, 0, 0

    cands: list[tuple[float, float, int, int]] = []  # (score, -dist, gi, pj)
    for gi, g in enumerate(gt_segments):
        glen = max(1e-6, float(g.a1 - g.a0))
        min_overlap = 0.35 * glen
        for pj, p in enumerate(pr_segments):
            if g.orient != p.orient:
                continue
            ov = _span_overlap(g.a0, g.a1, p.a0, p.a1)
            if ov < min_overlap:
                continue
            d = abs(float(g.pos) - float(p.pos))
            if d > float(g.tol):
                continue
            s = float(np.exp(-(d * d) / (2.0 * float(g.sigma) * float(g.sigma))))
            cands.append((s, -d, gi, pj))

    if not cands:
        return 0.0, 0, 0, 0

    cands.sort(reverse=True, key=lambda t: (t[0], t[1]))
    used_gt: set[int] = set()
    used_pr: set[int] = set()
    score_sum = 0.0
    for s, _nd, gi, pj in cands:
        if gi in used_gt or pj in used_pr:
            continue
        used_gt.add(gi)
        used_pr.add(pj)
        score_sum += float(s)

    return float(score_sum), int(len(used_gt)), int(len(used_pr)), int(len(cands))


def _compare_matched_tables(
    gt: TableStruct,
    pr: TableStruct,
    line_tol_abs_px: float,
    line_tol_rel: float,
    cell_iou_thr: float,
) -> dict[str, Any]:
    gt_w = max(1.0, float(gt.bbox[2] - gt.bbox[0]))
    gt_h = max(1.0, float(gt.bbox[3] - gt.bbox[1]))
    tol_x = max(float(line_tol_abs_px), float(line_tol_rel) * gt_w)
    tol_y = max(float(line_tol_abs_px), float(line_tol_rel) * gt_h)

    mx, dx = _match_lines_dp(gt.x_lines, pr.x_lines, tol=tol_x)
    my, dy = _match_lines_dp(gt.y_lines, pr.y_lines, tol=tol_y)

    # Line counts (boundaries included, consistently for GT/pred).
    ltp = len(mx) + len(my)
    lfp = (len(pr.x_lines) - len(mx)) + (len(pr.y_lines) - len(my))
    lfn = (len(gt.x_lines) - len(mx)) + (len(gt.y_lines) - len(my))

    # Segment comparison on unambiguous bands only (adjacent matched boundaries).
    seg_tp = 0
    seg_fp = 0
    seg_fn = 0

    mx_s = sorted(mx, key=lambda t: t[0])
    my_s = sorted(my, key=lambda t: t[0])

    # Vertical segments at matched x boundaries, across matched adjacent y bands.
    for gx, px in mx_s:
        for k in range(len(my_s) - 1):
            gy0, py0 = my_s[k]
            gy1, py1 = my_s[k + 1]
            if (gy1 - gy0) != 1 or (py1 - py0) != 1:
                continue
            if gy0 < 0 or gy0 >= gt.v_presence.shape[0]:
                continue
            if py0 < 0 or py0 >= pr.v_presence.shape[0]:
                continue
            if gx < 0 or gx >= gt.v_presence.shape[1]:
                continue
            if px < 0 or px >= pr.v_presence.shape[1]:
                continue
            gv = int(gt.v_presence[gy0, gx])
            pv = int(pr.v_presence[py0, px])
            if gv == 1 and pv == 1:
                seg_tp += 1
            elif gv == 0 and pv == 1:
                seg_fp += 1
            elif gv == 1 and pv == 0:
                seg_fn += 1

    # Horizontal segments at matched y boundaries, across matched adjacent x bands.
    for gy, py in my_s:
        for k in range(len(mx_s) - 1):
            gx0, px0 = mx_s[k]
            gx1, px1 = mx_s[k + 1]
            if (gx1 - gx0) != 1 or (px1 - px0) != 1:
                continue
            if gy < 0 or gy >= gt.h_presence.shape[0]:
                continue
            if py < 0 or py >= pr.h_presence.shape[0]:
                continue
            if gx0 < 0 or gx0 >= gt.h_presence.shape[1]:
                continue
            if px0 < 0 or px0 >= pr.h_presence.shape[1]:
                continue
            gh = int(gt.h_presence[gy, gx0])
            ph = int(pr.h_presence[py, px0])
            if gh == 1 and ph == 1:
                seg_tp += 1
            elif gh == 0 and ph == 1:
                seg_fp += 1
            elif gh == 1 and ph == 0:
                seg_fn += 1

    gt_boxes = _spans_to_boxes(gt)
    pr_boxes = _spans_to_boxes(pr)
    ctp, cfp, cfn, cious = _greedy_match_boxes(gt_boxes, pr_boxes, iou_thr=float(cell_iou_thr))

    # Soft line-segment scoring (local 10% tolerance + truncated Gaussian).
    gt_soft_segments = _collect_line_segments(gt, with_local_tol=True)
    pr_soft_segments = _collect_line_segments(pr, with_local_tol=False)
    soft_score_sum, soft_match_gt, soft_match_pr, soft_cands = _soft_segment_match_score(
        gt_segments=gt_soft_segments,
        pr_segments=pr_soft_segments,
    )

    return {
        "line_tp": int(ltp),
        "line_fp": int(lfp),
        "line_fn": int(lfn),
        "line_abs_errors_px": [float(v) for v in (dx + dy)],
        "segment_tp": int(seg_tp),
        "segment_fp": int(seg_fp),
        "segment_fn": int(seg_fn),
        "cell_tp": int(ctp),
        "cell_fp": int(cfp),
        "cell_fn": int(cfn),
        "cell_match_ious": [float(v) for v in cious],
        "gt_cells": int(len(gt_boxes)),
        "pr_cells": int(len(pr_boxes)),
        "soft_seg_score_sum": float(soft_score_sum),
        "soft_seg_match_gt": int(soft_match_gt),
        "soft_seg_match_pr": int(soft_match_pr),
        "soft_seg_gt_count": int(len(gt_soft_segments)),
        "soft_seg_pr_count": int(len(pr_soft_segments)),
        "soft_seg_candidates": int(soft_cands),
    }


def evaluate(
    gt_json: Path,
    pred_json: Path,
    min_cols_keep: int = 9,
    table_iou_match_thr: float = 0.20,
    line_tol_abs_px: float = 4.0,
    line_tol_rel: float = 0.02,
    cell_iou_thr: float = 0.75,
    line_soft_alpha: float = 0.5,
    w_table: float = 0.10,
    w_line: float = 0.15,
    w_segment: float = 0.25,
    w_cell: float = 0.50,
) -> dict[str, Any]:
    gt_tables_all = _load_tables(gt_json)
    pr_tables_all = _load_tables(pred_json)
    min_cols_keep = int(max(1, min_cols_keep))

    # Enforce minimum columns for both GT and predicted tables.
    gt_tables = [t for t in gt_tables_all if int(t.cols) >= min_cols_keep]
    pr_tables = [t for t in pr_tables_all if int(t.cols) >= min_cols_keep]
    pairs, ug, up, table_ious = _best_table_matching_exact(
        gt_tables=gt_tables,
        pr_tables=pr_tables,
        iou_thr=float(table_iou_match_thr),
    )

    table_tp = len(pairs)
    table_fp = len(up)
    table_fn = len(ug)

    line_tp = 0
    line_fp = 0
    line_fn = 0
    line_errs: list[float] = []

    seg_tp = 0
    seg_fp = 0
    seg_fn = 0

    cell_tp = 0
    cell_fp = 0
    cell_fn = 0
    cell_ious: list[float] = []

    soft_seg_score_sum = 0.0
    soft_seg_match_gt = 0
    soft_seg_match_pr = 0
    soft_seg_gt_count = 0
    soft_seg_pr_count = 0
    soft_seg_candidates = 0

    matched_table_details: list[dict[str, Any]] = []
    for gi, pj in pairs:
        r = _compare_matched_tables(
            gt=gt_tables[gi],
            pr=pr_tables[pj],
            line_tol_abs_px=float(line_tol_abs_px),
            line_tol_rel=float(line_tol_rel),
            cell_iou_thr=float(cell_iou_thr),
        )
        line_tp += int(r["line_tp"])
        line_fp += int(r["line_fp"])
        line_fn += int(r["line_fn"])
        line_errs.extend([float(v) for v in r["line_abs_errors_px"]])

        seg_tp += int(r["segment_tp"])
        seg_fp += int(r["segment_fp"])
        seg_fn += int(r["segment_fn"])

        cell_tp += int(r["cell_tp"])
        cell_fp += int(r["cell_fp"])
        cell_fn += int(r["cell_fn"])
        cell_ious.extend([float(v) for v in r["cell_match_ious"]])

        soft_seg_score_sum += float(r["soft_seg_score_sum"])
        soft_seg_match_gt += int(r["soft_seg_match_gt"])
        soft_seg_match_pr += int(r["soft_seg_match_pr"])
        soft_seg_gt_count += int(r["soft_seg_gt_count"])
        soft_seg_pr_count += int(r["soft_seg_pr_count"])
        soft_seg_candidates += int(r["soft_seg_candidates"])

        matched_table_details.append(
            {
                "gt_table_index": int(gi),
                "pred_table_index": int(pj),
                "table_iou": float(_bbox_iou(gt_tables[gi].bbox, pr_tables[pj].bbox)),
                "gt_rows": int(gt_tables[gi].rows),
                "gt_cols": int(gt_tables[gi].cols),
                "pred_rows": int(pr_tables[pj].rows),
                "pred_cols": int(pr_tables[pj].cols),
                "line_tp": int(r["line_tp"]),
                "line_fp": int(r["line_fp"]),
                "line_fn": int(r["line_fn"]),
                "segment_tp": int(r["segment_tp"]),
                "segment_fp": int(r["segment_fp"]),
                "segment_fn": int(r["segment_fn"]),
                "cell_tp": int(r["cell_tp"]),
                "cell_fp": int(r["cell_fp"]),
                "cell_fn": int(r["cell_fn"]),
                "soft_seg_score_sum": float(r["soft_seg_score_sum"]),
                "soft_seg_match_gt": int(r["soft_seg_match_gt"]),
                "soft_seg_match_pr": int(r["soft_seg_match_pr"]),
                "soft_seg_gt_count": int(r["soft_seg_gt_count"]),
                "soft_seg_pr_count": int(r["soft_seg_pr_count"]),
            }
        )

    # Unmatched tables count entirely as misses/extras for line/segment/cell families.
    for i in ug:
        gt = gt_tables[i]
        line_fn += int(len(gt.x_lines) + len(gt.y_lines))
        seg_fn += int(_segment_counts_for_table(gt))
        cell_fn += int(len(gt.spans))
        soft_seg_gt_count += int(len(_collect_line_segments(gt, with_local_tol=True)))
    for j in up:
        pr = pr_tables[j]
        line_fp += int(len(pr.x_lines) + len(pr.y_lines))
        seg_fp += int(_segment_counts_for_table(pr))
        cell_fp += int(len(pr.spans))
        soft_seg_pr_count += int(len(_collect_line_segments(pr, with_local_tol=False)))

    table_prec = _precision(table_tp, table_fp)
    table_rec = _recall(table_tp, table_fn)
    table_f1 = _f1(table_tp, table_fp, table_fn)

    line_prec = _precision(line_tp, line_fp)
    line_rec = _recall(line_tp, line_fn)
    line_f1 = _f1(line_tp, line_fp, line_fn)

    seg_prec = _precision(seg_tp, seg_fp)
    seg_rec = _recall(seg_tp, seg_fn)
    seg_f1 = _f1(seg_tp, seg_fp, seg_fn)

    cell_prec = _precision(cell_tp, cell_fp)
    cell_rec = _recall(cell_tp, cell_fn)
    cell_f1 = _f1(cell_tp, cell_fp, cell_fn)

    # Proposed soft line-segment metric (truncated Gaussian + alpha penalty).
    alpha = max(0.0, float(line_soft_alpha))
    soft_unmatched_pred = max(0, int(soft_seg_pr_count - soft_seg_match_pr))
    soft_raw = float(soft_seg_score_sum - alpha * float(soft_unmatched_pred))
    soft_norm = float(soft_raw / float(max(1, soft_seg_gt_count)))
    soft_accuracy = float(np.clip(soft_norm, 0.0, 1.0))
    soft_precision = (
        float(soft_seg_score_sum / (soft_seg_score_sum + alpha * float(soft_unmatched_pred)))
        if (soft_seg_score_sum + alpha * float(soft_unmatched_pred)) > 0.0
        else 1.0
    )
    soft_recall = float(soft_seg_score_sum / float(max(1, soft_seg_gt_count)))
    soft_pred_penalty = float(alpha * float(soft_unmatched_pred))

    w_sum = float(w_table + w_line + w_segment + w_cell)
    if w_sum <= 0.0:
        w_table_n, w_line_n, w_seg_n, w_cell_n = 0.10, 0.15, 0.25, 0.50
    else:
        w_table_n = float(w_table / w_sum)
        w_line_n = float(w_line / w_sum)
        w_seg_n = float(w_segment / w_sum)
        w_cell_n = float(w_cell / w_sum)

    accuracy = (
        w_table_n * table_f1
        + w_line_n * line_f1
        + w_seg_n * seg_f1
        + w_cell_n * cell_f1
    )

    out = {
        "accuracy": float(accuracy),
        "accuracy_percent": float(100.0 * accuracy),
        "components": {
            "table_f1": float(table_f1),
            "line_f1": float(line_f1),
            "segment_f1": float(seg_f1),
            "cell_f1": float(cell_f1),
        },
        "weights_used": {
            "table": float(w_table_n),
            "line": float(w_line_n),
            "segment": float(w_seg_n),
            "cell": float(w_cell_n),
        },
        "table_metrics": {
            "tp": int(table_tp),
            "fp": int(table_fp),
            "fn": int(table_fn),
            "precision": float(table_prec),
            "recall": float(table_rec),
            "f1": float(table_f1),
            "mean_iou_matched": float(np.mean(table_ious)) if table_ious else 0.0,
        },
        "line_metrics": {
            "tp": int(line_tp),
            "fp": int(line_fp),
            "fn": int(line_fn),
            "precision": float(line_prec),
            "recall": float(line_rec),
            "f1": float(line_f1),
            "mean_abs_error_px_matched": float(np.mean(line_errs)) if line_errs else 0.0,
        },
        "segment_metrics": {
            "tp": int(seg_tp),
            "fp": int(seg_fp),
            "fn": int(seg_fn),
            "precision": float(seg_prec),
            "recall": float(seg_rec),
            "f1": float(seg_f1),
        },
        "cell_metrics": {
            "tp": int(cell_tp),
            "fp": int(cell_fp),
            "fn": int(cell_fn),
            "precision": float(cell_prec),
            "recall": float(cell_rec),
            "f1": float(cell_f1),
            "mean_iou_matched": float(np.mean(cell_ious)) if cell_ious else 0.0,
        },
        "soft_line_segment_metrics": {
            "alpha": float(alpha),
            "score_sum_matched": float(soft_seg_score_sum),
            "pred_unmatched_count": int(soft_unmatched_pred),
            "pred_unmatched_penalty": float(soft_pred_penalty),
            "raw_score": float(soft_raw),
            "normalized_score": float(soft_norm),
            "accuracy_clipped_0_1": float(soft_accuracy),
            "soft_precision": float(soft_precision),
            "soft_recall": float(soft_recall),
            "matched_gt_count": int(soft_seg_match_gt),
            "matched_pred_count": int(soft_seg_match_pr),
            "gt_segment_count": int(soft_seg_gt_count),
            "pred_segment_count": int(soft_seg_pr_count),
            "candidate_pairs_considered": int(soft_seg_candidates),
        },
        "matched_table_details": matched_table_details,
        "meta": {
            "min_cols_keep": int(min_cols_keep),
            "gt_table_count_raw": int(len(gt_tables_all)),
            "pred_table_count_raw": int(len(pr_tables_all)),
            "gt_table_count": int(len(gt_tables)),
            "pred_table_count": int(len(pr_tables)),
            "matched_tables": int(len(pairs)),
            "params": {
                "table_iou_match_thr": float(table_iou_match_thr),
                "line_tol_abs_px": float(line_tol_abs_px),
                "line_tol_rel": float(line_tol_rel),
                "cell_iou_thr": float(cell_iou_thr),
                "line_soft_alpha": float(alpha),
            },
        },
    }
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate scorecard grid JSON vs ground truth")
    p.add_argument("--gt_json", required=True, help="Ground-truth JSON path")
    p.add_argument("--pred_json", required=True, help="Predicted JSON path")
    p.add_argument("--out_json", default="", help="Optional output report JSON path")
    p.add_argument("--min_cols_keep", type=int, default=9, help="Only evaluate tables with at least this many columns")

    p.add_argument("--table_iou_match_thr", type=float, default=0.20)
    p.add_argument("--line_tol_abs_px", type=float, default=4.0)
    p.add_argument("--line_tol_rel", type=float, default=0.02)
    p.add_argument("--cell_iou_thr", type=float, default=0.75)
    p.add_argument("--line_soft_alpha", type=float, default=0.5)

    p.add_argument("--w_table", type=float, default=0.10)
    p.add_argument("--w_line", type=float, default=0.15)
    p.add_argument("--w_segment", type=float, default=0.25)
    p.add_argument("--w_cell", type=float, default=0.50)
    args = p.parse_args()

    gt_json = Path(args.gt_json)
    pred_json = Path(args.pred_json)
    if not gt_json.exists():
        raise FileNotFoundError(f"GT JSON not found: {gt_json}")
    if not pred_json.exists():
        raise FileNotFoundError(f"Pred JSON not found: {pred_json}")

    report = evaluate(
        gt_json=gt_json,
        pred_json=pred_json,
        min_cols_keep=int(args.min_cols_keep),
        table_iou_match_thr=float(args.table_iou_match_thr),
        line_tol_abs_px=float(args.line_tol_abs_px),
        line_tol_rel=float(args.line_tol_rel),
        cell_iou_thr=float(args.cell_iou_thr),
        line_soft_alpha=float(args.line_soft_alpha),
        w_table=float(args.w_table),
        w_line=float(args.w_line),
        w_segment=float(args.w_segment),
        w_cell=float(args.w_cell),
    )

    print(f"accuracy={report['accuracy']:.6f} ({report['accuracy_percent']:.2f}%)")
    print(
        "f1: "
        f"table={report['components']['table_f1']:.4f} "
        f"line={report['components']['line_f1']:.4f} "
        f"segment={report['components']['segment_f1']:.4f} "
        f"cell={report['components']['cell_f1']:.4f}"
    )
    print(
        "counts: "
        f"gt_tables={report['meta']['gt_table_count']} "
        f"pred_tables={report['meta']['pred_table_count']} "
        f"matched={report['meta']['matched_tables']}"
    )
    sl = report.get("soft_line_segment_metrics", {})
    if isinstance(sl, dict):
        print(
            "soft_line: "
            f"score={float(sl.get('accuracy_clipped_0_1', 0.0)):.4f} "
            f"raw={float(sl.get('raw_score', 0.0)):.3f} "
            f"precision={float(sl.get('soft_precision', 0.0)):.4f} "
            f"recall={float(sl.get('soft_recall', 0.0)):.4f} "
            f"alpha={float(sl.get('alpha', 0.5)):.2f}"
        )

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
