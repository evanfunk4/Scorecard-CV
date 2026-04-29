#!/usr/bin/env python3
"""
Template-transfer refinement using first N trusted labels.

For each target label (index >= start_index):
- Find best matching trusted reference image using ORB + RANSAC homography
- Warp reference label geometry into target image space
- Preserve target topology (table count and per-table line counts) by resampling
  warped reference geometry onto the target's expected counts.

This is intentionally stronger than local "snap" refiners.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


def _poly(raw: Any) -> list[list[int]]:
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


def _transform_points(points: np.ndarray, H: np.ndarray) -> np.ndarray:
    # points Nx2 float32
    pts = points.reshape(-1, 1, 2).astype(np.float32)
    out = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
    return out


def _transform_poly(poly: list[list[int]], H: np.ndarray, w: int, h: int) -> list[list[int]]:
    if len(poly) < 2:
        return []
    pts = np.asarray(poly, dtype=np.float32)
    out = _transform_points(pts, H)
    out_i: list[list[int]] = []
    for x, y in out:
        out_i.append([_clip(_safe_int(x), 0, w - 1), _clip(_safe_int(y), 0, h - 1)])
    return out_i


def _transform_bbox(bb: list[int], H: np.ndarray, w: int, h: int) -> list[int]:
    if not (isinstance(bb, list) and len(bb) >= 4):
        return [0, 0, w - 1, h - 1]
    x0, y0, x1, y1 = [_safe_int(v) for v in bb[:4]]
    corners = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
    out = _transform_points(corners, H)
    xs = [_clip(_safe_int(v), 0, w - 1) for v in out[:, 0]]
    ys = [_clip(_safe_int(v), 0, h - 1) for v in out[:, 1]]
    nx0, nx1 = min(xs), max(xs)
    ny0, ny1 = min(ys), max(ys)
    if nx1 <= nx0:
        nx1 = min(w - 1, nx0 + 8)
    if ny1 <= ny0:
        ny1 = min(h - 1, ny0 + 8)
    return [nx0, ny0, nx1, ny1]


def _map_poly_bbox_to_bbox(
    poly: list[list[int]],
    src_bb: list[int],
    dst_bb: list[int],
    w: int,
    h: int,
) -> list[list[int]]:
    if len(poly) < 2:
        return []
    sx0, sy0, sx1, sy1 = [_safe_int(v) for v in src_bb[:4]]
    dx0, dy0, dx1, dy1 = [_safe_int(v) for v in dst_bb[:4]]
    sw = max(1.0, float(sx1 - sx0))
    sh = max(1.0, float(sy1 - sy0))
    dw = float(dx1 - dx0)
    dh = float(dy1 - dy0)
    out: list[list[int]] = []
    for x, y in poly:
        rx = (float(x) - float(sx0)) / sw
        ry = (float(y) - float(sy0)) / sh
        xx = float(dx0) + rx * dw
        yy = float(dy0) + ry * dh
        out.append([_clip(_safe_int(xx), 0, w - 1), _clip(_safe_int(yy), 0, h - 1)])
    return out


def _resample_vertical_lines(
    source_v_lines: list[list[list[int]]],
    source_kinds: list[str],
    target_count: int,
    y0: int,
    y1: int,
) -> tuple[list[list[list[int]]], list[str]]:
    if target_count <= 0:
        return [], []
    if not source_v_lines:
        return [], []

    items = []
    for i, p in enumerate(source_v_lines):
        kind = source_kinds[i] if i < len(source_kinds) else "line"
        items.append((float(_line_center_x(p)), p, kind))
    items.sort(key=lambda t: t[0])
    xs = [t[0] for t in items]
    kinds = [t[2] for t in items]

    # Select or interpolate centers to requested count.
    if len(xs) >= target_count:
        idx = np.linspace(0, len(xs) - 1, target_count)
        sel = [int(round(i)) for i in idx]
        centers = [xs[i] for i in sel]
        out_kinds = [kinds[i] for i in sel]
    else:
        src_x = np.asarray(xs, dtype=np.float32)
        t = np.linspace(0, len(xs) - 1, target_count).astype(np.float32)
        centers = np.interp(t, np.arange(len(xs), dtype=np.float32), src_x).tolist()
        out_kinds = []
        for j in range(target_count):
            k = kinds[int(round(min(len(xs) - 1, max(0.0, t[j]))))]
            out_kinds.append(k)

    # Build canonical 3-point vertical polylines.
    ym = int(round((y0 + y1) / 2.0))
    out_lines: list[list[list[int]]] = []
    for cx in centers:
        xi = _safe_int(cx)
        out_lines.append([[xi, y0], [xi, ym], [xi, y1]])
    return out_lines, out_kinds


def _resample_separators(
    source_seps: list[dict[str, Any]],
    target_count: int,
    x0: int,
    x1: int,
) -> list[dict[str, Any]]:
    if target_count <= 0:
        return []
    if not source_seps:
        return []

    items = []
    for s in source_seps:
        p = _poly(s.get("polyline", []))
        if len(p) < 2:
            continue
        kind = str(s.get("kind", "line")).strip().lower()
        if kind not in {"line", "boundary"}:
            kind = "line"
        items.append((float(_line_center_y(p)), kind))
    if not items:
        return []
    items.sort(key=lambda t: t[0])
    ys = [t[0] for t in items]
    kinds = [t[1] for t in items]

    if len(ys) >= target_count:
        idx = np.linspace(0, len(ys) - 1, target_count)
        sel = [int(round(i)) for i in idx]
        centers = [ys[i] for i in sel]
        out_kinds = [kinds[i] for i in sel]
    else:
        src_y = np.asarray(ys, dtype=np.float32)
        t = np.linspace(0, len(ys) - 1, target_count).astype(np.float32)
        centers = np.interp(t, np.arange(len(ys), dtype=np.float32), src_y).tolist()
        out_kinds = []
        for j in range(target_count):
            out_kinds.append(kinds[int(round(min(len(ys) - 1, max(0.0, t[j]))))])

    xm = int(round((x0 + x1) / 2.0))
    out: list[dict[str, Any]] = []
    for cy, kind in zip(centers, out_kinds):
        yi = _safe_int(cy)
        out.append({"kind": kind, "polyline": [[x0, yi], [xm, yi], [x1, yi]]})
    return out


@dataclass
class RefRecord:
    stem: str
    image_path: Path
    image_gray: np.ndarray
    label: dict[str, Any]
    kps: Any
    desc: Any


def _orb_features(gray: np.ndarray) -> tuple[Any, Any]:
    orb = cv2.ORB_create(nfeatures=3500, scaleFactor=1.2, nlevels=8, edgeThreshold=15, patchSize=31, fastThreshold=12)
    kps, desc = orb.detectAndCompute(gray, None)
    return kps, desc


def _best_reference_homography(
    tgt_gray: np.ndarray,
    tgt_kps: Any,
    tgt_desc: Any,
    refs: list[RefRecord],
    prefer_table_count: int,
) -> tuple[RefRecord | None, np.ndarray | None, int]:
    if tgt_desc is None or len(tgt_kps) < 12:
        return None, None, 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    best_ref: RefRecord | None = None
    best_H: np.ndarray | None = None
    best_score = -1e9
    best_inliers = 0

    for r in refs:
        if r.desc is None or len(r.kps) < 12:
            continue

        knn = bf.knnMatch(r.desc, tgt_desc, k=2)
        good = []
        for m_n in knn:
            if len(m_n) < 2:
                continue
            m, n = m_n
            if m.distance < 0.76 * n.distance:
                good.append(m)
        if len(good) < 14:
            continue

        src_pts = np.float32([r.kps[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([tgt_kps[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 4.0)
        if H is None or mask is None:
            continue
        inliers = int(mask.ravel().sum())
        if inliers < 12:
            continue

        # Score with mild topology preference.
        topo_bonus = 0.0
        ref_tables = len(r.label.get("tables", []))
        if ref_tables == prefer_table_count:
            topo_bonus += 8.0
        score = float(inliers) + topo_bonus
        if score > best_score:
            best_score = score
            best_ref = r
            best_H = H
            best_inliers = inliers

    return best_ref, best_H, best_inliers


def _closest_reference_by_topology(
    target_label: dict[str, Any],
    refs: list[RefRecord],
) -> RefRecord | None:
    tgt_tabs = target_label.get("tables", [])
    if not isinstance(tgt_tabs, list):
        tgt_tabs = []
    best = None
    best_score = 1e18
    for r in refs:
        rtabs = r.label.get("tables", [])
        if not isinstance(rtabs, list):
            rtabs = []
        score = 0.0
        score += 18.0 * abs(len(rtabs) - len(tgt_tabs))
        for i in range(max(len(rtabs), len(tgt_tabs))):
            if i >= len(rtabs) or i >= len(tgt_tabs):
                score += 14.0
                continue
            rv = len(rtabs[i].get("v_lines", []))
            tv = len(tgt_tabs[i].get("v_lines", []))
            rs = len(rtabs[i].get("separators", []))
            ts = len(tgt_tabs[i].get("separators", []))
            score += 1.5 * abs(rv - tv) + 1.0 * abs(rs - ts)
        if score < best_score:
            best_score = score
            best = r
    return best


def transfer_refine(
    labels_dir: Path,
    out_dir: Path,
    start_index: int = 5,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    js_files = _collect_jsons(labels_dir)
    if not js_files:
        raise RuntimeError(f"No JSON files in {labels_dir}")
    if len(js_files) <= start_index:
        raise RuntimeError(f"Need at least {start_index + 1} JSONs in {labels_dir}")

    ref_files = js_files[:start_index]
    refs: list[RefRecord] = []
    for p in ref_files:
        d = json.loads(p.read_text(encoding="utf-8"))
        ip = Path(str(d.get("image", "")))
        if not ip.exists():
            continue
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        kps, desc = _orb_features(img)
        refs.append(RefRecord(stem=p.stem, image_path=ip, image_gray=img, label=d, kps=kps, desc=desc))
        # Copy trusted refs untouched.
        (out_dir / p.name).write_text(json.dumps(d, indent=2), encoding="utf-8")

    if not refs:
        raise RuntimeError("No usable reference images among trusted first files.")

    for p in js_files[start_index:]:
        d = json.loads(p.read_text(encoding="utf-8"))
        ip = Path(str(d.get("image", "")))
        if not ip.exists():
            (out_dir / p.name).write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue

        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if img is None:
            (out_dir / p.name).write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue
        h, w = img.shape[:2]
        tgt_kps, tgt_desc = _orb_features(img)
        prefer_table_count = len(d.get("tables", []))
        best_ref, H, inliers = _best_reference_homography(img, tgt_kps, tgt_desc, refs, prefer_table_count)
        use_h = best_ref is not None and H is not None and inliers >= 12
        if not use_h:
            best_ref = _closest_reference_by_topology(d, refs)
            H = None
            inliers = 0
        if best_ref is None:
            (out_dir / p.name).write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue

        ref_tables = best_ref.label.get("tables", [])
        if not isinstance(ref_tables, list) or not ref_tables:
            (out_dir / p.name).write_text(json.dumps(d, indent=2), encoding="utf-8")
            continue

        tgt_tables = d.get("tables", [])
        if not isinstance(tgt_tables, list):
            tgt_tables = []

        # Build warped reference tables first.
        warped_tables: list[dict[str, Any]] = []
        for rt in ref_tables:
            if not isinstance(rt, dict):
                continue
            rt_bb = rt.get("table_bbox", [0, 0, w - 1, h - 1])
            if use_h:
                bb = _transform_bbox(rt_bb, H, w, h)
            else:
                bb = [_safe_int(v) for v in rt_bb[:4]]
            v_lines = []
            v_kinds = rt.get("v_line_kinds", [])
            if not isinstance(v_kinds, list):
                v_kinds = []
            for vp in rt.get("v_lines", []):
                pp = _poly(vp)
                if use_h:
                    tp = _transform_poly(pp, H, w, h)
                else:
                    tp = pp
                if len(tp) >= 2:
                    v_lines.append(tp)
            seps = []
            for s in rt.get("separators", []):
                if not isinstance(s, dict):
                    continue
                pp = _poly(s.get("polyline", []))
                if use_h:
                    tp = _transform_poly(pp, H, w, h)
                else:
                    tp = pp
                if len(tp) >= 2:
                    kind = str(s.get("kind", "line")).strip().lower()
                    if kind not in {"line", "boundary"}:
                        kind = "line"
                    seps.append({"kind": kind, "polyline": tp})
            # Sort to canonical ordering.
            v_order = np.argsort([_line_center_x(p) for p in v_lines]).tolist() if v_lines else []
            v_lines = [v_lines[i] for i in v_order] if v_order else v_lines
            v_kinds = [v_kinds[i] if i < len(v_kinds) else "line" for i in v_order] if v_order else v_kinds
            seps = sorted(seps, key=lambda s: _line_center_y(_poly(s.get("polyline", []))))
            warped_tables.append(
                {
                    "table_bbox": bb,
                    "v_lines": v_lines,
                    "v_line_kinds": v_kinds,
                    "separators": seps,
                }
            )

        # Resample warped reference onto target topology table-by-table.
        out_tables: list[dict[str, Any]] = []
        n_tgt = len(tgt_tables)
        n_ref = len(warped_tables)
        for ti in range(n_tgt):
            tt = tgt_tables[ti] if ti < n_tgt else {}
            # map target table index to reference table index by relative position
            if n_tgt <= 1 or n_ref <= 1:
                ri = 0
            else:
                ri = int(round((ti * (n_ref - 1)) / float(max(1, n_tgt - 1))))
                ri = _clip(ri, 0, n_ref - 1)
            rt = warped_tables[ri]

            tb = tt.get("table_bbox", rt.get("table_bbox", [0, 0, w - 1, h - 1]))
            rb = rt.get("table_bbox", [0, 0, w - 1, h - 1])
            if isinstance(tb, list) and len(tb) >= 4:
                # Blend target bbox anchor with warped ref bbox (favor warped ref).
                bx = [_safe_int(v) for v in tb[:4]]
                rx = [_safe_int(v) for v in rb[:4]]
                if use_h:
                    bb = [
                        _clip(_safe_int(0.25 * bx[0] + 0.75 * rx[0]), 0, w - 1),
                        _clip(_safe_int(0.25 * bx[1] + 0.75 * rx[1]), 0, h - 1),
                        _clip(_safe_int(0.25 * bx[2] + 0.75 * rx[2]), 0, w - 1),
                        _clip(_safe_int(0.25 * bx[3] + 0.75 * rx[3]), 0, h - 1),
                    ]
                else:
                    # When homography failed, trust target bbox anchor.
                    bb = [
                        _clip(_safe_int(bx[0]), 0, w - 1),
                        _clip(_safe_int(bx[1]), 0, h - 1),
                        _clip(_safe_int(bx[2]), 0, w - 1),
                        _clip(_safe_int(bx[3]), 0, h - 1),
                    ]
            else:
                bb = rb
            x0, y0, x1, y1 = bb
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0

            tv = tt.get("v_lines", [])
            ts = tt.get("separators", [])
            tgt_v_count = len(tv) if isinstance(tv, list) else len(rt.get("v_lines", []))
            tgt_s_count = len(ts) if isinstance(ts, list) else len(rt.get("separators", []))

            src_v = rt.get("v_lines", [])
            src_s = rt.get("separators", [])
            if not use_h:
                rt_src_bb = rt.get("table_bbox", [0, 0, w - 1, h - 1])
                src_v2 = []
                for vv in src_v:
                    pp = _poly(vv)
                    mp = _map_poly_bbox_to_bbox(pp, rt_src_bb, bb, w, h)
                    if len(mp) >= 2:
                        src_v2.append(mp)
                src_v = src_v2
                src_s2 = []
                for ss in src_s:
                    if not isinstance(ss, dict):
                        continue
                    pp = _poly(ss.get("polyline", []))
                    mp = _map_poly_bbox_to_bbox(pp, rt_src_bb, bb, w, h)
                    if len(mp) >= 2:
                        src_s2.append({"kind": ss.get("kind", "line"), "polyline": mp})
                src_s = src_s2

            v_lines, v_kinds = _resample_vertical_lines(
                source_v_lines=src_v,
                source_kinds=rt.get("v_line_kinds", []),
                target_count=tgt_v_count,
                y0=y0,
                y1=y1,
            )
            seps = _resample_separators(
                source_seps=src_s,
                target_count=tgt_s_count,
                x0=x0,
                x1=x1,
            )

            out_tables.append(
                {
                    "table_bbox": [int(x0), int(y0), int(x1), int(y1)],
                    "v_lines": v_lines,
                    "v_line_kinds": v_kinds if len(v_kinds) == len(v_lines) else ["line"] * len(v_lines),
                    "separators": seps,
                }
            )

        out_d = dict(d)
        out_d["tables"] = out_tables
        (out_dir / p.name).write_text(json.dumps(out_d, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Template-transfer row-separator refinement.")
    ap.add_argument("--labels_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--start_index", type=int, default=5)
    args = ap.parse_args()
    transfer_refine(
        labels_dir=Path(args.labels_dir),
        out_dir=Path(args.out_dir),
        start_index=max(1, int(args.start_index)),
    )


if __name__ == "__main__":
    main()
