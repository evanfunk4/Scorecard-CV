"""Flexible line annotation tool for warped scorecard grids.

This tool reuses existing axis-aligned labels and upgrades them to a flexible
representation where each separator is a 3-point polyline:
  p0 -> p1 -> p2

Workflow:
1) Convert existing labels (ml_labels/*.json) into flex labels.
2) Interactively edit flex labels (move table corners / move line control points).
3) Render masks back to training format for segmentation training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import argparse
import json

import cv2
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _is_base_label_record(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    req = ("image", "table_mask", "v_mask", "h_mask")
    return all(k in data for k in req)


def _collect_base_jsons(labels_dir: Path) -> list[Path]:
    out: list[Path] = []
    for js in sorted(labels_dir.glob("*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _is_base_label_record(data):
            out.append(js)
    return out


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape[:2]
    if xs.size == 0 or ys.size == 0:
        return (0, 0, w - 1, h - 1)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


def _extract_line_positions(mask: np.ndarray, axis: str) -> list[int]:
    if axis == "v":
        proj = (mask > 0).sum(axis=0).astype(np.float32)
        span = mask.shape[0]
    else:
        proj = (mask > 0).sum(axis=1).astype(np.float32)
        span = mask.shape[1]
    if proj.size == 0 or float(proj.max()) <= 0.0:
        return []
    thr = max(2.0, 0.10 * float(span))
    active = np.where(proj >= thr)[0]
    if active.size == 0:
        active = np.where(proj > 0)[0]
    if active.size == 0:
        return []

    out: list[int] = []
    s = int(active[0])
    prev = int(active[0])
    for v in active[1:]:
        vv = int(v)
        if vv == prev + 1:
            prev = vv
            continue
        seg = np.arange(s, prev + 1, dtype=np.int32)
        wts = proj[s : prev + 1]
        c = int(round(float(np.sum(seg * wts) / max(1e-6, np.sum(wts)))))
        out.append(c)
        s = vv
        prev = vv
    seg = np.arange(s, prev + 1, dtype=np.int32)
    wts = proj[s : prev + 1]
    c = int(round(float(np.sum(seg * wts) / max(1e-6, np.sum(wts)))))
    out.append(c)

    out = sorted(set(int(x) for x in out))
    if not out:
        return []
    dedup = [out[0]]
    for v in out[1:]:
        if abs(v - dedup[-1]) >= 4:
            dedup.append(v)
    return dedup


def _clip_bbox(bbox: list[int], w: int, h: int) -> list[int]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = int(np.clip(x0, 0, max(0, w - 1)))
    x1 = int(np.clip(x1, 0, max(0, w - 1)))
    y0 = int(np.clip(y0, 0, max(0, h - 1)))
    y1 = int(np.clip(y1, 0, max(0, h - 1)))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def _build_v_polyline(x: int, y0: int, y1: int) -> list[list[int]]:
    ym = int(round(0.5 * (y0 + y1)))
    return [[int(x), int(y0)], [int(x), int(ym)], [int(x), int(y1)]]


def _build_h_polyline(y: int, x0: int, x1: int) -> list[list[int]]:
    xm = int(round(0.5 * (x0 + x1)))
    return [[int(x0), int(y)], [int(xm), int(y)], [int(x1), int(y)]]


def _normalize_v_polyline(line: list[list[int]], bbox: list[int]) -> list[list[int]]:
    x0, y0, x1, y1 = bbox
    if len(line) != 3:
        return _build_v_polyline(int(round(0.5 * (x0 + x1))), y0, y1)
    pts = [[int(np.clip(p[0], x0, x1)), int(np.clip(p[1], y0, y1))] for p in line]
    # Keep vertical ordering, but do not force endpoints to table edges.
    pts.sort(key=lambda p: (p[1], p[0]))
    p0, p1, p2 = pts
    return [p0, p1, p2]


def _normalize_h_polyline(line: list[list[int]], bbox: list[int]) -> list[list[int]]:
    x0, y0, x1, y1 = bbox
    if len(line) != 3:
        return _build_h_polyline(int(round(0.5 * (y0 + y1))), x0, x1)
    pts = [[int(np.clip(p[0], x0, x1)), int(np.clip(p[1], y0, y1))] for p in line]
    # Keep horizontal ordering, but do not force endpoints to table edges.
    pts.sort(key=lambda p: (p[0], p[1]))
    p0, p1, p2 = pts
    return [p0, p1, p2]


def _line_distance_sq(line: list[list[int]], x: int, y: int) -> float:
    def seg_dist_sq(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx * wx + vy * wy
        if c1 <= 0:
            dx, dy = px - ax, py - ay
            return dx * dx + dy * dy
        c2 = vx * vx + vy * vy
        if c2 <= 1e-9:
            dx, dy = px - ax, py - ay
            return dx * dx + dy * dy
        t = max(0.0, min(1.0, c1 / c2))
        qx, qy = ax + t * vx, ay + t * vy
        dx, dy = px - qx, py - qy
        return dx * dx + dy * dy

    p0, p1, p2 = line
    d1 = seg_dist_sq(float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), float(x), float(y))
    d2 = seg_dist_sq(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), float(x), float(y))
    return min(d1, d2)


def convert_to_flex(base_labels_dir: Path, flex_labels_dir: Path, overwrite: bool = False) -> None:
    _ensure_dir(flex_labels_dir)
    base_jsons = _collect_base_jsons(base_labels_dir)
    if not base_jsons:
        print(f"No base labels found in {base_labels_dir}")
        return

    for js in base_jsons:
        stem = js.stem
        out_js = flex_labels_dir / f"{stem}.json"
        if out_js.exists() and not overwrite:
            print(f"[skip] {stem}")
            continue

        rec = json.loads(js.read_text(encoding="utf-8"))
        image_path = Path(rec["image"])
        table_mask = cv2.imread(str(rec["table_mask"]), cv2.IMREAD_GRAYSCALE)
        vmask = cv2.imread(str(rec["v_mask"]), cv2.IMREAD_GRAYSCALE)
        hmask = cv2.imread(str(rec["h_mask"]), cv2.IMREAD_GRAYSCALE)
        if table_mask is None or vmask is None or hmask is None:
            print(f"[warn] failed to load masks for {stem}")
            continue
        h, w = table_mask.shape[:2]

        bbox = list(_mask_bbox(table_mask))
        bbox = _clip_bbox(bbox, w, h)
        x0, y0, x1, y1 = bbox

        v_pos = _extract_line_positions(vmask, axis="v")
        h_pos = _extract_line_positions(hmask, axis="h")

        v_lines = [_build_v_polyline(int(np.clip(x, x0, x1)), y0, y1) for x in v_pos]
        h_lines = [_build_h_polyline(int(np.clip(y, y0, y1)), x0, x1) for y in h_pos]

        out = {
            "version": "flex_v1",
            "source_json": str(js.resolve()),
            "source_image": rec.get("source_image", ""),
            "image": str(image_path.resolve()),
            "table_bbox": [int(v) for v in bbox],
            "v_lines": v_lines,
            "h_lines": h_lines,
        }
        out_js.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"[ok] {stem}")


def _load_flex_record(js: Path) -> dict[str, Any]:
    data = json.loads(js.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid JSON in {js}")
    if "image" not in data or "table_bbox" not in data or "v_lines" not in data or "h_lines" not in data:
        raise RuntimeError(f"Missing keys in {js}")
    return data


def _save_flex_record(js: Path, rec: dict[str, Any]) -> None:
    js.write_text(json.dumps(rec, indent=2), encoding="utf-8")


def _collect_flex_jsons(flex_labels_dir: Path) -> list[Path]:
    out = []
    for js in sorted(flex_labels_dir.glob("*.json")):
        try:
            rec = _load_flex_record(js)
        except Exception:
            continue
        if isinstance(rec, dict):
            out.append(js)
    return out


@dataclass
class _EditorState:
    records: list[Path]
    idx: int = 0
    mode: str = "table"  # table | v | h
    dragging: bool = False
    drag_kind: str = ""  # corner | line_point | line_whole
    drag_idx: int = -1
    drag_point_idx: int = -1
    add_mode: bool = False
    delete_mode: bool = False
    last_mouse: tuple[int, int] = (0, 0)


class FlexLabelEditor:
    def __init__(self, flex_labels_dir: Path, start_index: int = 0):
        self.flex_labels_dir = flex_labels_dir
        self.records = _collect_flex_jsons(flex_labels_dir)
        if not self.records:
            raise RuntimeError(f"No flex labels in {flex_labels_dir}")
        self.state = _EditorState(records=self.records, idx=max(0, min(start_index, len(self.records) - 1)))

        self.window = "scorecard_flex_label_tool"
        self.display_pad = 44
        self.corner_tol = 56
        self.point_tol = 16
        self.line_tol = 20

        self.current_json: Optional[Path] = None
        self.rec: dict[str, Any] | None = None
        self.image: np.ndarray | None = None

    def _sanitize(self) -> None:
        if self.image is None or self.rec is None:
            return
        h, w = self.image.shape[:2]
        bbox = self.rec.get("table_bbox", [0, 0, w - 1, h - 1])
        bbox = _clip_bbox([int(v) for v in bbox], w, h)
        self.rec["table_bbox"] = bbox

        v_new: list[list[list[int]]] = []
        for ln in self.rec.get("v_lines", []):
            try:
                v_new.append(_normalize_v_polyline([[int(p[0]), int(p[1])] for p in ln], bbox))
            except Exception:
                continue
        h_new: list[list[list[int]]] = []
        for ln in self.rec.get("h_lines", []):
            try:
                h_new.append(_normalize_h_polyline([[int(p[0]), int(p[1])] for p in ln], bbox))
            except Exception:
                continue

        v_new.sort(key=lambda ln: float(np.mean([p[0] for p in ln])))
        h_new.sort(key=lambda ln: float(np.mean([p[1] for p in ln])))
        self.rec["v_lines"] = v_new
        self.rec["h_lines"] = h_new

    def _load_current(self) -> None:
        self.current_json = self.state.records[self.state.idx]
        self.rec = _load_flex_record(self.current_json)
        self.image = cv2.imread(str(self.rec["image"]), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Failed to load image {self.rec['image']}")
        self._sanitize()
        self.state.dragging = False
        self.state.drag_kind = ""
        self.state.drag_idx = -1
        self.state.drag_point_idx = -1
        self.state.add_mode = False
        self.state.delete_mode = False

    def _save_current(self) -> None:
        if self.current_json is None or self.rec is None:
            return
        self._sanitize()
        _save_flex_record(self.current_json, self.rec)

    def _table_corners(self) -> list[tuple[int, int]]:
        x0, y0, x1, y1 = [int(v) for v in self.rec["table_bbox"]]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _nearest_corner_idx(self, x: int, y: int) -> int:
        best = -1
        best_d2 = 1e18
        for i, (cx, cy) in enumerate(self._table_corners()):
            d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
            if d2 < best_d2:
                best_d2 = d2
                best = i
        return best if best_d2 <= self.corner_tol * self.corner_tol else -1

    def _nearest_line_point(self, lines: list[list[list[int]]], x: int, y: int) -> tuple[int, int]:
        best_li = -1
        best_pi = -1
        best_d2 = 1e18
        for li, ln in enumerate(lines):
            for pi, p in enumerate(ln):
                d2 = (int(p[0]) - x) * (int(p[0]) - x) + (int(p[1]) - y) * (int(p[1]) - y)
                if d2 < best_d2:
                    best_d2 = d2
                    best_li = li
                    best_pi = pi
        if best_d2 <= self.point_tol * self.point_tol:
            return best_li, best_pi
        return -1, -1

    def _nearest_line_idx(self, lines: list[list[list[int]]], x: int, y: int) -> int:
        best = -1
        best_d2 = 1e18
        for li, ln in enumerate(lines):
            d2 = _line_distance_sq(ln, x, y)
            if d2 < best_d2:
                best_d2 = d2
                best = li
        return best if best_d2 <= self.line_tol * self.line_tol else -1

    def _render(self) -> np.ndarray:
        vis = self.image.copy()
        x0, y0, x1, y1 = [int(v) for v in self.rec["table_bbox"]]

        # Table overlay and border
        tmp = vis.copy()
        cv2.rectangle(tmp, (x0, y0), (x1, y1), (0, 180, 0), -1)
        vis = cv2.addWeighted(vis, 0.86, tmp, 0.14, 0.0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 0), 2)
        for cx, cy in self._table_corners():
            cv2.circle(vis, (cx, cy), 8, (0, 255, 255), -1)
            cv2.circle(vis, (cx, cy), 11, (0, 180, 255), 1)

        # Flexible lines
        for ln in self.rec["v_lines"]:
            pts = np.array(ln, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (255, 80, 80), 2, cv2.LINE_AA)
            for p in ln:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (255, 180, 180), -1)
        for ln in self.rec["h_lines"]:
            pts = np.array(ln, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (80, 80, 255), 2, cv2.LINE_AA)
            for p in ln:
                cv2.circle(vis, (int(p[0]), int(p[1])), 4, (180, 180, 255), -1)

        stem = self.current_json.stem if self.current_json else ""
        add_tag = " add=ON" if self.state.add_mode else ""
        del_tag = " del=ON" if self.state.delete_mode else ""
        t1 = f"{self.state.idx+1}/{len(self.state.records)} {stem}"
        t2 = f"mode={self.state.mode}{add_tag}{del_tag} v={len(self.rec['v_lines'])} h={len(self.rec['h_lines'])}"
        t3 = "keys: 1 table 2 vertical 3 horizontal | drag points/lines (endpoints can make partial segments) | a add-line | d delete-line | s save | n/p next/prev | q quit"

        pad = int(self.display_pad)
        h, w = vis.shape[:2]
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3), (42, 42, 42), dtype=np.uint8)
        canvas[pad : pad + h, pad : pad + w] = vis
        cv2.rectangle(canvas, (pad - 1, pad - 1), (pad + w, pad + h), (130, 130, 130), 1)
        cv2.putText(canvas, t1, (pad + 12, pad + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, t2, (pad + 12, pad + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, t3, (pad + 12, pad + h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _constrain_line_after_move(self, axis: str, line: list[list[int]]) -> list[list[int]]:
        bbox = [int(v) for v in self.rec["table_bbox"]]
        if axis == "v":
            return _normalize_v_polyline(line, bbox)
        return _normalize_h_polyline(line, bbox)

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.image is None or self.rec is None:
            return
        pad = int(self.display_pad)
        h, w = self.image.shape[:2]
        x = int(np.clip(x - pad, 0, w - 1))
        y = int(np.clip(y - pad, 0, h - 1))
        self._sanitize()
        x0, y0, x1, y1 = [int(v) for v in self.rec["table_bbox"]]

        lines_key = "v_lines" if self.state.mode == "v" else "h_lines"
        axis = "v" if self.state.mode == "v" else "h"

        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.last_mouse = (x, y)
            if self.state.mode == "table":
                self.state.add_mode = False
                self.state.delete_mode = False
                cidx = self._nearest_corner_idx(x, y)
                if cidx >= 0:
                    self.state.dragging = True
                    self.state.drag_kind = "corner"
                    self.state.drag_idx = cidx
                return

            lines = self.rec[lines_key]
            if self.state.delete_mode:
                li = self._nearest_line_idx(lines, x, y)
                if li >= 0:
                    lines.pop(li)
                self.state.delete_mode = False
                return

            if self.state.add_mode:
                if axis == "v":
                    nx = int(np.clip(x, x0, x1))
                    lines.append(_build_v_polyline(nx, y0, y1))
                else:
                    ny = int(np.clip(y, y0, y1))
                    lines.append(_build_h_polyline(ny, x0, x1))
                self.state.add_mode = False
                self._sanitize()
                return

            li, pi = self._nearest_line_point(lines, x, y)
            if li >= 0 and pi >= 0:
                self.state.dragging = True
                self.state.drag_kind = "line_point"
                self.state.drag_idx = li
                self.state.drag_point_idx = pi
                return

            li = self._nearest_line_idx(lines, x, y)
            if li >= 0:
                self.state.dragging = True
                self.state.drag_kind = "line_whole"
                self.state.drag_idx = li
                self.state.drag_point_idx = -1
                return

        elif event == cv2.EVENT_MOUSEMOVE and self.state.dragging:
            if self.state.drag_kind == "corner" and self.state.mode == "table":
                cx = int(np.clip(x, 0, w - 1))
                cy = int(np.clip(y, 0, h - 1))
                tx0, ty0, tx1, ty1 = [int(v) for v in self.rec["table_bbox"]]
                if self.state.drag_idx == 0:
                    tx0 = int(np.clip(cx, 0, tx1))
                    ty0 = int(np.clip(cy, 0, ty1))
                elif self.state.drag_idx == 1:
                    tx1 = int(np.clip(cx, tx0, w - 1))
                    ty0 = int(np.clip(cy, 0, ty1))
                elif self.state.drag_idx == 2:
                    tx1 = int(np.clip(cx, tx0, w - 1))
                    ty1 = int(np.clip(cy, ty0, h - 1))
                elif self.state.drag_idx == 3:
                    tx0 = int(np.clip(cx, 0, tx1))
                    ty1 = int(np.clip(cy, ty0, h - 1))
                self.rec["table_bbox"] = [tx0, ty0, tx1, ty1]
                self._sanitize()
                return

            if self.state.mode not in ("v", "h"):
                return

            lines = self.rec["v_lines"] if self.state.mode == "v" else self.rec["h_lines"]
            if not (0 <= self.state.drag_idx < len(lines)):
                return
            ln = [[int(p[0]), int(p[1])] for p in lines[self.state.drag_idx]]
            px, py = self.state.last_mouse
            dx = int(x - px)
            dy = int(y - py)
            self.state.last_mouse = (x, y)

            if self.state.drag_kind == "line_point" and 0 <= self.state.drag_point_idx < 3:
                pi = self.state.drag_point_idx
                ln[pi][0] += dx
                ln[pi][1] += dy
            elif self.state.drag_kind == "line_whole":
                for p in ln:
                    p[0] += dx
                    p[1] += dy

            lines[self.state.drag_idx] = self._constrain_line_after_move("v" if self.state.mode == "v" else "h", ln)
            self._sanitize()

        elif event == cv2.EVENT_LBUTTONUP:
            self.state.dragging = False
            self.state.drag_kind = ""
            self.state.drag_idx = -1
            self.state.drag_point_idx = -1

    def run(self) -> None:
        self._load_current()
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self._mouse_cb)

        while True:
            vis = self._render()
            cv2.imshow(self.window, vis)
            k = cv2.waitKey(20) & 0xFF
            if k == 255:
                continue
            if k == ord("q"):
                self._save_current()
                break
            if k == ord("s"):
                self._save_current()
                print(f"[saved] {self.current_json.name}")
                continue
            if k == ord("1"):
                self.state.mode = "table"
                self.state.add_mode = False
                self.state.delete_mode = False
                continue
            if k == ord("2"):
                self.state.mode = "v"
                self.state.add_mode = False
                self.state.delete_mode = False
                continue
            if k == ord("3"):
                self.state.mode = "h"
                self.state.add_mode = False
                self.state.delete_mode = False
                continue
            if k == ord("a"):
                if self.state.mode in ("v", "h"):
                    self.state.add_mode = not self.state.add_mode
                    if self.state.add_mode:
                        self.state.delete_mode = False
                continue
            if k == ord("d"):
                if self.state.mode in ("v", "h"):
                    self.state.delete_mode = not self.state.delete_mode
                    if self.state.delete_mode:
                        self.state.add_mode = False
                continue
            if k == ord("n"):
                self._save_current()
                self.state.idx = min(len(self.state.records) - 1, self.state.idx + 1)
                self._load_current()
                continue
            if k == ord("p"):
                self._save_current()
                self.state.idx = max(0, self.state.idx - 1)
                self._load_current()
                continue

        cv2.destroyWindow(self.window)


def render_flex_masks(
    flex_labels_dir: Path,
    out_labels_dir: Path,
    line_thickness: int = 3,
    overwrite: bool = True,
) -> None:
    _ensure_dir(out_labels_dir)
    img_dir = out_labels_dir / "images"
    msk_dir = out_labels_dir / "masks"
    _ensure_dir(img_dir)
    _ensure_dir(msk_dir)

    recs = _collect_flex_jsons(flex_labels_dir)
    if not recs:
        print(f"No flex labels in {flex_labels_dir}")
        return

    for js in recs:
        stem = js.stem
        out_json = out_labels_dir / f"{stem}.json"
        if out_json.exists() and not overwrite:
            print(f"[skip] {stem}")
            continue

        rec = _load_flex_record(js)
        img = cv2.imread(str(rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] failed image for {stem}")
            continue
        h, w = img.shape[:2]
        bbox = _clip_bbox([int(v) for v in rec["table_bbox"]], w, h)

        table = np.zeros((h, w), dtype=np.uint8)
        vmask = np.zeros((h, w), dtype=np.uint8)
        hmask = np.zeros((h, w), dtype=np.uint8)

        x0, y0, x1, y1 = bbox
        cv2.rectangle(table, (x0, y0), (x1, y1), 255, -1)

        for ln in rec.get("v_lines", []):
            nln = _normalize_v_polyline([[int(p[0]), int(p[1])] for p in ln], bbox)
            pts = np.array(nln, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vmask, [pts], False, 255, int(max(1, line_thickness)), cv2.LINE_AA)

        for ln in rec.get("h_lines", []):
            nln = _normalize_h_polyline([[int(p[0]), int(p[1])] for p in ln], bbox)
            pts = np.array(nln, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(hmask, [pts], False, 255, int(max(1, line_thickness)), cv2.LINE_AA)

        p_img = img_dir / f"{stem}.png"
        p_t = msk_dir / f"{stem}_table.png"
        p_v = msk_dir / f"{stem}_v.png"
        p_h = msk_dir / f"{stem}_h.png"

        cv2.imwrite(str(p_img), img)
        cv2.imwrite(str(p_t), table)
        cv2.imwrite(str(p_v), vmask)
        cv2.imwrite(str(p_h), hmask)

        out_rec = {
            "source_image": rec.get("source_image", ""),
            "image": str(p_img.resolve()),
            "table_mask": str(p_t.resolve()),
            "v_mask": str(p_v.resolve()),
            "h_mask": str(p_h.resolve()),
            "source_flex_json": str(js.resolve()),
        }
        out_json.write_text(json.dumps(out_rec, indent=2), encoding="utf-8")
        print(f"[ok] {stem}")

    # index.jsonl for training
    lines = []
    for js in sorted(out_labels_dir.glob("*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _is_base_label_record(data):
            lines.append(json.dumps(data))
    (out_labels_dir / "index.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[ok] wrote {(out_labels_dir / 'index.jsonl')}")


def write_flex_overlays(flex_labels_dir: Path, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    recs = _collect_flex_jsons(flex_labels_dir)
    for js in recs:
        rec = _load_flex_record(js)
        img = cv2.imread(str(rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]
        bbox = _clip_bbox([int(v) for v in rec["table_bbox"]], w, h)
        x0, y0, x1, y1 = bbox
        vis = img.copy()

        t = vis.copy()
        cv2.rectangle(t, (x0, y0), (x1, y1), (0, 180, 0), -1)
        vis = cv2.addWeighted(vis, 0.86, t, 0.14, 0.0)
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 0), 2)

        for ln in rec.get("v_lines", []):
            pts = np.array(_normalize_v_polyline([[int(p[0]), int(p[1])] for p in ln], bbox), dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (255, 80, 80), 2, cv2.LINE_AA)
        for ln in rec.get("h_lines", []):
            pts = np.array(_normalize_h_polyline([[int(p[0]), int(p[1])] for p in ln], bbox), dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, (80, 80, 255), 2, cv2.LINE_AA)

        cv2.putText(vis, js.stem, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2, cv2.LINE_AA)
        out = out_dir / f"{js.stem}_overlay.png"
        cv2.imwrite(str(out), vis)
    print(f"[ok] wrote overlays to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Flexible scorecard annotation tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Convert existing base labels to flex labels")
    c.add_argument("--base_labels_dir", default="ml_labels")
    c.add_argument("--flex_labels_dir", default="ml_labels_flex")
    c.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("edit", help="Edit flex labels interactively")
    e.add_argument("--flex_labels_dir", default="ml_labels_flex")
    e.add_argument("--start", type=int, default=0)

    r = sub.add_parser("render", help="Render flex labels into mask dataset for training")
    r.add_argument("--flex_labels_dir", default="ml_labels_flex")
    r.add_argument("--out_labels_dir", default="ml_labels_flex_rendered")
    r.add_argument("--line_thickness", type=int, default=3)
    r.add_argument("--no_overwrite", action="store_true")

    o = sub.add_parser("overlay", help="Write overlay previews for flex labels")
    o.add_argument("--flex_labels_dir", default="ml_labels_flex")
    o.add_argument("--out_dir", default=None)

    args = p.parse_args()

    if args.cmd == "convert":
        convert_to_flex(
            base_labels_dir=Path(args.base_labels_dir),
            flex_labels_dir=Path(args.flex_labels_dir),
            overwrite=bool(args.overwrite),
        )
        return

    if args.cmd == "edit":
        ed = FlexLabelEditor(Path(args.flex_labels_dir), start_index=int(args.start))
        ed.run()
        return

    if args.cmd == "render":
        render_flex_masks(
            flex_labels_dir=Path(args.flex_labels_dir),
            out_labels_dir=Path(args.out_labels_dir),
            line_thickness=int(args.line_thickness),
            overwrite=not bool(args.no_overwrite),
        )
        return

    if args.cmd == "overlay":
        out_dir = Path(args.out_dir) if args.out_dir else (Path(args.flex_labels_dir) / "overlays")
        write_flex_overlays(Path(args.flex_labels_dir), out_dir)
        return


if __name__ == "__main__":
    main()
