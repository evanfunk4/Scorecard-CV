"""Row-separator labeling tool for scorecard grids.

Purpose:
- Label horizontal row separators, including separators that are only color
  transitions (no drawn line).
- Keep existing table bbox + vertical lines from flex labels for context.
- Export masks for:
  - hard horizontal lines
  - soft row-boundary separators
  - union horizontal mask (backward compatibility)

JSON format (rowsep_v2):
{
  "version": "rowsep_v2",
  "source_flex_json": "...",
  "image": ".../clean1.png",
  "tables": [
    {
      "table_bbox": [x0, y0, x1, y1],
      "v_lines": [[[x,y],[x,y],[x,y]], ...],
      "separators": [
        {"polyline": [[x,y],[x,y],[x,y]], "kind": "line"},
        {"polyline": [[x,y],[x,y],[x,y]], "kind": "boundary"}
      ]
    }
  ]
}
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple
import argparse
import json

import cv2
import numpy as np


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def _build_h_polyline(y: int, x0: int, x1: int) -> list[list[int]]:
    xm = int(round(0.5 * (x0 + x1)))
    return [[int(x0), int(y)], [int(xm), int(y)], [int(x1), int(y)]]


def _normalize_h_polyline(line: list[list[int]], bbox: list[int]) -> list[list[int]]:
    x0, y0, x1, y1 = bbox
    if len(line) != 3:
        return _build_h_polyline(int(round(0.5 * (y0 + y1))), x0, x1)
    pts = []
    for p in line:
        px = int(np.clip(int(p[0]), x0, x1))
        py = int(np.clip(int(p[1]), y0, y1))
        pts.append([px, py])
    pts.sort(key=lambda q: (q[0], q[1]))
    return [pts[0], pts[1], pts[2]]


def _normalize_v_polyline(line: list[list[int]], bbox: list[int]) -> list[list[int]]:
    x0, y0, x1, y1 = bbox
    if len(line) != 3:
        x = int(round(0.5 * (x0 + x1)))
        ym = int(round(0.5 * (y0 + y1)))
        return [[x, y0], [x, ym], [x, y1]]
    pts = []
    for p in line:
        px = int(np.clip(int(p[0]), x0, x1))
        py = int(np.clip(int(p[1]), y0, y1))
        pts.append([px, py])
    pts.sort(key=lambda q: (q[1], q[0]))
    return [pts[0], pts[1], pts[2]]


def _line_distance_sq(line: list[list[int]], x: int, y: int) -> float:
    def _seg_dist_sq(ax: float, ay: float, bx: float, by: float, px: float, py: float) -> float:
        vx, vy = bx - ax, by - ay
        wx, wy = px - ax, py - ay
        c1 = vx * wx + vy * wy
        if c1 <= 0.0:
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
    d1 = _seg_dist_sq(float(p0[0]), float(p0[1]), float(p1[0]), float(p1[1]), float(x), float(y))
    d2 = _seg_dist_sq(float(p1[0]), float(p1[1]), float(p2[0]), float(p2[1]), float(x), float(y))
    return min(d1, d2)


def _load_flex_record(js: Path) -> dict[str, Any]:
    data = json.loads(js.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid JSON in {js}")
    req = ("image", "table_bbox", "v_lines", "h_lines")
    if not all(k in data for k in req):
        raise RuntimeError(f"Missing keys in {js}")
    return data


def _load_rowsep_record(js: Path) -> dict[str, Any]:
    data = json.loads(js.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"Invalid JSON in {js}")
    if "image" not in data:
        raise RuntimeError(f"Missing keys in {js}")

    # Backward compatible reader: accept v1 single-table records.
    if "tables" in data and isinstance(data["tables"], list):
        tables = data["tables"]
    elif all(k in data for k in ("table_bbox", "v_lines", "separators")):
        tables = [
            {
                "table_bbox": data["table_bbox"],
                "v_lines": data["v_lines"],
                "separators": data["separators"],
            }
        ]
    else:
        raise RuntimeError(f"Missing keys in {js}")

    out_tables: list[dict[str, Any]] = []
    for t in tables:
        if not isinstance(t, dict):
            continue
        bbox = t.get("table_bbox", [0, 0, 1, 1])
        v_lines = t.get("v_lines", [])
        seps = t.get("separators", [])
        out_tables.append(
            {
                "table_bbox": [int(v) for v in bbox[:4]],
                "v_lines": v_lines if isinstance(v_lines, list) else [],
                "separators": seps if isinstance(seps, list) else [],
            }
        )
    if not out_tables:
        raise RuntimeError(f"No valid tables in {js}")

    return {
        "version": "rowsep_v2",
        "source_flex_json": str(data.get("source_flex_json", "")),
        "image": str(data["image"]),
        "tables": out_tables,
    }


def _save_json(js: Path, payload: dict[str, Any]) -> None:
    js.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _collect_rowsep_jsons(rowsep_labels_dir: Path) -> list[Path]:
    out: list[Path] = []
    for js in sorted(rowsep_labels_dir.glob("*.json")):
        try:
            _ = _load_rowsep_record(js)
        except Exception:
            continue
        out.append(js)
    return out


def bootstrap_from_flex(
    flex_labels_dir: Path,
    rowsep_labels_dir: Path,
    overwrite: bool = False,
) -> None:
    _ensure_dir(rowsep_labels_dir)
    flex_jsons = sorted(flex_labels_dir.glob("*.json"))
    if not flex_jsons:
        print(f"No flex labels found in {flex_labels_dir}")
        return

    for js in flex_jsons:
        stem = js.stem
        out_js = rowsep_labels_dir / f"{stem}.json"
        if out_js.exists() and not overwrite:
            print(f"[skip] {stem}")
            continue

        rec = _load_flex_record(js)
        image_path = Path(str(rec["image"]))
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] image missing for {stem}: {image_path}")
            continue
        h, w = img.shape[:2]
        bbox = _clip_bbox([int(v) for v in rec.get("table_bbox", [0, 0, w - 1, h - 1])], w, h)

        v_lines: list[list[list[int]]] = []
        for ln in rec.get("v_lines", []):
            try:
                v_lines.append(_normalize_v_polyline([[int(p[0]), int(p[1])] for p in ln], bbox))
            except Exception:
                continue
        v_lines.sort(key=lambda ln: float(np.mean([p[0] for p in ln])))

        separators: list[dict[str, Any]] = []
        for ln in rec.get("h_lines", []):
            try:
                hln = _normalize_h_polyline([[int(p[0]), int(p[1])] for p in ln], bbox)
            except Exception:
                continue
            separators.append({"polyline": hln, "kind": "line"})
        separators.sort(key=lambda item: float(np.mean([p[1] for p in item["polyline"]])))

        out = {
            "version": "rowsep_v2",
            "source_flex_json": str(js.resolve()),
            "image": str(image_path.resolve()),
            "tables": [
                {
                    "table_bbox": [int(v) for v in bbox],
                    "v_lines": v_lines,
                    "separators": separators,
                }
            ],
        }
        _save_json(out_js, out)
        print(f"[ok] {stem}")


@dataclass
class _EditorState:
    records: list[Path]
    idx: int = 0
    mode: str = "sep"  # table | sep | v
    table_idx: int = 0
    dragging: bool = False
    drag_kind: str = ""  # corner | point | line
    drag_table_idx: int = -1
    drag_idx: int = -1
    drag_point_idx: int = -1
    add_mode: bool = False
    delete_mode: bool = False
    toggle_kind_mode: bool = False
    split_mode: bool = False
    move_table_mode: bool = False
    last_mouse: tuple[int, int] = (0, 0)


class RowSepLabelEditor:
    def __init__(self, rowsep_labels_dir: Path, start_index: int = 0):
        self.rowsep_labels_dir = rowsep_labels_dir
        self.records = _collect_rowsep_jsons(rowsep_labels_dir)
        if not self.records:
            raise RuntimeError(f"No row-separator labels in {rowsep_labels_dir}")
        self.state = _EditorState(records=self.records, idx=max(0, min(start_index, len(self.records) - 1)))

        self.window = "scorecard_rowsep_label_tool"
        self.display_pad = 44
        self.corner_tol = 56
        self.point_tol = 16
        self.line_tol = 20

        self.current_json: Optional[Path] = None
        self.rec: Optional[dict[str, Any]] = None
        self.image: Optional[np.ndarray] = None

    def _table_count(self) -> int:
        if self.rec is None:
            return 0
        return len(self.rec.get("tables", []))

    def _active_table(self) -> dict[str, Any]:
        return self.rec["tables"][self.state.table_idx]

    def _sanitize(self) -> None:
        if self.rec is None or self.image is None:
            return
        h, w = self.image.shape[:2]
        tabs = self.rec.get("tables", [])
        if not isinstance(tabs, list):
            tabs = []
        if not tabs:
            tabs = [{"table_bbox": [0, 0, w - 1, h - 1], "v_lines": [], "separators": []}]

        out_tabs: list[dict[str, Any]] = []
        for t in tabs:
            if not isinstance(t, dict):
                continue
            bbox = _clip_bbox([int(v) for v in t.get("table_bbox", [0, 0, w - 1, h - 1])], w, h)

            v_new: list[list[list[int]]] = []
            for ln in t.get("v_lines", []):
                try:
                    nln = _normalize_v_polyline([[int(p[0]), int(p[1])] for p in ln], bbox)
                    v_new.append(nln)
                except Exception:
                    continue
            v_new.sort(key=lambda ln: float(np.mean([p[0] for p in ln])))

            s_new: list[dict[str, Any]] = []
            for item in t.get("separators", []):
                try:
                    poly = item.get("polyline", [])
                    kind = str(item.get("kind", "line")).strip().lower()
                    if kind not in ("line", "boundary"):
                        kind = "line"
                    nln = _normalize_h_polyline([[int(p[0]), int(p[1])] for p in poly], bbox)
                    s_new.append({"polyline": nln, "kind": kind})
                except Exception:
                    continue
            s_new.sort(key=lambda item: float(np.mean([p[1] for p in item["polyline"]])))

            out_tabs.append({"table_bbox": bbox, "v_lines": v_new, "separators": s_new})

        if not out_tabs:
            out_tabs = [{"table_bbox": [0, 0, w - 1, h - 1], "v_lines": [], "separators": []}]

        self.rec["tables"] = out_tabs
        self.state.table_idx = int(np.clip(self.state.table_idx, 0, len(out_tabs) - 1))

    def _load_current(self) -> None:
        self.current_json = self.state.records[self.state.idx]
        self.rec = _load_rowsep_record(self.current_json)
        self.image = cv2.imread(str(self.rec["image"]), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Failed to load image: {self.rec['image']}")
        self._sanitize()
        self.state.dragging = False
        self.state.drag_kind = ""
        self.state.drag_table_idx = -1
        self.state.drag_idx = -1
        self.state.drag_point_idx = -1
        self.state.add_mode = False
        self.state.delete_mode = False
        self.state.toggle_kind_mode = False
        self.state.split_mode = False
        self.state.move_table_mode = False

    def _save_current(self) -> None:
        if self.current_json is None or self.rec is None:
            return
        self._sanitize()
        _save_json(self.current_json, self.rec)

    def _table_corners(self, table_idx: int) -> list[tuple[int, int]]:
        x0, y0, x1, y1 = [int(v) for v in self.rec["tables"][table_idx]["table_bbox"]]
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]

    def _nearest_corner(self, x: int, y: int) -> Tuple[int, int]:
        best_t = -1
        best_c = -1
        best_d2 = 1e18
        for ti in range(self._table_count()):
            for ci, (cx, cy) in enumerate(self._table_corners(ti)):
                d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
                if d2 < best_d2:
                    best_d2 = d2
                    best_t = ti
                    best_c = ci
        if best_d2 <= self.corner_tol * self.corner_tol:
            return best_t, best_c
        return -1, -1

    def _table_at_point(self, x: int, y: int) -> int:
        # Prefer direct containment so a click inside a grid activates it.
        for ti, t in enumerate(self.rec.get("tables", [])):
            x0, y0, x1, y1 = [int(v) for v in t["table_bbox"]]
            if x0 <= x <= x1 and y0 <= y <= y1:
                return ti
        return -1

    def _nearest_line_point(self, lines: list[list[list[int]]], x: int, y: int) -> Tuple[int, int]:
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

    def _polyline_y_at_x(self, line: list[list[int]], xq: int) -> int:
        pts = sorted([[int(p[0]), int(p[1])] for p in line], key=lambda p: p[0])
        if len(pts) != 3:
            return int(pts[0][1]) if pts else 0
        x = float(xq)
        p0, p1, p2 = pts

        def _lerp_y(a: list[int], b: list[int], xx: float) -> float:
            ax, ay = float(a[0]), float(a[1])
            bx, by = float(b[0]), float(b[1])
            dx = bx - ax
            if abs(dx) < 1e-6:
                return 0.5 * (ay + by)
            t = max(0.0, min(1.0, (xx - ax) / dx))
            return ay + t * (by - ay)

        if x <= float(p1[0]):
            y = _lerp_y(p0, p1, x)
        else:
            y = _lerp_y(p1, p2, x)
        return int(round(y))

    def _vertical_x_guides(self, table: dict[str, Any]) -> list[int]:
        bbox = [int(v) for v in table["table_bbox"]]
        x0, _, x1, _ = bbox
        xs = [x0, x1]
        for ln in table.get("v_lines", []):
            try:
                xs.append(int(round(float(np.mean([int(p[0]) for p in ln])))))
            except Exception:
                continue
        xs = sorted(set(int(np.clip(v, x0, x1)) for v in xs))
        return xs

    def _split_separator_at_click(self, table: dict[str, Any], li: int, x_click: int) -> bool:
        seps = table.get("separators", [])
        if not (0 <= li < len(seps)):
            return False

        bbox = [int(v) for v in table["table_bbox"]]
        x0, _, x1, _ = bbox
        if x1 - x0 < 8:
            return False

        guides = self._vertical_x_guides(table)
        left = [gx for gx in guides if gx <= x_click]
        right = [gx for gx in guides if gx >= x_click]
        if not left or not right:
            return False
        left_x = int(max(left))
        right_x = int(min(right))
        if right_x - left_x < 6:
            return False

        src = seps[li]
        src_poly = _normalize_h_polyline([[int(p[0]), int(p[1])] for p in src["polyline"]], bbox)
        kind = str(src.get("kind", "line")).strip().lower()
        if kind not in ("line", "boundary"):
            kind = "line"

        new_items: list[dict[str, Any]] = []
        min_len = 10

        if left_x - x0 >= min_len:
            lx0 = int(x0)
            lx1 = int(left_x)
            lxm = int(round(0.5 * (lx0 + lx1)))
            left_poly = [
                [lx0, self._polyline_y_at_x(src_poly, lx0)],
                [lxm, self._polyline_y_at_x(src_poly, lxm)],
                [lx1, self._polyline_y_at_x(src_poly, lx1)],
            ]
            new_items.append({"polyline": _normalize_h_polyline(left_poly, bbox), "kind": kind})

        if x1 - right_x >= min_len:
            rx0 = int(right_x)
            rx1 = int(x1)
            rxm = int(round(0.5 * (rx0 + rx1)))
            right_poly = [
                [rx0, self._polyline_y_at_x(src_poly, rx0)],
                [rxm, self._polyline_y_at_x(src_poly, rxm)],
                [rx1, self._polyline_y_at_x(src_poly, rx1)],
            ]
            new_items.append({"polyline": _normalize_h_polyline(right_poly, bbox), "kind": kind})

        if not new_items:
            return False

        seps.pop(li)
        for off, item in enumerate(new_items):
            seps.insert(li + off, item)
        return True

    def _render(self) -> np.ndarray:
        vis = self.image.copy()
        for ti, t in enumerate(self.rec.get("tables", [])):
            x0, y0, x1, y1 = [int(v) for v in t["table_bbox"]]
            active = ti == self.state.table_idx
            tmp = vis.copy()
            fill_col = (0, 170, 0) if active else (0, 90, 0)
            cv2.rectangle(tmp, (x0, y0), (x1, y1), fill_col, -1)
            alpha = 0.12 if active else 0.07
            vis = cv2.addWeighted(vis, 1.0 - alpha, tmp, alpha, 0.0)
            border_col = (0, 230, 0) if active else (0, 130, 0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), border_col, 2 if active else 1)

            if active:
                for cx, cy in self._table_corners(ti):
                    cv2.circle(vis, (cx, cy), 8, (0, 255, 255), -1)
                    cv2.circle(vis, (cx, cy), 11, (0, 180, 255), 1)

            for ln in t.get("v_lines", []):
                pts = np.array(ln, dtype=np.int32).reshape(-1, 1, 2)
                col = (255, 170, 60) if active else (210, 120, 120)
                w = 2 if (active and self.state.mode == "v") else 1
                cv2.polylines(vis, [pts], False, col, w, cv2.LINE_AA)

            for item in t.get("separators", []):
                ln = item["polyline"]
                kind = str(item.get("kind", "line"))
                if kind == "line":
                    col = (80, 80, 255) if active else (120, 120, 180)
                    pcol = (180, 180, 255) if active else (160, 160, 200)
                else:
                    col = (0, 190, 255) if active else (80, 150, 180)
                    pcol = (80, 230, 255) if active else (140, 190, 210)
                pts = np.array(ln, dtype=np.int32).reshape(-1, 1, 2)
                w = 3 if (active and self.state.mode == "sep") else 2
                cv2.polylines(vis, [pts], False, col, w, cv2.LINE_AA)
                if active:
                    for p in ln:
                        cv2.circle(vis, (int(p[0]), int(p[1])), 4, pcol, -1)

        add_tag = " add=ON" if self.state.add_mode else ""
        del_tag = " del=ON" if self.state.delete_mode else ""
        tog_tag = " toggle=ON" if self.state.toggle_kind_mode else ""
        spl_tag = " split=ON" if self.state.split_mode else ""
        mov_tag = " move=ON" if self.state.move_table_mode else ""
        stem = self.current_json.stem if self.current_json else ""
        t_act = self._active_table()
        t1 = f"{self.state.idx+1}/{len(self.state.records)} {stem}"
        t2 = (
            f"grid={self.state.table_idx+1}/{self._table_count()} "
            f"mode={self.state.mode}{add_tag}{del_tag}{tog_tag}{spl_tag}{mov_tag} "
            f"v={len(t_act['v_lines'])} "
            f"line={sum(1 for s in t_act['separators'] if s.get('kind')=='line')} "
            f"boundary={sum(1 for s in t_act['separators'] if s.get('kind')=='boundary')}"
        )
        t3 = "keys: 1 table 2 separators 3 verticals | [ ] switch grid | g add-grid | r remove-grid | m move-grid(table) | drag points/lines/corners | a add | x delete | t toggle-kind(sep) | b split-at-cell(sep) | s save | n/p next/prev | q quit"

        pad = int(self.display_pad)
        h, w = vis.shape[:2]
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3), (42, 42, 42), dtype=np.uint8)
        canvas[pad : pad + h, pad : pad + w] = vis
        cv2.rectangle(canvas, (pad - 1, pad - 1), (pad + w, pad + h), (130, 130, 130), 1)
        cv2.putText(canvas, t1, (pad + 12, pad + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, t2, (pad + 12, pad + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, t3, (pad + 12, pad + h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.image is None or self.rec is None:
            return
        pad = int(self.display_pad)
        h, w = self.image.shape[:2]
        x = int(np.clip(x - pad, 0, w - 1))
        y = int(np.clip(y - pad, 0, h - 1))
        self._sanitize()

        if event == cv2.EVENT_LBUTTONDOWN:
            self.state.last_mouse = (x, y)

            # Click-to-activate table in any mode.
            hit_t = self._table_at_point(x, y)
            if hit_t >= 0:
                self.state.table_idx = hit_t

            if self.state.mode == "table":
                if self.state.move_table_mode and hit_t >= 0:
                    self.state.dragging = True
                    self.state.drag_kind = "table_whole"
                    self.state.drag_table_idx = hit_t
                    return
                ti, cidx = self._nearest_corner(x, y)
                if ti >= 0 and cidx >= 0:
                    self.state.table_idx = ti
                    self.state.dragging = True
                    self.state.drag_kind = "corner"
                    self.state.drag_table_idx = ti
                    self.state.drag_idx = cidx
                return

            table = self._active_table()
            x0, y0, x1, y1 = [int(v) for v in table["table_bbox"]]
            if self.state.mode == "sep":
                sep_lines = [item["polyline"] for item in table["separators"]]
                if self.state.delete_mode:
                    li = self._nearest_line_idx(sep_lines, x, y)
                    if li >= 0:
                        table["separators"].pop(li)
                    self.state.delete_mode = False
                    return

                if self.state.toggle_kind_mode:
                    li = self._nearest_line_idx(sep_lines, x, y)
                    if li >= 0:
                        kind = str(table["separators"][li].get("kind", "line"))
                        table["separators"][li]["kind"] = "boundary" if kind == "line" else "line"
                    self.state.toggle_kind_mode = False
                    return

                if self.state.split_mode:
                    li = self._nearest_line_idx(sep_lines, x, y)
                    if li >= 0:
                        _ = self._split_separator_at_click(table, li, x)
                        self._sanitize()
                    self.state.split_mode = False
                    return

                if self.state.add_mode:
                    ny = int(np.clip(y, y0, y1))
                    table["separators"].append({"polyline": _build_h_polyline(ny, x0, x1), "kind": "boundary"})
                    self.state.add_mode = False
                    self._sanitize()
                    return

                li, pi = self._nearest_line_point(sep_lines, x, y)
                if li >= 0 and pi >= 0:
                    self.state.dragging = True
                    self.state.drag_kind = "point"
                    self.state.drag_table_idx = self.state.table_idx
                    self.state.drag_idx = li
                    self.state.drag_point_idx = pi
                    return

                li = self._nearest_line_idx(sep_lines, x, y)
                if li >= 0:
                    self.state.dragging = True
                    self.state.drag_kind = "line"
                    self.state.drag_table_idx = self.state.table_idx
                    self.state.drag_idx = li
                    self.state.drag_point_idx = -1
                    return

            if self.state.mode == "v":
                v_lines = table["v_lines"]
                if self.state.delete_mode:
                    li = self._nearest_line_idx(v_lines, x, y)
                    if li >= 0:
                        v_lines.pop(li)
                    self.state.delete_mode = False
                    return

                if self.state.add_mode:
                    nx = int(np.clip(x, x0, x1))
                    ym = int(round(0.5 * (y0 + y1)))
                    v_lines.append([[nx, y0], [nx, ym], [nx, y1]])
                    self.state.add_mode = False
                    self._sanitize()
                    return

                li, pi = self._nearest_line_point(v_lines, x, y)
                if li >= 0 and pi >= 0:
                    self.state.dragging = True
                    self.state.drag_kind = "point"
                    self.state.drag_table_idx = self.state.table_idx
                    self.state.drag_idx = li
                    self.state.drag_point_idx = pi
                    return

                li = self._nearest_line_idx(v_lines, x, y)
                if li >= 0:
                    self.state.dragging = True
                    self.state.drag_kind = "line"
                    self.state.drag_table_idx = self.state.table_idx
                    self.state.drag_idx = li
                    self.state.drag_point_idx = -1
                    return

        elif event == cv2.EVENT_MOUSEMOVE and self.state.dragging:
            if self.state.drag_kind == "table_whole":
                ti = int(np.clip(self.state.drag_table_idx, 0, self._table_count() - 1))
                tabs = self.rec["tables"][ti]
                px, py = self.state.last_mouse
                dx = int(x - px)
                dy = int(y - py)
                self.state.last_mouse = (x, y)

                bx0, by0, bx1, by1 = [int(v) for v in tabs["table_bbox"]]
                dx = int(np.clip(dx, -bx0, (w - 1) - bx1))
                dy = int(np.clip(dy, -by0, (h - 1) - by1))
                if dx == 0 and dy == 0:
                    return

                tabs["table_bbox"] = [bx0 + dx, by0 + dy, bx1 + dx, by1 + dy]
                for ln in tabs.get("v_lines", []):
                    for p in ln:
                        p[0] = int(p[0] + dx)
                        p[1] = int(p[1] + dy)
                for it in tabs.get("separators", []):
                    for p in it.get("polyline", []):
                        p[0] = int(p[0] + dx)
                        p[1] = int(p[1] + dy)
                self._sanitize()
                return

            if self.state.drag_kind == "corner" and self.state.mode == "table":
                cx = int(np.clip(x, 0, w - 1))
                cy = int(np.clip(y, 0, h - 1))
                ti = int(np.clip(self.state.drag_table_idx, 0, self._table_count() - 1))
                tx0, ty0, tx1, ty1 = [int(v) for v in self.rec["tables"][ti]["table_bbox"]]
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
                self.rec["tables"][ti]["table_bbox"] = [tx0, ty0, tx1, ty1]
                self._sanitize()
                return

            if self.state.mode not in ("sep", "v"):
                return
            if self.state.mode == "sep":
                ti = int(np.clip(self.state.drag_table_idx, 0, self._table_count() - 1))
                tabs = self.rec["tables"][ti]
                if not (0 <= self.state.drag_idx < len(tabs["separators"])):
                    return
                poly = [[int(p[0]), int(p[1])] for p in tabs["separators"][self.state.drag_idx]["polyline"]]
            else:
                ti = int(np.clip(self.state.drag_table_idx, 0, self._table_count() - 1))
                tabs = self.rec["tables"][ti]
                if not (0 <= self.state.drag_idx < len(tabs["v_lines"])):
                    return
                poly = [[int(p[0]), int(p[1])] for p in tabs["v_lines"][self.state.drag_idx]]
            px, py = self.state.last_mouse
            dx = int(x - px)
            dy = int(y - py)
            self.state.last_mouse = (x, y)

            if self.state.drag_kind == "point" and 0 <= self.state.drag_point_idx < 3:
                pi = self.state.drag_point_idx
                poly[pi][0] += dx
                poly[pi][1] += dy
            elif self.state.drag_kind == "line":
                for p in poly:
                    p[0] += dx
                    p[1] += dy

            bbox = [int(v) for v in tabs["table_bbox"]]
            if self.state.mode == "sep":
                tabs["separators"][self.state.drag_idx]["polyline"] = _normalize_h_polyline(poly, bbox)
            else:
                tabs["v_lines"][self.state.drag_idx] = _normalize_v_polyline(poly, bbox)
            self._sanitize()

        elif event == cv2.EVENT_LBUTTONUP:
            self.state.dragging = False
            self.state.drag_kind = ""
            self.state.drag_table_idx = -1
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
                self.state.toggle_kind_mode = False
                self.state.split_mode = False
                continue
            if k == ord("2"):
                self.state.mode = "sep"
                self.state.add_mode = False
                self.state.delete_mode = False
                self.state.toggle_kind_mode = False
                self.state.split_mode = False
                self.state.move_table_mode = False
                continue
            if k == ord("3"):
                self.state.mode = "v"
                self.state.add_mode = False
                self.state.delete_mode = False
                self.state.toggle_kind_mode = False
                self.state.split_mode = False
                self.state.move_table_mode = False
                continue
            if k == ord("["):
                self.state.table_idx = max(0, self.state.table_idx - 1)
                self.state.add_mode = False
                self.state.delete_mode = False
                self.state.toggle_kind_mode = False
                self.state.split_mode = False
                continue
            if k == ord("]"):
                self.state.table_idx = min(self._table_count() - 1, self.state.table_idx + 1)
                self.state.add_mode = False
                self.state.delete_mode = False
                self.state.toggle_kind_mode = False
                self.state.split_mode = False
                continue
            if k == ord("g"):
                t = self._active_table()
                clone = json.loads(json.dumps(t))
                if self.image is not None:
                    h, w = self.image.shape[:2]
                    dx, dy = 24, 24
                    bx0, by0, bx1, by1 = [int(v) for v in clone["table_bbox"]]
                    bw, bh = bx1 - bx0, by1 - by0
                    nbx0 = int(np.clip(bx0 + dx, 0, max(0, w - 1)))
                    nby0 = int(np.clip(by0 + dy, 0, max(0, h - 1)))
                    nbx1 = int(np.clip(nbx0 + bw, 0, max(0, w - 1)))
                    nby1 = int(np.clip(nby0 + bh, 0, max(0, h - 1)))
                    clone["table_bbox"] = [nbx0, nby0, nbx1, nby1]
                    sx = nbx0 - bx0
                    sy = nby0 - by0
                    for ln in clone.get("v_lines", []):
                        for p in ln:
                            p[0] = int(p[0] + sx)
                            p[1] = int(p[1] + sy)
                    for it in clone.get("separators", []):
                        for p in it.get("polyline", []):
                            p[0] = int(p[0] + sx)
                            p[1] = int(p[1] + sy)
                self.rec["tables"].append(clone)
                self.state.table_idx = len(self.rec["tables"]) - 1
                self._sanitize()
                continue
            if k == ord("r"):
                if self._table_count() > 1:
                    self.rec["tables"].pop(self.state.table_idx)
                    self.state.table_idx = max(0, self.state.table_idx - 1)
                    self._sanitize()
                continue
            if k == ord("a") and self.state.mode in ("sep", "v"):
                self.state.add_mode = not self.state.add_mode
                if self.state.add_mode:
                    self.state.delete_mode = False
                    self.state.toggle_kind_mode = False
                    self.state.split_mode = False
                    self.state.move_table_mode = False
                continue
            if k == ord("x") and self.state.mode in ("sep", "v"):
                self.state.delete_mode = not self.state.delete_mode
                if self.state.delete_mode:
                    self.state.add_mode = False
                    self.state.toggle_kind_mode = False
                    self.state.split_mode = False
                    self.state.move_table_mode = False
                continue
            if k == ord("t") and self.state.mode == "sep":
                self.state.toggle_kind_mode = not self.state.toggle_kind_mode
                if self.state.toggle_kind_mode:
                    self.state.add_mode = False
                    self.state.delete_mode = False
                    self.state.split_mode = False
                    self.state.move_table_mode = False
                continue
            if k == ord("b") and self.state.mode == "sep":
                self.state.split_mode = not self.state.split_mode
                if self.state.split_mode:
                    self.state.add_mode = False
                    self.state.delete_mode = False
                    self.state.toggle_kind_mode = False
                    self.state.move_table_mode = False
                continue
            if k == ord("m") and self.state.mode == "table":
                self.state.move_table_mode = not self.state.move_table_mode
                if self.state.move_table_mode:
                    self.state.add_mode = False
                    self.state.delete_mode = False
                    self.state.toggle_kind_mode = False
                    self.state.split_mode = False
                continue
            if k == ord("n"):
                self._save_current()
                self.state.idx = min(len(self.state.records) - 1, self.state.idx + 1)
                self.state.table_idx = 0
                self._load_current()
                continue
            if k == ord("p"):
                self._save_current()
                self.state.idx = max(0, self.state.idx - 1)
                self.state.table_idx = 0
                self._load_current()
                continue

        cv2.destroyWindow(self.window)


def render_rowsep_masks(
    rowsep_labels_dir: Path,
    out_labels_dir: Path,
    line_thickness: int = 3,
    overwrite: bool = True,
) -> None:
    _ensure_dir(out_labels_dir)
    img_dir = out_labels_dir / "images"
    msk_dir = out_labels_dir / "masks"
    _ensure_dir(img_dir)
    _ensure_dir(msk_dir)

    recs = _collect_rowsep_jsons(rowsep_labels_dir)
    if not recs:
        print(f"No row-separator labels in {rowsep_labels_dir}")
        return

    for js in recs:
        stem = js.stem
        out_json = out_labels_dir / f"{stem}.json"
        if out_json.exists() and not overwrite:
            print(f"[skip] {stem}")
            continue

        rec = _load_rowsep_record(js)
        img = cv2.imread(str(rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] failed image for {stem}")
            continue
        h, w = img.shape[:2]
        tables = rec.get("tables", [])
        if not isinstance(tables, list) or not tables:
            print(f"[warn] no tables for {stem}")
            continue

        table = np.zeros((h, w), dtype=np.uint8)
        vmask = np.zeros((h, w), dtype=np.uint8)
        hline = np.zeros((h, w), dtype=np.uint8)
        hbound = np.zeros((h, w), dtype=np.uint8)

        for t in tables:
            if not isinstance(t, dict):
                continue
            bbox = _clip_bbox([int(v) for v in t.get("table_bbox", [0, 0, w - 1, h - 1])], w, h)
            x0, y0, x1, y1 = bbox
            cv2.rectangle(table, (x0, y0), (x1, y1), 255, -1)

            for ln in t.get("v_lines", []):
                nln = _normalize_v_polyline([[int(p[0]), int(p[1])] for p in ln], bbox)
                pts = np.array(nln, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vmask, [pts], False, 255, int(max(1, line_thickness)), cv2.LINE_AA)

            for item in t.get("separators", []):
                nln = _normalize_h_polyline([[int(p[0]), int(p[1])] for p in item.get("polyline", [])], bbox)
                pts = np.array(nln, dtype=np.int32).reshape(-1, 1, 2)
                kind = str(item.get("kind", "line")).strip().lower()
                if kind == "boundary":
                    cv2.polylines(hbound, [pts], False, 255, int(max(1, line_thickness)), cv2.LINE_AA)
                else:
                    cv2.polylines(hline, [pts], False, 255, int(max(1, line_thickness)), cv2.LINE_AA)

        hunion = cv2.bitwise_or(hline, hbound)

        p_img = img_dir / f"{stem}.png"
        p_t = msk_dir / f"{stem}_table.png"
        p_v = msk_dir / f"{stem}_v.png"
        p_h = msk_dir / f"{stem}_h.png"
        p_hline = msk_dir / f"{stem}_h_line.png"
        p_hbound = msk_dir / f"{stem}_h_boundary.png"

        cv2.imwrite(str(p_img), img)
        cv2.imwrite(str(p_t), table)
        cv2.imwrite(str(p_v), vmask)
        cv2.imwrite(str(p_h), hunion)
        cv2.imwrite(str(p_hline), hline)
        cv2.imwrite(str(p_hbound), hbound)

        # Keep exported records portable for GitHub: all paths are relative to
        # out_labels_dir, not absolute local machine paths.
        out_rec = {
            "image": str(p_img.relative_to(out_labels_dir)),
            "table_mask": str(p_t.relative_to(out_labels_dir)),
            "v_mask": str(p_v.relative_to(out_labels_dir)),
            "h_mask": str(p_h.relative_to(out_labels_dir)),
            "h_line_mask": str(p_hline.relative_to(out_labels_dir)),
            "h_boundary_mask": str(p_hbound.relative_to(out_labels_dir)),
            "source_rowsep_json": str(js.name),
        }
        _save_json(out_json, out_rec)
        print(f"[ok] {stem}")

    lines = []
    for js in sorted(out_labels_dir.glob("*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        req = ("image", "table_mask", "v_mask", "h_mask")
        if all(k in data for k in req):
            lines.append(json.dumps(data))
    (out_labels_dir / "index.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[ok] wrote {(out_labels_dir / 'index.jsonl')}")


def write_rowsep_overlays(rowsep_labels_dir: Path, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    recs = _collect_rowsep_jsons(rowsep_labels_dir)
    for js in recs:
        rec = _load_rowsep_record(js)
        img = cv2.imread(str(rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        vis = img.copy()
        h, w = img.shape[:2]
        tables = rec.get("tables", [])
        for t in tables:
            if not isinstance(t, dict):
                continue
            bbox = _clip_bbox([int(v) for v in t.get("table_bbox", [0, 0, w - 1, h - 1])], w, h)
            x0, y0, x1, y1 = bbox

            tmp = vis.copy()
            cv2.rectangle(tmp, (x0, y0), (x1, y1), (0, 160, 0), -1)
            vis = cv2.addWeighted(vis, 0.89, tmp, 0.11, 0.0)
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 0), 2)

            for ln in t.get("v_lines", []):
                nln = _normalize_v_polyline([[int(p[0]), int(p[1])] for p in ln], bbox)
                pts = np.array(nln, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, (255, 120, 120), 1, cv2.LINE_AA)

            for item in t.get("separators", []):
                nln = _normalize_h_polyline([[int(p[0]), int(p[1])] for p in item.get("polyline", [])], bbox)
                kind = str(item.get("kind", "line")).strip().lower()
                col = (80, 80, 255) if kind == "line" else (0, 190, 255)
                pts = np.array(nln, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, col, 2, cv2.LINE_AA)

        cv2.putText(vis, js.stem, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imwrite(str(out_dir / f"{js.stem}_overlay.png"), vis)
    print(f"[ok] wrote overlays to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Row-separator labeling tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="Create row-separator labels from flex labels")
    b.add_argument("--flex_labels_dir", default="ml_labels_flex")
    b.add_argument("--rowsep_labels_dir", default="ml_rowsep_labels")
    b.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("edit", help="Edit row-separator labels")
    e.add_argument("--rowsep_labels_dir", default="ml_rowsep_labels")
    e.add_argument("--start", type=int, default=0)

    r = sub.add_parser("render", help="Render row-separator labels to masks")
    r.add_argument("--rowsep_labels_dir", default="ml_rowsep_labels")
    r.add_argument("--out_labels_dir", default="ml_rowsep_labels_exported")
    r.add_argument("--line_thickness", type=int, default=3)
    r.add_argument("--no_overwrite", action="store_true")

    o = sub.add_parser("overlay", help="Write overlay previews")
    o.add_argument("--rowsep_labels_dir", default="ml_rowsep_labels")
    o.add_argument("--out_dir", default=None)

    args = p.parse_args()

    if args.cmd == "bootstrap":
        bootstrap_from_flex(
            flex_labels_dir=Path(args.flex_labels_dir),
            rowsep_labels_dir=Path(args.rowsep_labels_dir),
            overwrite=bool(args.overwrite),
        )
        return

    if args.cmd == "edit":
        ed = RowSepLabelEditor(Path(args.rowsep_labels_dir), start_index=int(args.start))
        ed.run()
        return

    if args.cmd == "render":
        render_rowsep_masks(
            rowsep_labels_dir=Path(args.rowsep_labels_dir),
            out_labels_dir=Path(args.out_labels_dir),
            line_thickness=int(args.line_thickness),
            overwrite=not bool(args.no_overwrite),
        )
        return

    if args.cmd == "overlay":
        out_dir = Path(args.out_dir) if args.out_dir else (Path(args.rowsep_labels_dir) / "overlays")
        write_rowsep_overlays(Path(args.rowsep_labels_dir), out_dir)
        return


if __name__ == "__main__":
    main()
