"""Intersection-first scorecard annotation tool.

You label line intersections only. The tool:
- auto-connects neighboring intersections into grid segments
- auto-derives base boxes from the intersection lattice
- lets you delete a specific segment between two points (for merged cells)

Main commands:
- bootstrap: make fresh label records from images
- edit: interactive intersection/segment editor
- overlay: visualize labels
- export: export masks + matrix-preserving cell crops + metadata
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import argparse
import json

import cv2
import numpy as np

from scorecard_preprocessing import PreprocessConfig, preprocess_scorecard


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_images(images_dir: Path) -> list[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    out: list[Path] = []
    for e in exts:
        out.extend(images_dir.glob(e))
    return sorted(set(out))


def _record_paths(labels_dir: Path, stem: str) -> dict[str, Path]:
    img_dir = labels_dir / "images"
    _ensure_dir(img_dir)
    return {
        "json": labels_dir / f"{stem}.json",
        "image": img_dir / f"{stem}.png",
    }


def _edge_key(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _pt_clip(x: int, y: int, w: int, h: int) -> list[int]:
    return [int(np.clip(x, 0, max(0, w - 1))), int(np.clip(y, 0, max(0, h - 1)))]


def _order_quad(quad: list[list[int]]) -> list[list[int]]:
    q = np.array(quad, dtype=np.float32)
    if q.shape != (4, 2):
        raise ValueError("table_quad must be 4x2")
    s = q[:, 0] + q[:, 1]
    d = q[:, 0] - q[:, 1]
    tl = int(np.argmin(s))
    br = int(np.argmax(s))
    tr = int(np.argmax(d))
    bl = int(np.argmin(d))
    idx = [tl, tr, br, bl]
    if len(set(idx)) != 4:
        order = np.argsort(q[:, 1] * 100000.0 + q[:, 0])
        top = sorted(order[:2], key=lambda i: q[i, 0])
        bot = sorted(order[2:], key=lambda i: q[i, 0])
        idx = [int(top[0]), int(top[1]), int(bot[1]), int(bot[0])]
    out = q[idx].astype(np.int32)
    return [[int(v) for v in p] for p in out.tolist()]


def _table_quad_from_points(points: list[dict[str, Any]], w: int, h: int) -> list[list[int]]:
    if len(points) < 4:
        return [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]
    xy = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.float32)
    s = xy[:, 0] + xy[:, 1]
    d = xy[:, 0] - xy[:, 1]
    tl = xy[int(np.argmin(s))]
    br = xy[int(np.argmax(s))]
    tr = xy[int(np.argmax(d))]
    bl = xy[int(np.argmin(d))]
    quad = [_pt_clip(int(tl[0]), int(tl[1]), w, h), _pt_clip(int(tr[0]), int(tr[1]), w, h), _pt_clip(int(br[0]), int(br[1]), w, h), _pt_clip(int(bl[0]), int(bl[1]), w, h)]
    return _order_quad(quad)


def _is_record(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    req = ("image", "table_quad", "points", "blocked_edges")
    return all(k in data for k in req)


def _load_record(js: Path) -> dict[str, Any]:
    data = json.loads(js.read_text(encoding="utf-8"))
    if not _is_record(data):
        raise RuntimeError(f"Invalid label record: {js}")
    return data


def _save_record(js: Path, rec: dict[str, Any]) -> None:
    js.write_text(json.dumps(rec, indent=2), encoding="utf-8")


def _collect_records(labels_dir: Path) -> list[Path]:
    out: list[Path] = []
    for js in sorted(labels_dir.glob("*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        if _is_record(data):
            out.append(js)
    return out


def _point_map(points: list[dict[str, Any]]) -> dict[int, tuple[int, int]]:
    out: dict[int, tuple[int, int]] = {}
    for p in points:
        out[int(p["id"])] = (int(p["x"]), int(p["y"]))
    return out


def _seg_dist_sq(a: tuple[int, int], b: tuple[int, int], p: tuple[int, int]) -> float:
    ax, ay = float(a[0]), float(a[1])
    bx, by = float(b[0]), float(b[1])
    px, py = float(p[0]), float(p[1])
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


def _cluster_1d(values: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    n = int(values.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    order = np.argsort(values)
    groups: list[list[int]] = []
    for idx in order.tolist():
        v = float(values[idx])
        if not groups:
            groups.append([idx])
            continue
        g = groups[-1]
        gm = float(np.mean(values[g]))
        if abs(v - gm) <= float(eps):
            g.append(idx)
        else:
            groups.append([idx])
    centers = np.array([float(np.mean(values[g])) for g in groups], dtype=np.float32)
    assign = np.zeros((n,), dtype=np.int32)
    for gi, g in enumerate(groups):
        for idx in g:
            assign[idx] = int(gi)
    return centers, assign


def _project_points(points_xy: np.ndarray, table_quad: np.ndarray) -> tuple[np.ndarray, float, float]:
    tl, tr, br, bl = table_quad.astype(np.float32)
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    W = max(300.0, 0.5 * (w_top + w_bot))
    H = max(220.0, 0.5 * (h_left + h_right))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(table_quad.astype(np.float32), dst)
    pts = cv2.perspectiveTransform(points_xy.reshape(-1, 1, 2).astype(np.float32), M).reshape(-1, 2)
    return pts, W, H


def _build_grid_from_points(points: list[dict[str, Any]], table_quad: list[list[int]]) -> dict[str, Any]:
    pmap = _point_map(points)
    if len(pmap) < 2:
        return {
            "rows": 0,
            "cols": 0,
            "point_rc": {},
            "rc_point": {},
            "default_edges": set(),
            "row_centers": [],
            "col_centers": [],
        }
    pids = sorted(pmap.keys())
    xy = np.array([[pmap[pid][0], pmap[pid][1]] for pid in pids], dtype=np.float32)
    tq = np.array(table_quad, dtype=np.float32)
    uv, W, H = _project_points(xy, tq)

    eps_x = max(8.0, 0.020 * W)
    eps_y = max(8.0, 0.020 * H)
    cx, ax = _cluster_1d(uv[:, 0], eps=eps_x)
    cy, ay = _cluster_1d(uv[:, 1], eps=eps_y)

    col_order = np.argsort(cx)
    row_order = np.argsort(cy)
    col_rank = {int(old): int(new) for new, old in enumerate(col_order.tolist())}
    row_rank = {int(old): int(new) for new, old in enumerate(row_order.tolist())}
    col_centers = [float(cx[i]) for i in col_order.tolist()]
    row_centers = [float(cy[i]) for i in row_order.tolist()]

    point_rc: dict[int, tuple[int, int]] = {}
    rc_point: dict[tuple[int, int], int] = {}
    rc_dist: dict[tuple[int, int], float] = {}

    for i, pid in enumerate(pids):
        r = int(row_rank[int(ay[i])])
        c = int(col_rank[int(ax[i])])
        key = (r, c)
        # Deduplicate collisions by rectified center distance.
        d = abs(float(uv[i, 0]) - col_centers[c]) + abs(float(uv[i, 1]) - row_centers[r])
        if key in rc_point and d >= rc_dist[key]:
            continue
        if key in rc_point:
            old_pid = int(rc_point[key])
            point_rc.pop(old_pid, None)
        rc_point[key] = int(pid)
        rc_dist[key] = float(d)
        point_rc[int(pid)] = (r, c)

    rows = len(row_centers)
    cols = len(col_centers)
    default_edges: set[tuple[int, int]] = set()
    for r in range(rows):
        for c in range(cols - 1):
            a = rc_point.get((r, c))
            b = rc_point.get((r, c + 1))
            if a is not None and b is not None:
                default_edges.add(_edge_key(a, b))
    for c in range(cols):
        for r in range(rows - 1):
            a = rc_point.get((r, c))
            b = rc_point.get((r + 1, c))
            if a is not None and b is not None:
                default_edges.add(_edge_key(a, b))

    return {
        "rows": int(rows),
        "cols": int(cols),
        "point_rc": point_rc,
        "rc_point": rc_point,
        "default_edges": default_edges,
        "row_centers": row_centers,
        "col_centers": col_centers,
    }


def _blocked_set_from_record(rec: dict[str, Any]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for e in rec.get("blocked_edges", []):
        try:
            a = int(e["a"])
            b = int(e["b"])
        except Exception:
            continue
        out.add(_edge_key(a, b))
    return out


def _blocked_list(edges: set[tuple[int, int]]) -> list[dict[str, int]]:
    return [{"a": int(a), "b": int(b)} for a, b in sorted(edges)]


def _manual_set_from_record(rec: dict[str, Any]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for e in rec.get("manual_edges", []):
        try:
            a = int(e["a"])
            b = int(e["b"])
        except Exception:
            continue
        out.add(_edge_key(a, b))
    return out


def _manual_list(edges: set[tuple[int, int]]) -> list[dict[str, int]]:
    return [{"a": int(a), "b": int(b)} for a, b in sorted(edges)]


def _full_line_entries(rec: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in rec.get("added_full_lines", []):
        if not isinstance(it, dict):
            continue
        try:
            lid = int(it.get("id", -1))
            axis = str(it.get("axis", ""))
            pids = [int(v) for v in it.get("point_ids", [])]
            new_pids = [int(v) for v in it.get("new_point_ids", [])]
        except Exception:
            continue
        if lid < 0 or axis not in ("v", "h"):
            continue
        out.append(
            {
                "id": int(lid),
                "axis": axis,
                "point_ids": pids,
                "new_point_ids": new_pids,
            }
        )
    return out


def _sanitize_record(rec: dict[str, Any], image: np.ndarray) -> dict[str, Any]:
    h, w = image.shape[:2]
    pts = rec.get("points", [])
    cleaned: list[dict[str, int]] = []
    used_ids: set[int] = set()
    for p in pts:
        if not isinstance(p, dict):
            continue
        if "id" not in p or "x" not in p or "y" not in p:
            continue
        pid = int(p["id"])
        if pid in used_ids:
            continue
        used_ids.add(pid)
        x, y = _pt_clip(int(p["x"]), int(p["y"]), w, h)
        cleaned.append({"id": pid, "x": x, "y": y})
    cleaned.sort(key=lambda d: int(d["id"]))
    rec["points"] = cleaned
    rec["next_point_id"] = max([int(p["id"]) for p in cleaned], default=-1) + 1
    # Table corners are inferred from intersection extremes.
    rec["table_quad"] = _table_quad_from_points(rec["points"], w=w, h=h)
    point_ids_set = {int(p["id"]) for p in rec["points"]}

    grid = _build_grid_from_points(rec["points"], rec["table_quad"])
    allowed = set(grid["default_edges"])
    blocked = _blocked_set_from_record(rec)
    blocked = {e for e in blocked if e in allowed}
    rec["blocked_edges"] = _blocked_list(blocked)

    # Manual edges let user add missing separators between arbitrary labeled
    # intersection points (for partial row/col separators, merged cells, etc).
    pmap = _point_map(rec["points"])
    manual = _manual_set_from_record(rec)
    cleaned_manual: set[tuple[int, int]] = set()
    for e in manual:
        a, b = int(e[0]), int(e[1])
        if a == b or a not in point_ids_set or b not in point_ids_set:
            continue
        if e in allowed:
            continue
        pa = pmap[a]
        pb = pmap[b]
        dx = abs(int(pa[0]) - int(pb[0]))
        dy = abs(int(pa[1]) - int(pb[1]))
        if max(dx, dy) < 4:
            continue
        # Keep near-horizontal or near-vertical manual segments only.
        # 2.5x corresponds to about 21.8 degrees max tilt from axis.
        if not (dx >= 2.5 * dy or dy >= 2.5 * dx):
            continue
        cleaned_manual.add(e)
    rec["manual_edges"] = _manual_list(cleaned_manual)

    # Keep only valid full-line records (used to undo mistaken full-line adds).
    full = _full_line_entries(rec)
    valid_full: list[dict[str, Any]] = []
    for it in full:
        pids = [pid for pid in it["point_ids"] if pid in point_ids_set]
        new_pids = [pid for pid in it["new_point_ids"] if pid in point_ids_set]
        if len(set(pids)) < 2:
            continue
        valid_full.append(
            {
                "id": int(it["id"]),
                "axis": str(it["axis"]),
                "point_ids": sorted(set(int(v) for v in pids)),
                "new_point_ids": sorted(set(int(v) for v in new_pids)),
            }
        )
    valid_full.sort(key=lambda d: int(d["id"]))
    rec["added_full_lines"] = valid_full
    rec["next_full_line_id"] = max([int(v["id"]) for v in valid_full], default=-1) + 1
    return grid


def _active_edges(rec: dict[str, Any], grid: dict[str, Any]) -> set[tuple[int, int]]:
    blocked = _blocked_set_from_record(rec)
    manual = _manual_set_from_record(rec)
    return (set(grid["default_edges"]) - blocked) | manual


def bootstrap_labels(images_dir: Path, labels_dir: Path, overwrite: bool = False) -> None:
    _ensure_dir(labels_dir)
    imgs = _collect_images(images_dir)
    if not imgs:
        print(f"No images found in {images_dir}")
        return

    cfg = PreprocessConfig()
    for src in imgs:
        stem = src.stem
        pp = _record_paths(labels_dir, stem)
        if pp["json"].exists() and not overwrite:
            print(f"[skip] {src.name}")
            continue
        raw = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if raw is None:
            print(f"[warn] failed image {src}")
            continue
        prep = preprocess_scorecard(raw, cfg)
        img = prep.image_bgr
        h, w = img.shape[:2]
        cv2.imwrite(str(pp["image"]), img)
        rec = {
            "version": "intersection_grid_v1",
            "source_image": str(src.resolve()),
            "image": str(pp["image"].resolve()),
            "upright_rotation_degrees": int(prep.upright_rotation_degrees),
            "table_quad": [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            "points": [],
            "blocked_edges": [],
            "manual_edges": [],
            "next_point_id": 0,
            "added_full_lines": [],
            "next_full_line_id": 0,
        }
        _save_record(pp["json"], rec)
        print(f"[ok] {src.name}")


@dataclass
class _EditorState:
    records: list[Path]
    idx: int = 0
    mode: str = "point"  # point | edge
    dragging: bool = False
    drag_kind: str = ""  # point
    drag_idx: int = -1
    add_mode: bool = False
    delete_mode: bool = False
    restore_mode: bool = False
    edge_add_start: int = -1
    add_full_v_mode: bool = False
    add_full_h_mode: bool = False
    remove_added_full_mode: bool = False
    restore_removed_full_mode: bool = False


class IntersectionEditor:
    def __init__(self, labels_dir: Path, start: int = 0):
        self.labels_dir = labels_dir
        recs = _collect_records(labels_dir)
        if not recs:
            raise RuntimeError(f"No label JSONs in {labels_dir}. Run bootstrap first.")
        self.state = _EditorState(records=recs, idx=max(0, min(start, len(recs) - 1)))
        self.window = "scorecard_intersection_label_tool"
        self.display_pad = 44
        self.corner_tol = 68
        self.point_tol = 18
        self.seg_tol = 28

        self.current_json: Optional[Path] = None
        self.rec: Optional[dict[str, Any]] = None
        self.image: Optional[np.ndarray] = None
        self.grid: dict[str, Any] = {}

    def _load_current(self) -> None:
        self.current_json = self.state.records[self.state.idx]
        self.rec = _load_record(self.current_json)
        self.image = cv2.imread(str(self.rec["image"]), cv2.IMREAD_COLOR)
        if self.image is None:
            raise RuntimeError(f"Failed image: {self.rec['image']}")
        self.grid = _sanitize_record(self.rec, self.image)
        self.state.dragging = False
        self.state.drag_kind = ""
        self.state.drag_idx = -1
        self.state.add_mode = False
        self.state.delete_mode = False
        self.state.restore_mode = False
        self.state.edge_add_start = -1
        self.state.add_full_v_mode = False
        self.state.add_full_h_mode = False
        self.state.remove_added_full_mode = False
        self.state.restore_removed_full_mode = False

    def _save_current(self) -> None:
        if self.current_json is None or self.rec is None or self.image is None:
            return
        self.grid = _sanitize_record(self.rec, self.image)
        _save_record(self.current_json, self.rec)

    def _point_by_id(self, pid: int) -> Optional[dict[str, Any]]:
        for p in self.rec["points"]:
            if int(p["id"]) == int(pid):
                return p
        return None

    def _nearest_point_id(self, x: int, y: int, tol: Optional[int] = None) -> int:
        tol = self.point_tol if tol is None else int(tol)
        best = -1
        best_d2 = 1e18
        for p in self.rec["points"]:
            dx = int(p["x"]) - x
            dy = int(p["y"]) - y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = int(p["id"])
        return best if best_d2 <= tol * tol else -1

    def _upsert_point(self, x: int, y: int, tol: Optional[int] = None) -> int:
        tol = self.point_tol if tol is None else int(tol)
        pid = self._nearest_point_id(x, y, tol=tol)
        if pid >= 0:
            return int(pid)
        pid = int(self.rec["next_point_id"])
        self.rec["next_point_id"] = pid + 1
        self.rec["points"].append({"id": pid, "x": int(x), "y": int(y)})
        self.grid = _sanitize_record(self.rec, self.image)
        return int(pid)

    def _add_point(self, x: int, y: int) -> None:
        _ = self._upsert_point(x, y, tol=self.point_tol)

    def _delete_point(self, pid: int) -> None:
        if pid < 0:
            return
        self.rec["points"] = [p for p in self.rec["points"] if int(p["id"]) != int(pid)]
        blocked = _blocked_set_from_record(self.rec)
        blocked = {e for e in blocked if pid not in e}
        self.rec["blocked_edges"] = _blocked_list(blocked)
        self.grid = _sanitize_record(self.rec, self.image)

    def _nearest_segment(self, x: int, y: int) -> Optional[tuple[int, int]]:
        pmap = _point_map(self.rec["points"])
        candidates = set(self.grid["default_edges"]) | _manual_set_from_record(self.rec)
        best_e = None
        best_d2 = 1e18
        for e in sorted(candidates):
            a, b = int(e[0]), int(e[1])
            if a not in pmap or b not in pmap:
                continue
            d2 = _seg_dist_sq(pmap[a], pmap[b], (x, y))
            if d2 < best_d2:
                best_d2 = d2
                best_e = (a, b)
        if best_e is None:
            return None
        if best_d2 > float(self.seg_tol * self.seg_tol):
            return None
        return _edge_key(best_e[0], best_e[1])

    def _remove_segment(self, e: tuple[int, int]) -> None:
        manual = _manual_set_from_record(self.rec)
        if e in manual:
            manual.discard(e)
            self.rec["manual_edges"] = _manual_list(manual)
            self.grid = _sanitize_record(self.rec, self.image)
            return
        blocked = _blocked_set_from_record(self.rec)
        if e in self.grid["default_edges"]:
            blocked.add(e)
        self.rec["blocked_edges"] = _blocked_list(blocked)
        self.grid = _sanitize_record(self.rec, self.image)

    def _restore_segment(self, e: tuple[int, int]) -> None:
        # For default segments: un-block. Manual segments are added via edge-add mode.
        blocked = _blocked_set_from_record(self.rec)
        blocked.discard(e)
        self.rec["blocked_edges"] = _blocked_list(blocked)
        self.grid = _sanitize_record(self.rec, self.image)

    def _add_segment_between_points(self, a: int, b: int) -> None:
        if a < 0 or b < 0 or a == b:
            return
        e = _edge_key(a, b)
        # If this is a default edge, adding means ensure it's active (not blocked).
        if e in self.grid["default_edges"]:
            blocked = _blocked_set_from_record(self.rec)
            blocked.discard(e)
            self.rec["blocked_edges"] = _blocked_list(blocked)
            self.grid = _sanitize_record(self.rec, self.image)
            return
        manual = _manual_set_from_record(self.rec)
        manual.add(e)
        self.rec["manual_edges"] = _manual_list(manual)
        self.grid = _sanitize_record(self.rec, self.image)

    def _add_full_line_from_click(self, axis: str, x: int, y: int) -> None:
        points = self.rec.get("points", [])
        if len(points) < 2:
            return
        before = {int(p["id"]) for p in self.rec.get("points", [])}
        touched: list[int] = []
        tq = np.array(self.rec["table_quad"], dtype=np.float32)
        tl, tr, br, bl = tq
        w_top = float(np.linalg.norm(tr - tl))
        w_bot = float(np.linalg.norm(br - bl))
        h_left = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        W = max(300.0, 0.5 * (w_top + w_bot))
        H = max(220.0, 0.5 * (h_left + h_right))
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(tq.astype(np.float32), dst)
        M_inv = cv2.getPerspectiveTransform(dst, tq.astype(np.float32))

        p_click = np.array([[[float(x), float(y)]]], dtype=np.float32)
        uv_click = cv2.perspectiveTransform(p_click, M).reshape(2)
        u_click = float(np.clip(uv_click[0], 0.0, W - 1.0))
        v_click = float(np.clip(uv_click[1], 0.0, H - 1.0))

        if axis == "v":
            levels = [float(v) for v in self.grid.get("row_centers", [])]
            if not levels:
                return
            for v in levels:
                uv = np.array([[[u_click, float(np.clip(v, 0.0, H - 1.0))]]], dtype=np.float32)
                xy = cv2.perspectiveTransform(uv, M_inv).reshape(2)
                pid = self._upsert_point(int(round(float(xy[0]))), int(round(float(xy[1]))), tol=max(12, self.point_tol))
                touched.append(int(pid))
        else:
            levels = [float(u) for u in self.grid.get("col_centers", [])]
            if not levels:
                return
            for u in levels:
                uv = np.array([[[float(np.clip(u, 0.0, W - 1.0)), v_click]]], dtype=np.float32)
                xy = cv2.perspectiveTransform(uv, M_inv).reshape(2)
                pid = self._upsert_point(int(round(float(xy[0]))), int(round(float(xy[1]))), tol=max(12, self.point_tol))
                touched.append(int(pid))

        self.grid = _sanitize_record(self.rec, self.image)
        touched = sorted(set(int(v) for v in touched))
        if len(touched) < 2:
            return
        created = [pid for pid in touched if pid not in before]
        lid = int(self.rec.get("next_full_line_id", 0))
        self.rec["next_full_line_id"] = lid + 1
        entries = _full_line_entries(self.rec)
        entries.append(
            {
                "id": int(lid),
                "axis": str(axis),
                "point_ids": touched,
                "new_point_ids": sorted(set(int(v) for v in created)),
            }
        )
        self.rec["added_full_lines"] = entries
        self.grid = _sanitize_record(self.rec, self.image)

    def _nearest_added_full_line_id(self, x: int, y: int) -> int:
        pmap = _point_map(self.rec["points"])
        best_id = -1
        best_d2 = 1e18
        for it in _full_line_entries(self.rec):
            pts = [pmap[pid] for pid in it["point_ids"] if pid in pmap]
            if len(pts) < 2:
                continue
            if str(it["axis"]) == "v":
                pts = sorted(pts, key=lambda p: (p[1], p[0]))
            else:
                pts = sorted(pts, key=lambda p: (p[0], p[1]))
            # Use piecewise segments (not only endpoints) to tolerate mild warp.
            d2 = 1e18
            for i in range(1, len(pts)):
                d2 = min(d2, _seg_dist_sq(pts[i - 1], pts[i], (x, y)))
            if d2 < best_d2:
                best_d2 = d2
                best_id = int(it["id"])
        tol = max(float(self.seg_tol + 16), 42.0)
        if best_id < 0 or best_d2 > float(tol * tol):
            return -1
        return best_id

    def _remove_added_full_line(self, line_id: int) -> None:
        entries = _full_line_entries(self.rec)
        target = None
        keep = []
        for it in entries:
            if int(it["id"]) == int(line_id):
                target = it
            else:
                keep.append(it)
        if target is None:
            return

        other_refs: set[int] = set()
        for it in keep:
            other_refs.update(int(v) for v in it["point_ids"])

        remove_pids = [int(pid) for pid in target["new_point_ids"] if int(pid) not in other_refs]
        if remove_pids:
            rm = set(remove_pids)
            self.rec["points"] = [p for p in self.rec["points"] if int(p["id"]) not in rm]

            blocked = _blocked_set_from_record(self.rec)
            blocked = {e for e in blocked if e[0] not in rm and e[1] not in rm}
            self.rec["blocked_edges"] = _blocked_list(blocked)

            manual = _manual_set_from_record(self.rec)
            manual = {e for e in manual if e[0] not in rm and e[1] not in rm}
            self.rec["manual_edges"] = _manual_list(manual)

        self.rec["added_full_lines"] = keep
        self.grid = _sanitize_record(self.rec, self.image)

    def _restore_removed_full_line_from_click(self, x: int, y: int) -> None:
        points = self.rec.get("points", [])
        if len(points) < 2:
            return

        tq = np.array(self.rec["table_quad"], dtype=np.float32)
        tl, tr, br, bl = tq
        w_top = float(np.linalg.norm(tr - tl))
        w_bot = float(np.linalg.norm(br - bl))
        h_left = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        W = max(300.0, 0.5 * (w_top + w_bot))
        H = max(220.0, 0.5 * (h_left + h_right))
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(tq.astype(np.float32), dst)

        p_click = np.array([[[float(x), float(y)]]], dtype=np.float32)
        uv_click = cv2.perspectiveTransform(p_click, M).reshape(2)
        u_click = float(np.clip(uv_click[0], 0.0, W - 1.0))
        v_click = float(np.clip(uv_click[1], 0.0, H - 1.0))

        blocked = _blocked_set_from_record(self.rec)
        if not blocked:
            return
        rc_point = self.grid.get("rc_point", {})
        rows = int(self.grid.get("rows", 0))
        cols = int(self.grid.get("cols", 0))
        col_centers = [float(v) for v in self.grid.get("col_centers", [])]
        row_centers = [float(v) for v in self.grid.get("row_centers", [])]
        if rows <= 0 or cols <= 0:
            return

        def _pitch(vals: list[float], fallback: float = 40.0) -> float:
            if len(vals) < 2:
                return float(fallback)
            d = np.diff(np.asarray(vals, dtype=np.float32))
            d = d[d > 1e-6]
            if d.size == 0:
                return float(fallback)
            return float(np.median(d))

        col_pitch = _pitch(col_centers, 40.0)
        row_pitch = _pitch(row_centers, 40.0)

        candidates: list[tuple[float, str, int]] = []  # (score, axis, index)

        for c in range(cols):
            segs = []
            blk = 0
            for r in range(max(0, rows - 1)):
                a = rc_point.get((r, c))
                b = rc_point.get((r + 1, c))
                if a is None or b is None:
                    continue
                e = _edge_key(int(a), int(b))
                if e not in self.grid["default_edges"]:
                    continue
                segs.append(e)
                if e in blocked:
                    blk += 1
            if blk <= 0 or not segs:
                continue
            dist = abs((col_centers[c] if c < len(col_centers) else u_click) - u_click)
            score = dist / max(1.0, col_pitch)
            candidates.append((float(score), "v", int(c)))

        for r in range(rows):
            segs = []
            blk = 0
            for c in range(max(0, cols - 1)):
                a = rc_point.get((r, c))
                b = rc_point.get((r, c + 1))
                if a is None or b is None:
                    continue
                e = _edge_key(int(a), int(b))
                if e not in self.grid["default_edges"]:
                    continue
                segs.append(e)
                if e in blocked:
                    blk += 1
            if blk <= 0 or not segs:
                continue
            dist = abs((row_centers[r] if r < len(row_centers) else v_click) - v_click)
            score = dist / max(1.0, row_pitch)
            candidates.append((float(score), "h", int(r)))

        if not candidates:
            return
        candidates.sort(key=lambda t: t[0])
        _, axis, idx = candidates[0]

        if axis == "v":
            c = int(idx)
            for r in range(max(0, rows - 1)):
                a = rc_point.get((r, c))
                b = rc_point.get((r + 1, c))
                if a is None or b is None:
                    continue
                e = _edge_key(int(a), int(b))
                if e in self.grid["default_edges"]:
                    blocked.discard(e)
        else:
            r = int(idx)
            for c in range(max(0, cols - 1)):
                a = rc_point.get((r, c))
                b = rc_point.get((r, c + 1))
                if a is None or b is None:
                    continue
                e = _edge_key(int(a), int(b))
                if e in self.grid["default_edges"]:
                    blocked.discard(e)

        self.rec["blocked_edges"] = _blocked_list(blocked)
        self.grid = _sanitize_record(self.rec, self.image)

    def _render(self) -> np.ndarray:
        vis = self.image.copy()
        h, w = vis.shape[:2]

        # table area
        tq = np.array(self.rec["table_quad"], dtype=np.int32).reshape(-1, 1, 2)
        fill = vis.copy()
        cv2.fillPoly(fill, [tq], (0, 140, 0))
        vis = cv2.addWeighted(vis, 0.88, fill, 0.12, 0.0)
        cv2.polylines(vis, [tq], True, (0, 220, 0), 2, cv2.LINE_AA)
        for p in self.rec["table_quad"]:
            cv2.circle(vis, (int(p[0]), int(p[1])), 8, (0, 255, 255), -1)
            cv2.circle(vis, (int(p[0]), int(p[1])), 11, (0, 180, 255), 1)

        blocked = _blocked_set_from_record(self.rec)
        manual = _manual_set_from_record(self.rec)
        full_lines = _full_line_entries(self.rec)
        pmap = _point_map(self.rec["points"])
        point_rc = self.grid.get("point_rc", {})

        for e in sorted(self.grid.get("default_edges", [])):
            a, b = int(e[0]), int(e[1])
            if a not in pmap or b not in pmap:
                continue
            pa = pmap[a]
            pb = pmap[b]
            ra, ca = point_rc.get(a, (-1, -1))
            rb, cb = point_rc.get(b, (-1, -1))
            is_h = (ra == rb and ca != cb)
            if e in blocked:
                color = (0, 255, 0)
                thick = 3
            else:
                color = (0, 0, 255) if is_h else (255, 0, 0)
                thick = 2
            cv2.line(vis, pa, pb, color, thick, cv2.LINE_AA)
            if e in blocked:
                mx = int(round(0.5 * (pa[0] + pb[0])))
                my = int(round(0.5 * (pa[1] + pb[1])))
                cv2.line(vis, (mx - 4, my - 4), (mx + 4, my + 4), (0, 0, 0), 1, cv2.LINE_AA)
                cv2.line(vis, (mx - 4, my + 4), (mx + 4, my - 4), (0, 0, 0), 1, cv2.LINE_AA)
        for e in sorted(manual):
            a, b = int(e[0]), int(e[1])
            if a not in pmap or b not in pmap:
                continue
            cv2.line(vis, pmap[a], pmap[b], (0, 255, 255), 2, cv2.LINE_AA)
        for it in full_lines:
            pts = [pmap[pid] for pid in it["point_ids"] if pid in pmap]
            if len(pts) < 2:
                continue
            if str(it["axis"]) == "v":
                pts = sorted(pts, key=lambda p: (p[1], p[0]))
                color = (255, 0, 0)
            else:
                pts = sorted(pts, key=lambda p: (p[0], p[1]))
                color = (0, 0, 255)
            cv2.line(vis, pts[0], pts[-1], color, 2, cv2.LINE_AA)

        for p in self.rec["points"]:
            pid = int(p["id"])
            px, py = int(p["x"]), int(p["y"])
            cv2.circle(vis, (px, py), 4, (30, 220, 255), -1)
            cv2.circle(vis, (px, py), 7, (10, 120, 180), 1)
            cv2.putText(vis, str(pid), (px + 4, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        blocked_ct = len(blocked)
        seg_ct = len(self.grid.get("default_edges", []))
        mode_tags = []
        if self.state.add_mode:
            mode_tags.append("add=ON")
        if self.state.delete_mode:
            mode_tags.append("delete=ON")
        if self.state.restore_mode:
            mode_tags.append("restore=ON")
        if self.state.mode == "edge" and self.state.add_mode:
            mode_tags.append("add-seg=ON")
        if self.state.mode == "edge" and self.state.edge_add_start >= 0:
            mode_tags.append(f"from={self.state.edge_add_start}")
        if self.state.mode == "point" and self.state.add_full_v_mode:
            mode_tags.append("full-v=ON")
        if self.state.mode == "point" and self.state.add_full_h_mode:
            mode_tags.append("full-h=ON")
        if self.state.mode == "point" and self.state.remove_added_full_mode:
            mode_tags.append("rm-added-full=ON")
        if self.state.mode == "point" and self.state.restore_removed_full_mode:
            mode_tags.append("restore-removed-full=ON")
        tag_str = (" " + " ".join(mode_tags)) if mode_tags else ""

        stem = self.current_json.stem if self.current_json else ""
        t1 = f"{self.state.idx+1}/{len(self.state.records)} {stem}"
        t2 = (
            f"mode={self.state.mode}{tag_str} points={len(self.rec['points'])} "
            f"rows={self.grid.get('rows',0)} cols={self.grid.get('cols',0)} seg={seg_ct} blocked={blocked_ct} full_added={len(full_lines)}"
        )
        t3 = (
            "keys: 1 intersections 2 segments | "
            "intersections: a add, d delete-point, v add full-vertical, h add full-horizontal, x remove previously added full-line, r restore removed full-line, drag move | "
            "segments: a add-segment (click point A then B), d remove-segment, u restore-segment, then click | colors: v=blue h=red blocked=green(X) manual=yellow full-added=(v blue / h red) | s save | n/p next/prev | q quit"
        )

        pad = int(self.display_pad)
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3), (42, 42, 42), dtype=np.uint8)
        canvas[pad : pad + h, pad : pad + w] = vis
        cv2.rectangle(canvas, (pad - 1, pad - 1), (pad + w, pad + h), (130, 130, 130), 1)
        cv2.putText(canvas, t1, (pad + 12, pad + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, t2, (pad + 12, pad + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, t3, (pad + 12, pad + h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.image is None or self.rec is None:
            return
        pad = int(self.display_pad)
        h, w = self.image.shape[:2]
        x = int(np.clip(x - pad, 0, w - 1))
        y = int(np.clip(y - pad, 0, h - 1))
        self.grid = _sanitize_record(self.rec, self.image)

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state.mode == "point":
                if self.state.delete_mode:
                    pid = self._nearest_point_id(x, y)
                    if pid >= 0:
                        self._delete_point(pid)
                    self.state.delete_mode = False
                    return
                if self.state.restore_removed_full_mode:
                    self._restore_removed_full_line_from_click(x, y)
                    self.state.restore_removed_full_mode = False
                    return
                if self.state.remove_added_full_mode:
                    lid = self._nearest_added_full_line_id(x, y)
                    if lid >= 0:
                        self._remove_added_full_line(lid)
                        self.state.remove_added_full_mode = False
                    return
                if self.state.add_full_v_mode:
                    self._add_full_line_from_click("v", x, y)
                    self.state.add_full_v_mode = False
                    return
                if self.state.add_full_h_mode:
                    self._add_full_line_from_click("h", x, y)
                    self.state.add_full_h_mode = False
                    return
                if self.state.add_mode:
                    self._add_point(x, y)
                    return
                pid = self._nearest_point_id(x, y)
                if pid >= 0:
                    self.state.dragging = True
                    self.state.drag_kind = "point"
                    self.state.drag_idx = pid
                return

            if self.state.mode == "edge":
                if self.state.add_mode:
                    pid = self._nearest_point_id(x, y, tol=max(self.point_tol, 24))
                    if pid < 0:
                        return
                    if self.state.edge_add_start < 0:
                        self.state.edge_add_start = int(pid)
                        return
                    self._add_segment_between_points(int(self.state.edge_add_start), int(pid))
                    self.state.edge_add_start = -1
                    return
                seg = self._nearest_segment(x, y)
                if seg is None:
                    return
                if self.state.delete_mode:
                    self._remove_segment(seg)
                    self.state.delete_mode = False
                    return
                if self.state.restore_mode:
                    self._restore_segment(seg)
                    self.state.restore_mode = False
                    return
                return

        elif event == cv2.EVENT_MOUSEMOVE and self.state.dragging:
            if self.state.drag_kind == "point" and self.state.mode == "point":
                p = self._point_by_id(int(self.state.drag_idx))
                if p is not None:
                    px, py = _pt_clip(x, y, w, h)
                    p["x"], p["y"] = int(px), int(py)
                    self.grid = _sanitize_record(self.rec, self.image)
                return

        elif event == cv2.EVENT_LBUTTONUP:
            self.state.dragging = False
            self.state.drag_kind = ""
            self.state.drag_idx = -1

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
                self.state.mode = "point"
                self.state.delete_mode = False
                self.state.restore_mode = False
                self.state.edge_add_start = -1
                self.state.add_full_v_mode = False
                self.state.add_full_h_mode = False
                self.state.remove_added_full_mode = False
                self.state.restore_removed_full_mode = False
                continue
            if k == ord("2"):
                self.state.mode = "edge"
                self.state.add_mode = False
                self.state.delete_mode = False
                self.state.restore_mode = False
                self.state.edge_add_start = -1
                self.state.add_full_v_mode = False
                self.state.add_full_h_mode = False
                self.state.remove_added_full_mode = False
                self.state.restore_removed_full_mode = False
                continue
            if k == ord("a"):
                if self.state.mode == "point":
                    self.state.add_mode = not self.state.add_mode
                    if self.state.add_mode:
                        self.state.delete_mode = False
                        self.state.add_full_v_mode = False
                        self.state.add_full_h_mode = False
                        self.state.remove_added_full_mode = False
                        self.state.restore_removed_full_mode = False
                elif self.state.mode == "edge":
                    self.state.add_mode = not self.state.add_mode
                    self.state.edge_add_start = -1
                    if self.state.add_mode:
                        self.state.delete_mode = False
                        self.state.restore_mode = False
                continue
            if k == ord("d"):
                if self.state.mode == "point":
                    self.state.delete_mode = not self.state.delete_mode
                    if self.state.delete_mode:
                        self.state.add_mode = False
                        self.state.add_full_v_mode = False
                        self.state.add_full_h_mode = False
                        self.state.remove_added_full_mode = False
                        self.state.restore_removed_full_mode = False
                elif self.state.mode == "edge":
                    self.state.delete_mode = not self.state.delete_mode
                    if self.state.delete_mode:
                        self.state.add_mode = False
                        self.state.restore_mode = False
                        self.state.edge_add_start = -1
                continue
            if k == ord("v") and self.state.mode == "point":
                self.state.add_full_v_mode = not self.state.add_full_v_mode
                if self.state.add_full_v_mode:
                    self.state.add_mode = False
                    self.state.delete_mode = False
                    self.state.add_full_h_mode = False
                    self.state.remove_added_full_mode = False
                    self.state.restore_removed_full_mode = False
                continue
            if k == ord("h") and self.state.mode == "point":
                self.state.add_full_h_mode = not self.state.add_full_h_mode
                if self.state.add_full_h_mode:
                    self.state.add_mode = False
                    self.state.delete_mode = False
                    self.state.add_full_v_mode = False
                    self.state.remove_added_full_mode = False
                    self.state.restore_removed_full_mode = False
                continue
            if k == ord("x") and self.state.mode == "point":
                self.state.remove_added_full_mode = not self.state.remove_added_full_mode
                if self.state.remove_added_full_mode:
                    self.state.add_mode = False
                    self.state.delete_mode = False
                    self.state.add_full_v_mode = False
                    self.state.add_full_h_mode = False
                    self.state.restore_removed_full_mode = False
                continue
            if k == ord("r") and self.state.mode == "point":
                self.state.restore_removed_full_mode = not self.state.restore_removed_full_mode
                if self.state.restore_removed_full_mode:
                    self.state.add_mode = False
                    self.state.delete_mode = False
                    self.state.add_full_v_mode = False
                    self.state.add_full_h_mode = False
                    self.state.remove_added_full_mode = False
                continue
            if k == ord("u") and self.state.mode == "edge":
                self.state.restore_mode = not self.state.restore_mode
                if self.state.restore_mode:
                    self.state.add_mode = False
                    self.state.delete_mode = False
                    self.state.edge_add_start = -1
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


def _warp_quad(image: np.ndarray, quad_xy: np.ndarray, inset: int = 0) -> np.ndarray:
    pts = quad_xy.astype(np.float32)
    tl, tr, br, bl = pts
    w_top = float(np.linalg.norm(tr - tl))
    w_bot = float(np.linalg.norm(br - bl))
    h_left = float(np.linalg.norm(bl - tl))
    h_right = float(np.linalg.norm(br - tr))
    W = max(6, int(round(0.5 * (w_top + w_bot))))
    H = max(6, int(round(0.5 * (h_left + h_right))))
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(pts, dst)
    out = cv2.warpPerspective(image, M, (W, H))
    ins = max(0, int(inset))
    if ins > 0 and out.shape[1] > 2 * ins and out.shape[0] > 2 * ins:
        out = out[ins : out.shape[0] - ins, ins : out.shape[1] - ins]
    return out


def export_labels(labels_dir: Path, out_dir: Path, line_thickness: int = 3, cell_inset: int = 2) -> None:
    _ensure_dir(out_dir)
    img_dir = out_dir / "images"
    msk_dir = out_dir / "masks"
    meta_dir = out_dir / "meta"
    _ensure_dir(img_dir)
    _ensure_dir(msk_dir)
    _ensure_dir(meta_dir)

    recs = _collect_records(labels_dir)
    if not recs:
        print(f"No labels in {labels_dir}")
        return

    index_lines: list[str] = []

    for js in recs:
        rec = _load_record(js)
        img = cv2.imread(str(rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] failed image for {js.stem}")
            continue
        grid = _sanitize_record(rec, img)
        active = _active_edges(rec, grid)
        pmap = _point_map(rec["points"])
        point_rc = grid["point_rc"]
        rc_point = grid["rc_point"]
        rows = int(grid["rows"])
        cols = int(grid["cols"])
        stem = js.stem
        h, w = img.shape[:2]

        table = np.zeros((h, w), dtype=np.uint8)
        tq = np.array(rec["table_quad"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(table, [tq], 255)

        vmask = np.zeros((h, w), dtype=np.uint8)
        hmask = np.zeros((h, w), dtype=np.uint8)
        for a, b in sorted(active):
            if a not in pmap or b not in pmap:
                continue
            pa, pb = pmap[a], pmap[b]
            ra, ca = point_rc.get(a, (-1, -1))
            rb, cb = point_rc.get(b, (-1, -1))
            is_h = (ra == rb and ca != cb)
            target = hmask if is_h else vmask
            cv2.line(target, pa, pb, 255, int(max(1, line_thickness)), cv2.LINE_AA)

        p_img = img_dir / f"{stem}.png"
        p_t = msk_dir / f"{stem}_table.png"
        p_v = msk_dir / f"{stem}_v.png"
        p_h = msk_dir / f"{stem}_h.png"
        cv2.imwrite(str(p_img), img)
        cv2.imwrite(str(p_t), table)
        cv2.imwrite(str(p_v), vmask)
        cv2.imwrite(str(p_h), hmask)

        # Build base cells from adjacent intersection quads.
        base_cells: list[dict[str, Any]] = []
        base_lookup: dict[tuple[int, int], int] = {}
        for r in range(max(0, rows - 1)):
            for c in range(max(0, cols - 1)):
                ids = [
                    rc_point.get((r, c)),
                    rc_point.get((r, c + 1)),
                    rc_point.get((r + 1, c + 1)),
                    rc_point.get((r + 1, c)),
                ]
                if any(v is None for v in ids):
                    continue
                tl, tr, br, bl = [int(v) for v in ids]
                top = _edge_key(tl, tr) in active
                right = _edge_key(tr, br) in active
                bottom = _edge_key(bl, br) in active
                left = _edge_key(tl, bl) in active
                bid = len(base_cells)
                base_cells.append(
                    {
                        "base_id": int(bid),
                        "r": int(r),
                        "c": int(c),
                        "point_ids": [tl, tr, br, bl],
                        "edges": {
                            "top": bool(top),
                            "right": bool(right),
                            "bottom": bool(bottom),
                            "left": bool(left),
                        },
                    }
                )
                base_lookup[(r, c)] = int(bid)

        # Export OCR-ready base crops in strict row-major matrix order.
        cell_dir = out_dir / stem / "cells"
        _ensure_dir(cell_dir)
        for bc in base_cells:
            tl, tr, br, bl = bc["point_ids"]
            quad = np.array([pmap[tl], pmap[tr], pmap[br], pmap[bl]], dtype=np.float32)
            crop = _warp_quad(img, quad, inset=int(cell_inset))
            row = int(bc["r"])
            col = int(bc["c"])
            name = f"base_r{row:02d}_c{col:02d}.png"
            p = cell_dir / name
            cv2.imwrite(str(p), crop)
            bc["path"] = str(p.relative_to(out_dir))

        # Merge base cells where shared boundary segment is missing.
        n = len(base_cells)
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for bc in base_cells:
            r = int(bc["r"])
            c = int(bc["c"])
            a = int(bc["base_id"])

            # right neighbor uses shared vertical boundary at col c+1
            b = base_lookup.get((r, c + 1))
            if b is not None:
                p_top = rc_point.get((r, c + 1))
                p_bot = rc_point.get((r + 1, c + 1))
                if p_top is not None and p_bot is not None:
                    shared = _edge_key(int(p_top), int(p_bot))
                    if shared not in active:
                        union(a, int(b))

            # down neighbor uses shared horizontal boundary at row r+1
            d = base_lookup.get((r + 1, c))
            if d is not None:
                p_l = rc_point.get((r + 1, c))
                p_r = rc_point.get((r + 1, c + 1))
                if p_l is not None and p_r is not None:
                    shared = _edge_key(int(p_l), int(p_r))
                    if shared not in active:
                        union(a, int(d))

        groups: dict[int, list[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        merged_cells: list[dict[str, Any]] = []
        for cid, members in enumerate(sorted(groups.values(), key=lambda v: min(v))):
            rs = [int(base_cells[m]["r"]) for m in members]
            cs = [int(base_cells[m]["c"]) for m in members]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)

            tl = rc_point.get((r0, c0))
            tr = rc_point.get((r0, c1 + 1))
            br = rc_point.get((r1 + 1, c1 + 1))
            bl = rc_point.get((r1 + 1, c0))
            path = None
            if None not in (tl, tr, br, bl):
                quad = np.array([pmap[int(tl)], pmap[int(tr)], pmap[int(br)], pmap[int(bl)]], dtype=np.float32)
                crop = _warp_quad(img, quad, inset=int(cell_inset))
                name = f"cell_{cid:04d}_r{r0:02d}_c{c0:02d}_rs{(r1-r0+1):02d}_cs{(c1-c0+1):02d}.png"
                p = cell_dir / name
                cv2.imwrite(str(p), crop)
                path = str(p.relative_to(out_dir))

            merged_cells.append(
                {
                    "cell_id": int(cid),
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "rowspan": int(r1 - r0 + 1),
                    "colspan": int(c1 - c0 + 1),
                    "path": path,
                    "members": [int(m) for m in members],
                }
            )

        # Matrix preserve: top-left index [0][0].
        rows_base = max(0, rows - 1)
        cols_base = max(0, cols - 1)
        matrix: list[list[dict[str, Any]]] = [[{} for _ in range(cols_base)] for _ in range(rows_base)]

        base_by_rc = {(int(b["r"]), int(b["c"])): b for b in base_cells}
        owner_by_base: dict[int, dict[str, Any]] = {}
        for mc in merged_cells:
            for m in mc["members"]:
                owner_by_base[int(m)] = mc

        for r in range(rows_base):
            for c in range(cols_base):
                bc = base_by_rc.get((r, c))
                if bc is None:
                    matrix[r][c] = {"missing": True}
                    continue
                owner = owner_by_base.get(int(bc["base_id"]))
                anchor = bool(owner is not None and r == owner["r0"] and c == owner["c0"])
                matrix[r][c] = {
                    "base_cell_id": int(bc["base_id"]),
                    "base_path": bc.get("path"),
                    "cell_id": int(owner["cell_id"]) if owner is not None else None,
                    "anchor": bool(anchor),
                    "rowspan": int(owner["rowspan"]) if owner is not None else 1,
                    "colspan": int(owner["colspan"]) if owner is not None else 1,
                    "path": owner["path"] if (owner is not None and anchor) else None,
                }

        meta = {
            "image": str(p_img.resolve()),
            "source_image": rec.get("source_image", ""),
            "table_quad": rec["table_quad"],
            "rows_points": int(rows),
            "cols_points": int(cols),
            "rows_base": int(rows_base),
            "cols_base": int(cols_base),
            "points": rec["points"],
            "blocked_edges": rec["blocked_edges"],
            "base_cells": base_cells,
            "cells": merged_cells,
            "matrix": matrix,
        }
        (meta_dir / f"{stem}_cells.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        label_rec = {
            "source_image": rec.get("source_image", ""),
            "image": str(p_img.resolve()),
            "table_mask": str(p_t.resolve()),
            "v_mask": str(p_v.resolve()),
            "h_mask": str(p_h.resolve()),
            "source_intersection_json": str(js.resolve()),
        }
        (out_dir / f"{stem}.json").write_text(json.dumps(label_rec, indent=2), encoding="utf-8")
        index_lines.append(json.dumps(label_rec))
        print(f"[ok] {stem}")

    (out_dir / "index.jsonl").write_text("\n".join(index_lines) + ("\n" if index_lines else ""), encoding="utf-8")
    print(f"[ok] wrote {(out_dir / 'index.jsonl')}")


def write_overlays(labels_dir: Path, out_dir: Path) -> None:
    _ensure_dir(out_dir)
    recs = _collect_records(labels_dir)
    if not recs:
        print(f"No labels in {labels_dir}")
        return
    for js in recs:
        rec = _load_record(js)
        img = cv2.imread(str(rec["image"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        grid = _sanitize_record(rec, img)
        blocked = _blocked_set_from_record(rec)
        manual = _manual_set_from_record(rec)
        full_lines = _full_line_entries(rec)
        pmap = _point_map(rec["points"])
        point_rc = grid["point_rc"]

        vis = img.copy()
        tq = np.array(rec["table_quad"], dtype=np.int32).reshape(-1, 1, 2)
        fill = vis.copy()
        cv2.fillPoly(fill, [tq], (0, 140, 0))
        vis = cv2.addWeighted(vis, 0.88, fill, 0.12, 0.0)
        cv2.polylines(vis, [tq], True, (0, 220, 0), 2, cv2.LINE_AA)

        for a, b in sorted(grid["default_edges"]):
            if a not in pmap or b not in pmap:
                continue
            pa, pb = pmap[a], pmap[b]
            ra, ca = point_rc.get(a, (-1, -1))
            rb, cb = point_rc.get(b, (-1, -1))
            is_h = (ra == rb and ca != cb)
            if (a, b) in blocked:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255) if is_h else (255, 0, 0)
            cv2.line(vis, pa, pb, color, 2, cv2.LINE_AA)
            if (a, b) in blocked:
                mx = int(round(0.5 * (pa[0] + pb[0])))
                my = int(round(0.5 * (pa[1] + pb[1])))
                cv2.line(vis, (mx - 4, my - 4), (mx + 4, my + 4), (0, 0, 0), 1, cv2.LINE_AA)
                cv2.line(vis, (mx - 4, my + 4), (mx + 4, my - 4), (0, 0, 0), 1, cv2.LINE_AA)
        for a, b in sorted(manual):
            if a in pmap and b in pmap:
                cv2.line(vis, pmap[a], pmap[b], (0, 255, 255), 2, cv2.LINE_AA)
        for it in full_lines:
            pts = [pmap[pid] for pid in it["point_ids"] if pid in pmap]
            if len(pts) < 2:
                continue
            if str(it["axis"]) == "v":
                pts = sorted(pts, key=lambda p: (p[1], p[0]))
                color = (255, 0, 0)
            else:
                pts = sorted(pts, key=lambda p: (p[0], p[1]))
                color = (0, 0, 255)
            cv2.line(vis, pts[0], pts[-1], color, 2, cv2.LINE_AA)

        for p in rec["points"]:
            cv2.circle(vis, (int(p["x"]), int(p["y"])), 4, (30, 220, 255), -1)

        cv2.putText(vis, js.stem, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2, cv2.LINE_AA)
        out = out_dir / f"{js.stem}_overlay.png"
        cv2.imwrite(str(out), vis)
    print(f"[ok] wrote overlays to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Intersection-first scorecard annotation tool")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="Create fresh intersection records")
    b.add_argument("--images_dir", required=True)
    b.add_argument("--labels_dir", default="ml_intersection_labels")
    b.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("edit", help="Interactive editor")
    e.add_argument("--labels_dir", default="ml_intersection_labels")
    e.add_argument("--start", type=int, default=0)

    o = sub.add_parser("overlay", help="Write overlay previews")
    o.add_argument("--labels_dir", default="ml_intersection_labels")
    o.add_argument("--out_dir", default=None)

    x = sub.add_parser("export", help="Export masks + matrix-preserving crops")
    x.add_argument("--labels_dir", default="ml_intersection_labels")
    x.add_argument("--out_dir", default="ml_intersection_labels_exported")
    x.add_argument("--line_thickness", type=int, default=3)
    x.add_argument("--cell_inset", type=int, default=2)

    args = p.parse_args()

    if args.cmd == "bootstrap":
        bootstrap_labels(Path(args.images_dir), Path(args.labels_dir), overwrite=bool(args.overwrite))
        return
    if args.cmd == "edit":
        ed = IntersectionEditor(Path(args.labels_dir), start=int(args.start))
        ed.run()
        return
    if args.cmd == "overlay":
        out_dir = Path(args.out_dir) if args.out_dir else (Path(args.labels_dir) / "overlays")
        write_overlays(Path(args.labels_dir), out_dir)
        return
    if args.cmd == "export":
        export_labels(
            labels_dir=Path(args.labels_dir),
            out_dir=Path(args.out_dir),
            line_thickness=int(args.line_thickness),
            cell_inset=int(args.cell_inset),
        )
        return


if __name__ == "__main__":
    main()
