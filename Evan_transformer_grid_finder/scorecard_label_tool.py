"""Assisted ground-truth labeling for scorecard grid structure.

Efficient workflow:
1) bootstrap: auto-prefill table/v/h masks from current extractor.
2) edit: paint corrections quickly with OpenCV UI.

Outputs per sample:
- labels_dir/images/<stem>.png       (preprocessed/upright image)
- labels_dir/masks/<stem>_table.png  (table region mask)
- labels_dir/masks/<stem>_v.png      (vertical separator mask)
- labels_dir/masks/<stem>_h.png      (horizontal separator mask)
- labels_dir/<stem>.json             (record tying image + masks together)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import shutil

import cv2
import numpy as np

from scorecard_hough_extraction import extract_scorecard_grids, HoughConfig


MASK_TABLE = "table"
MASK_V = "v"
MASK_H = "h"
MODE_ORDER = [MASK_TABLE, MASK_V, MASK_H]
MODE_COLOR = {
    MASK_TABLE: (0, 255, 0),   # green
    MASK_V: (255, 80, 80),     # blue-ish in BGR
    MASK_H: (80, 80, 255),     # red-ish in BGR
}


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _collect_images(images_dir: Path) -> list[Path]:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    out: list[Path] = []
    for e in exts:
        out.extend(images_dir.glob(e))
    return sorted(set(out))


def _paths_for_stem(labels_dir: Path, stem: str) -> dict[str, Path]:
    p_img_dir = labels_dir / "images"
    p_msk_dir = labels_dir / "masks"
    _ensure_dir(p_img_dir)
    _ensure_dir(p_msk_dir)
    return {
        "json": labels_dir / f"{stem}.json",
        "image": p_img_dir / f"{stem}.png",
        "table": p_msk_dir / f"{stem}_table.png",
        "v": p_msk_dir / f"{stem}_v.png",
        "h": p_msk_dir / f"{stem}_h.png",
    }


def _write_record(labels_dir: Path, stem: str, source_image: Path) -> None:
    pp = _paths_for_stem(labels_dir, stem)
    rec = {
        "source_image": str(source_image.resolve()),
        "image": str(pp["image"].resolve()),
        "table_mask": str(pp["table"].resolve()),
        "v_mask": str(pp["v"].resolve()),
        "h_mask": str(pp["h"].resolve()),
    }
    pp["json"].write_text(json.dumps(rec, indent=2), encoding="utf-8")


def _load_record(json_path: Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def _is_label_record(data: object) -> bool:
    if not isinstance(data, dict):
        return False
    required = ("image", "table_mask", "v_mask", "h_mask")
    return all(k in data for k in required)


def _collect_label_records(labels_dir: Path) -> list[Path]:
    out: list[Path] = []
    for js in sorted(labels_dir.glob("*.json")):
        try:
            data = _load_record(js)
        except Exception:
            continue
        if _is_label_record(data):
            out.append(js)
    return out


def _prefill_from_extractor(image_path: Path, labels_dir: Path, stem: str, hough_cfg: HoughConfig) -> None:
    tmp = labels_dir / "_prefill_tmp" / stem
    _ensure_dir(tmp)

    result = extract_scorecard_grids(image_path, tmp, hough_cfg=hough_cfg)
    prep_img = result.preprocess.image_bgr
    h, w = prep_img.shape[:2]

    table_mask = np.zeros((h, w), dtype=np.uint8)
    v_mask = np.zeros((h, w), dtype=np.uint8)
    h_mask = np.zeros((h, w), dtype=np.uint8)

    if result.table_grids:
        g = result.table_grids[0]
        x, y, bw, bh = g.bbox
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w - 1, x + bw)
        y1 = min(h - 1, y + bh)
        cv2.rectangle(table_mask, (x0, y0), (x1, y1), 255, -1)

        for xv in g.x_lines:
            xx = int(np.clip(xv, 0, w - 1))
            cv2.line(v_mask, (xx, y0), (xx, y1), 255, 3)
        for yv in g.y_lines:
            yy = int(np.clip(yv, 0, h - 1))
            cv2.line(h_mask, (x0, yy), (x1, yy), 255, 3)

    pp = _paths_for_stem(labels_dir, stem)
    cv2.imwrite(str(pp["image"]), prep_img)
    cv2.imwrite(str(pp["table"]), table_mask)
    cv2.imwrite(str(pp["v"]), v_mask)
    cv2.imwrite(str(pp["h"]), h_mask)
    _write_record(labels_dir, stem, source_image=image_path)

    shutil.rmtree(tmp, ignore_errors=True)


def bootstrap_labels(images_dir: Path, labels_dir: Path, overwrite: bool = False) -> None:
    _ensure_dir(labels_dir)
    imgs = _collect_images(images_dir)
    if not imgs:
        print(f"No images found in {images_dir}")
        return

    hcfg = HoughConfig()
    for p in imgs:
        stem = p.stem
        pp = _paths_for_stem(labels_dir, stem)
        if pp["json"].exists() and not overwrite:
            print(f"[skip] {p.name} (label exists)")
            continue
        _prefill_from_extractor(p, labels_dir, stem, hcfg)
        print(f"[ok] {p.name}")


def _blend_mask(img: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    out = img.copy()
    m = mask > 0
    if not np.any(m):
        return out
    layer = np.zeros_like(out, dtype=np.uint8)
    layer[:, :] = color
    out[m] = cv2.addWeighted(out[m], 1.0 - alpha, layer[m], alpha, 0)
    return out


@dataclass
class EditorState:
    labels_dir: Path
    records: list[Path]
    idx: int = 0
    mode: str = MASK_TABLE
    dragging: bool = False
    drag_kind: str = ""
    drag_idx: int = -1
    add_mode: bool = False
    delete_mode: bool = False


class LabelEditor:
    def __init__(self, labels_dir: Path, start_index: int = 0):
        self.labels_dir = labels_dir
        self.records = _collect_label_records(labels_dir)
        if not self.records:
            raise RuntimeError(f"No JSON labels found in {labels_dir}. Run bootstrap first.")
        self.state = EditorState(
            labels_dir=labels_dir,
            records=self.records,
            idx=max(0, min(start_index, len(self.records) - 1)),
        )

        self.window = "scorecard_label_tool"
        self.image: np.ndarray | None = None
        self.table: np.ndarray | None = None
        self.vmask: np.ndarray | None = None
        self.hmask: np.ndarray | None = None
        self.current_json: Path | None = None
        self.table_bbox: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1
        self.v_lines: list[int] = []
        self.h_lines: list[int] = []
        self.display_pad: int = 44
        self.display_bg_color: tuple[int, int, int] = (42, 42, 42)
        self.corner_tol_px: int = 72
        self.line_tol_px: int = 22

    def _mask_bbox(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        h, w = mask.shape[:2]
        if xs.size == 0 or ys.size == 0:
            return (0, 0, w - 1, h - 1)
        return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    def _extract_line_positions(self, mask: np.ndarray, axis: str) -> list[int]:
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

    def _sanitize_geometry(self) -> None:
        h, w = self.image.shape[:2]
        if self.table_bbox is None:
            self.table_bbox = (0, 0, w - 1, h - 1)
        x0, y0, x1, y1 = self.table_bbox
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        self.table_bbox = (x0, y0, x1, y1)
        self.v_lines = sorted(set(int(np.clip(v, x0, x1)) for v in self.v_lines))
        self.h_lines = sorted(set(int(np.clip(v, y0, y1)) for v in self.h_lines))

    def _geometry_to_masks(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = self.image.shape[:2]
        table = np.zeros((h, w), dtype=np.uint8)
        vmask = np.zeros((h, w), dtype=np.uint8)
        hmask = np.zeros((h, w), dtype=np.uint8)
        if self.table_bbox is None:
            return table, vmask, hmask
        x0, y0, x1, y1 = self.table_bbox
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        cv2.rectangle(table, (x0, y0), (x1, y1), 255, -1)
        for xv in self.v_lines:
            xx = int(np.clip(xv, x0, x1))
            cv2.line(vmask, (xx, y0), (xx, y1), 255, 3)
        for yv in self.h_lines:
            yy = int(np.clip(yv, y0, y1))
            cv2.line(hmask, (x0, yy), (x1, yy), 255, 3)
        return table, vmask, hmask

    def _nearest_corner(self, x: int, y: int, tol: int | None = None) -> int:
        if self.table_bbox is None:
            return -1
        if tol is None:
            if self.image is not None:
                dyn = int(round(0.06 * min(self.image.shape[:2])))
                tol = max(self.corner_tol_px, dyn)
            else:
                tol = self.corner_tol_px
        x0, y0, x1, y1 = self.table_bbox
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        best_idx = -1
        best_d2 = 1e18
        for i, (cx, cy) in enumerate(corners):
            d2 = (cx - x) * (cx - x) + (cy - y) * (cy - y)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = i
        if best_d2 <= tol * tol:
            return best_idx
        return -1

    def _nearest_line_index(self, x: int, y: int, axis: str, tol: int | None = None) -> int:
        vals = self.v_lines if axis == "v" else self.h_lines
        if not vals:
            return -1
        if tol is None:
            tol = self.line_tol_px
        pos = int(x if axis == "v" else y)
        d = [abs(int(v) - pos) for v in vals]
        idx = int(np.argmin(np.array(d, dtype=np.int32)))
        if d[idx] <= tol:
            return idx
        return -1

    def _load_current(self) -> None:
        self.current_json = self.state.records[self.state.idx]
        rec = _load_record(self.current_json)
        self.image = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
        self.table = cv2.imread(rec["table_mask"], cv2.IMREAD_GRAYSCALE)
        self.vmask = cv2.imread(rec["v_mask"], cv2.IMREAD_GRAYSCALE)
        self.hmask = cv2.imread(rec["h_mask"], cv2.IMREAD_GRAYSCALE)
        if self.image is None or self.table is None or self.vmask is None or self.hmask is None:
            raise RuntimeError(f"Failed to load record: {self.current_json}")
        self.table_bbox = self._mask_bbox(self.table)
        self.v_lines = self._extract_line_positions(self.vmask, axis="v")
        self.h_lines = self._extract_line_positions(self.hmask, axis="h")
        self._sanitize_geometry()
        self.state.dragging = False
        self.state.drag_kind = ""
        self.state.drag_idx = -1
        self.state.add_mode = False
        self.state.delete_mode = False

    def _save_current(self) -> None:
        if self.current_json is None:
            return
        rec = _load_record(self.current_json)
        self._sanitize_geometry()
        table, vmask, hmask = self._geometry_to_masks()
        self.table = table
        self.vmask = vmask
        self.hmask = hmask
        cv2.imwrite(rec["table_mask"], table)
        cv2.imwrite(rec["v_mask"], vmask)
        cv2.imwrite(rec["h_mask"], hmask)

    def _render(self) -> np.ndarray:
        table, vmask, hmask = self._geometry_to_masks()
        vis = self.image.copy()
        vis = _blend_mask(vis, table, MODE_COLOR[MASK_TABLE], alpha=0.22)
        vis = _blend_mask(vis, vmask, MODE_COLOR[MASK_V], alpha=0.50)
        vis = _blend_mask(vis, hmask, MODE_COLOR[MASK_H], alpha=0.50)

        if self.table_bbox is not None:
            x0, y0, x1, y1 = self.table_bbox
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 220, 0), 2)
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            for c in corners:
                cv2.circle(vis, c, 9, (0, 255, 255), -1)
                cv2.circle(vis, c, 12, (0, 180, 255), 1)
        for xv in self.v_lines:
            cv2.line(vis, (xv, 0), (xv, vis.shape[0] - 1), (255, 80, 80), 1)
        for yv in self.h_lines:
            cv2.line(vis, (0, yv), (vis.shape[1] - 1, yv), (80, 80, 255), 1)

        stem = self.current_json.stem if self.current_json else ""
        text1 = f"{self.state.idx+1}/{len(self.state.records)} {stem}"
        add_tag = " add=ON" if self.state.add_mode else ""
        del_tag = " del=ON" if self.state.delete_mode else ""
        text2 = f"mode={self.state.mode}{add_tag}{del_tag} v={len(self.v_lines)} h={len(self.h_lines)}"
        text3 = "keys: 1 table-corners, 2 vertical, 3 horizontal, a add-line, d delete-line, L click+drag moves, s save, n/p next/prev, q quit"
        pad = int(self.display_pad)
        h, w = vis.shape[:2]
        canvas = np.full((h + 2 * pad, w + 2 * pad, 3), self.display_bg_color, dtype=np.uint8)
        canvas[pad : pad + h, pad : pad + w] = vis
        cv2.rectangle(canvas, (pad - 1, pad - 1), (pad + w, pad + h), (130, 130, 130), 1)

        cv2.putText(canvas, text1, (pad + 12, pad + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, text2, (pad + 12, pad + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, text3, (pad + 12, pad + h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def _mouse_cb(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if self.image is None:
            return
        pad = int(self.display_pad)
        x = int(np.clip(x - pad, 0, self.image.shape[1] - 1))
        y = int(np.clip(y - pad, 0, self.image.shape[0] - 1))
        self._sanitize_geometry()
        x0, y0, x1, y1 = self.table_bbox

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state.mode == MASK_TABLE:
                self.state.add_mode = False
                self.state.delete_mode = False
                cidx = self._nearest_corner(x, y)
                if cidx >= 0:
                    self.state.dragging = True
                    self.state.drag_kind = "corner"
                    self.state.drag_idx = cidx
            elif self.state.mode in (MASK_V, MASK_H):
                axis = "v" if self.state.mode == MASK_V else "h"
                if self.state.delete_mode:
                    idx = self._nearest_line_index(x, y, axis=axis)
                    if idx >= 0:
                        if axis == "v":
                            self.v_lines.pop(idx)
                        else:
                            self.h_lines.pop(idx)
                    self.state.delete_mode = False
                elif self.state.add_mode:
                    if axis == "v":
                        self.v_lines.append(int(np.clip(x, x0, x1)))
                        self.v_lines = sorted(set(self.v_lines))
                    else:
                        self.h_lines.append(int(np.clip(y, y0, y1)))
                        self.h_lines = sorted(set(self.h_lines))
                    self.state.add_mode = False
                else:
                    idx = self._nearest_line_index(x, y, axis=axis)
                    if idx >= 0:
                        self.state.dragging = True
                        self.state.drag_kind = "line"
                        self.state.drag_idx = idx
        elif event == cv2.EVENT_MOUSEMOVE and self.state.dragging:
            if self.state.drag_kind == "corner" and self.state.mode == MASK_TABLE:
                cx = int(np.clip(x, 0, self.image.shape[1] - 1))
                cy = int(np.clip(y, 0, self.image.shape[0] - 1))
                # Corner semantics on axis-aligned box:
                # 0: top-left, 1: top-right, 2: bottom-right, 3: bottom-left
                x0c, y0c, x1c, y1c = x0, y0, x1, y1
                if self.state.drag_idx == 0:
                    x0c = int(np.clip(cx, 0, x1c))
                    y0c = int(np.clip(cy, 0, y1c))
                elif self.state.drag_idx == 1:
                    x1c = int(np.clip(cx, x0c, self.image.shape[1] - 1))
                    y0c = int(np.clip(cy, 0, y1c))
                elif self.state.drag_idx == 2:
                    x1c = int(np.clip(cx, x0c, self.image.shape[1] - 1))
                    y1c = int(np.clip(cy, y0c, self.image.shape[0] - 1))
                elif self.state.drag_idx == 3:
                    x0c = int(np.clip(cx, 0, x1c))
                    y1c = int(np.clip(cy, y0c, self.image.shape[0] - 1))
                self.table_bbox = (x0c, y0c, x1c, y1c)
                self._sanitize_geometry()
            elif self.state.drag_kind == "line" and self.state.mode in (MASK_V, MASK_H):
                if self.state.mode == MASK_V and 0 <= self.state.drag_idx < len(self.v_lines):
                    self.v_lines[self.state.drag_idx] = int(np.clip(x, x0, x1))
                    self.v_lines = sorted(set(self.v_lines))
                elif self.state.mode == MASK_H and 0 <= self.state.drag_idx < len(self.h_lines):
                    self.h_lines[self.state.drag_idx] = int(np.clip(y, y0, y1))
                    self.h_lines = sorted(set(self.h_lines))
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
                self.state.mode = MASK_TABLE
                self.state.add_mode = False
                self.state.delete_mode = False
                continue
            if k == ord("2"):
                self.state.mode = MASK_V
                self.state.add_mode = False
                self.state.delete_mode = False
                continue
            if k == ord("3"):
                self.state.mode = MASK_H
                self.state.add_mode = False
                self.state.delete_mode = False
                continue
            if k == ord("a"):
                if self.state.mode in (MASK_V, MASK_H):
                    self.state.add_mode = not self.state.add_mode
                    if self.state.add_mode:
                        self.state.delete_mode = False
                continue
            if k == ord("d"):
                if self.state.mode in (MASK_V, MASK_H):
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


def write_index(labels_dir: Path) -> None:
    out = labels_dir / "index.jsonl"
    lines = []
    for js in sorted(labels_dir.glob("*.json")):
        data = _load_record(js)
        if not isinstance(data, dict):
            continue
        if "image" not in data or "table_mask" not in data or "v_mask" not in data or "h_mask" not in data:
            continue
        lines.append(json.dumps(data))
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"[ok] wrote {out}")


def write_overlays(labels_dir: Path, out_dir: Path, alpha_table: float = 0.22, alpha_lines: float = 0.50) -> None:
    """Render one verification overlay per label record."""

    _ensure_dir(out_dir)
    recs = sorted(labels_dir.glob("*.json"))
    if not recs:
        print(f"No JSON labels found in {labels_dir}")
        return

    n_written = 0
    n_skipped = 0
    for js in recs:
        try:
            rec = _load_record(js)
        except Exception:
            n_skipped += 1
            continue
        if not isinstance(rec, dict):
            n_skipped += 1
            continue
        required = ("image", "table_mask", "v_mask", "h_mask")
        if any(k not in rec for k in required):
            n_skipped += 1
            continue

        img = cv2.imread(str(rec["image"]), cv2.IMREAD_COLOR)
        table = cv2.imread(str(rec["table_mask"]), cv2.IMREAD_GRAYSCALE)
        vmask = cv2.imread(str(rec["v_mask"]), cv2.IMREAD_GRAYSCALE)
        hmask = cv2.imread(str(rec["h_mask"]), cv2.IMREAD_GRAYSCALE)
        if img is None or table is None or vmask is None or hmask is None:
            n_skipped += 1
            continue

        vis = img.copy()
        vis = _blend_mask(vis, table, MODE_COLOR[MASK_TABLE], alpha=alpha_table)
        vis = _blend_mask(vis, vmask, MODE_COLOR[MASK_V], alpha=alpha_lines)
        vis = _blend_mask(vis, hmask, MODE_COLOR[MASK_H], alpha=alpha_lines)

        stem = js.stem
        cv2.putText(vis, stem, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(
            vis,
            "table=green  vertical=blue  horizontal=red",
            (14, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out_path = out_dir / f"{stem}_overlay.png"
        cv2.imwrite(str(out_path), vis)
        n_written += 1

    print(f"[ok] wrote {n_written} overlays to {out_dir}")
    if n_skipped:
        print(f"[info] skipped {n_skipped} non-record/invalid JSON files")


def write_label_queue(images_dir: Path, labels_dir: Path) -> None:
    """Rank images by likely extraction uncertainty (hardest first).

    This gives an efficient manual-labeling order for active supervision.
    """

    imgs = _collect_images(images_dir)
    if not imgs:
        print(f"No images found in {images_dir}")
        return

    _ensure_dir(labels_dir)
    tmp_root = labels_dir / "_queue_tmp"
    _ensure_dir(tmp_root)

    hcfg = HoughConfig()
    ranked = []
    for p in imgs:
        tmp_out = tmp_root / p.stem
        try:
            result = extract_scorecard_grids(p, tmp_out, hough_cfg=hcfg)
            if not result.table_grids:
                uncertainty = 1.0
                info = {"tables": 0, "x_lines": 0, "y_lines": 0, "bbox_area_ratio": 0.0}
            else:
                g = result.table_grids[0]
                x_cnt = len(g.x_lines)
                y_cnt = len(g.y_lines)
                h, w = result.preprocess.image_bgr.shape[:2]
                bx, by, bw, bh = g.bbox
                area_ratio = float(bw * bh) / float(max(1, h * w))

                # Coarse priors for golf scorecards:
                # rows: usually around 8..18, cols: usually around 8..26.
                x_score = max(0.0, 1.0 - abs(x_cnt - 16) / 16.0)
                y_score = max(0.0, 1.0 - abs(y_cnt - 12) / 12.0)
                area_score = 1.0 if 0.08 <= area_ratio <= 0.90 else 0.35
                table_score = 1.0 if 1 <= len(result.table_grids) <= 2 else 0.25
                confidence = 0.40 * x_score + 0.35 * y_score + 0.15 * area_score + 0.10 * table_score
                uncertainty = float(np.clip(1.0 - confidence, 0.0, 1.0))
                info = {
                    "tables": int(len(result.table_grids)),
                    "x_lines": int(x_cnt),
                    "y_lines": int(y_cnt),
                    "bbox_area_ratio": round(float(area_ratio), 5),
                }
        except Exception:
            uncertainty = 1.0
            info = {"tables": 0, "x_lines": 0, "y_lines": 0, "bbox_area_ratio": 0.0}
        ranked.append(
            {
                "image": str(p.resolve()),
                "stem": p.stem,
                "uncertainty": round(float(uncertainty), 5),
                "has_label": bool((labels_dir / f"{p.stem}.json").exists()),
                **info,
            }
        )

    ranked.sort(key=lambda d: (d["uncertainty"], not d["has_label"]), reverse=True)

    queue_json = labels_dir / "label_queue.json"
    queue_txt = labels_dir / "label_queue.txt"
    queue_json.write_text(json.dumps(ranked, indent=2), encoding="utf-8")

    lines = []
    for i, r in enumerate(ranked, start=1):
        lines.append(
            f"{i:03d}. {Path(r['image']).name}  "
            f"uncertainty={r['uncertainty']:.3f}  "
            f"tables={r['tables']}  x={r['x_lines']}  y={r['y_lines']}  "
            f"labeled={r['has_label']}"
        )
    queue_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    shutil.rmtree(tmp_root, ignore_errors=True)
    print(f"[ok] wrote {queue_json}")
    print(f"[ok] wrote {queue_txt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Assisted scorecard label tool")
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="Auto-create initial labels from current extractor")
    b.add_argument("--images_dir", required=True)
    b.add_argument("--labels_dir", default="labels")
    b.add_argument("--overwrite", action="store_true")

    e = sub.add_parser("edit", help="Edit label masks interactively")
    e.add_argument("--images_dir", required=True, help="Used only to bootstrap missing labels")
    e.add_argument("--labels_dir", default="labels")
    e.add_argument("--start", type=int, default=0)
    e.add_argument("--bootstrap_missing", action="store_true")

    i = sub.add_parser("index", help="Write labels/index.jsonl")
    i.add_argument("--labels_dir", default="labels")

    q = sub.add_parser("queue", help="Rank images by likely labeling value (hardest first)")
    q.add_argument("--images_dir", required=True)
    q.add_argument("--labels_dir", default="labels")

    o = sub.add_parser("overlay", help="Write per-label verification overlays")
    o.add_argument("--labels_dir", default="labels")
    o.add_argument("--out_dir", default=None)
    o.add_argument("--alpha_table", type=float, default=0.22)
    o.add_argument("--alpha_lines", type=float, default=0.50)

    args = parser.parse_args()

    if args.cmd == "bootstrap":
        bootstrap_labels(Path(args.images_dir), Path(args.labels_dir), overwrite=bool(args.overwrite))
        write_index(Path(args.labels_dir))
        return

    if args.cmd == "edit":
        labels_dir = Path(args.labels_dir)
        _ensure_dir(labels_dir)
        if args.bootstrap_missing:
            bootstrap_labels(Path(args.images_dir), labels_dir, overwrite=False)
        editor = LabelEditor(labels_dir, start_index=args.start)
        editor.run()
        write_index(labels_dir)
        return

    if args.cmd == "index":
        write_index(Path(args.labels_dir))
        return

    if args.cmd == "queue":
        write_label_queue(Path(args.images_dir), Path(args.labels_dir))
        return

    if args.cmd == "overlay":
        labels_dir = Path(args.labels_dir)
        out_dir = Path(args.out_dir) if args.out_dir else (labels_dir / "overlays")
        write_overlays(
            labels_dir=labels_dir,
            out_dir=out_dir,
            alpha_table=float(args.alpha_table),
            alpha_lines=float(args.alpha_lines),
        )


if __name__ == "__main__":
    main()
