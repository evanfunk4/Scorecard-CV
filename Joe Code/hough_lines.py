"""
hough_lines.py  —  Golf scorecard grid extractor + RCNN training data generator
 
Usage:
    python hough_lines.py --input images/ --output training_data/
    python hough_lines.py --input images/ --output training_data/ --debug
 
Outputs:
    training_data/
        cells/            <- one PNG per detected cell (cropped from original)
        images/           <- copies of originals (for RCNN full-image training)
        annotations.json  <- COCO-format bounding boxes for all cells
        debug/            <- only with --debug: originals with grid overlaid
"""
 
import argparse
import json
import shutil
from pathlib import Path
 
import cv2
import numpy as np
 
 
# ---------------------------------------------------------------------------
# Morphological grid line extraction
# ---------------------------------------------------------------------------
 
def extract_grid_mask(img_gray: np.ndarray,
                      h_kernel_ratio: float = 0.25,
                      v_kernel_ratio: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    """
    Isolate horizontal and vertical grid lines using morphological opening.
 
    The kernel lengths are set as a fraction of the image dimension so they
    are always longer than any printed text/number but shorter than full grid
    lines. This erases cell content before line detection runs.
 
    Returns:
        h_mask: binary image containing only horizontal lines
        v_mask: binary image containing only vertical lines
    """
    h, w = img_gray.shape
 
    # Binarise: dark lines on light background
    _, binary = cv2.threshold(img_gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
    # Horizontal kernel: wider than any text char, spans the full grid width
    h_len = max(20, int(w * h_kernel_ratio))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    h_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)
 
    # Vertical kernel: taller than any text char, spans the full grid height
    v_len = max(20, int(h * v_kernel_ratio))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    v_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)
 
    # Light dilation to close small gaps left by ink/compression variation
    h_mask = cv2.dilate(h_mask, np.ones((3, 1), np.uint8), iterations=1)
    v_mask = cv2.dilate(v_mask, np.ones((1, 3), np.uint8), iterations=1)
 
    return h_mask, v_mask
 
 
# ---------------------------------------------------------------------------
# Line coordinate extraction from masks
# ---------------------------------------------------------------------------
 
def mask_to_lines(mask: np.ndarray,
                  axis: str,
                  cluster_gap: int = 8) -> list[int]:
    """
    Project a binary line mask onto one axis and find peak positions.
 
    axis='h' -> project columns, return y-coordinates of horizontal lines
    axis='v' -> project rows,    return x-coordinates of vertical lines
    """
    if axis == 'h':
        projection = np.sum(mask, axis=1).astype(np.float32)
    else:
        projection = np.sum(mask, axis=0).astype(np.float32)
 
    # A real grid line lights up many pixels; noise does not
    threshold = projection.max() * 0.25
    candidates = np.where(projection > threshold)[0].tolist()
 
    return _cluster(candidates, gap=cluster_gap)
 
 
def _cluster(coords: list[int], gap: int) -> list[int]:
    """Merge nearby detections into a single representative coordinate."""
    if not coords:
        return []
    clusters, current = [], [coords[0]]
    for c in coords[1:]:
        if c - current[-1] <= gap:
            current.append(c)
        else:
            clusters.append(int(np.mean(current)))
            current = [c]
    clusters.append(int(np.mean(current)))
    return clusters
 
 
def detect_grid_lines(img_gray: np.ndarray,
                      h_kernel_ratio: float = 0.25,
                      v_kernel_ratio: float = 0.25) -> tuple[list[int], list[int]]:
    """
    Full pipeline: morphological extraction -> projection -> clustering.
 
    Returns:
        h_lines: sorted y-coordinates of horizontal grid lines
        v_lines: sorted x-coordinates of vertical grid lines
    """
    h_mask, v_mask = extract_grid_mask(img_gray, h_kernel_ratio, v_kernel_ratio)
    h_lines = mask_to_lines(h_mask, axis='h')
    v_lines = mask_to_lines(v_mask, axis='v')
    return h_lines, v_lines
 
 
# ---------------------------------------------------------------------------
# Cell extraction
# ---------------------------------------------------------------------------
 
def extract_cells(h_lines: list[int],
                  v_lines: list[int],
                  min_cell_px: int = 15) -> list[dict]:
    """
    Enumerate every (row, col) cell defined by adjacent grid line pairs.
 
    Returns list of dicts: { 'x', 'y', 'w', 'h', 'row', 'col' }
    """
    cells = []
    for row_idx, (y0, y1) in enumerate(zip(h_lines, h_lines[1:])):
        for col_idx, (x0, x1) in enumerate(zip(v_lines, v_lines[1:])):
            cw, ch = x1 - x0, y1 - y0
            if cw < min_cell_px or ch < min_cell_px:
                continue
            cells.append({
                'x': x0, 'y': y0, 'w': cw, 'h': ch,
                'row': row_idx, 'col': col_idx,
            })
    return cells
 
 
def crop_cell(img: np.ndarray, cell: dict, padding: int = 3) -> np.ndarray:
    """
    Crop a single cell from the original image.
    Padding moves inward from the grid line so the border itself is excluded.
    """
    h_img, w_img = img.shape[:2]
    x  = min(w_img, cell['x'] + padding)
    y  = min(h_img, cell['y'] + padding)
    x2 = max(0,     cell['x'] + cell['w'] - padding)
    y2 = max(0,     cell['y'] + cell['h'] - padding)
    return img[y:y2, x:x2]
 
 
# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------
 
def draw_debug(img: np.ndarray,
               h_lines: list[int],
               v_lines: list[int],
               cells: list[dict]) -> np.ndarray:
    """Overlay detected grid lines and cell boxes on a copy of the image."""
    out = img.copy()
    h_img, w_img = out.shape[:2]
 
    for y in h_lines:
        cv2.line(out, (0, y), (w_img, y), (0, 210, 0), 1)   # green H-lines
    for x in v_lines:
        cv2.line(out, (x, 0), (x, h_img), (0, 0, 210), 1)   # red V-lines
    for cell in cells:
        cv2.rectangle(out,
                      (cell['x'], cell['y']),
                      (cell['x'] + cell['w'], cell['y'] + cell['h']),
                      (210, 0, 0), 1)                         # blue cell boxes
    return out
 
 
# ---------------------------------------------------------------------------
# COCO annotation builder
# ---------------------------------------------------------------------------
 
def build_coco(image_records: list[dict],
               annotation_records: list[dict]) -> dict:
    return {
        "info": {"description": "Golf scorecard grid cells", "version": "1.0"},
        "licenses": [],
        "categories": [{"id": 1, "name": "cell", "supercategory": "grid"}],
        "images": image_records,
        "annotations": annotation_records,
    }
 
 
# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------
 
def process_image(img_path: Path,
                  image_id: int,
                  cells_dir: Path,
                  debug_dir: Path | None,
                  annotation_records: list[dict],
                  ann_id: list[int],
                  h_kernel_ratio: float = 0.25,
                  v_kernel_ratio: float = 0.25) -> dict | None:
 
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  [WARN] Cannot read {img_path.name}, skipping.")
        return None
 
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h_lines, v_lines = detect_grid_lines(img_gray, h_kernel_ratio, v_kernel_ratio)
 
    if len(h_lines) < 2 or len(v_lines) < 2:
        print(f"  [WARN] Too few grid lines in {img_path.name} "
              f"(h={len(h_lines)}, v={len(v_lines)}). "
              f"Try lowering --h-kernel or --v-kernel.")
        return None
 
    cells = extract_cells(h_lines, v_lines)
    print(f"  {img_path.name}: {len(h_lines)} H-lines, "
          f"{len(v_lines)} V-lines -> {len(cells)} cells")
 
    for cell in cells:
        crop = crop_cell(img_bgr, cell)
        if crop.size == 0:
            continue
        fname = f"{img_path.stem}_r{cell['row']:02d}_c{cell['col']:02d}.png"
        cv2.imwrite(str(cells_dir / fname), crop)
 
        annotation_records.append({
            "id":          ann_id[0],
            "image_id":    image_id,
            "category_id": 1,
            "bbox":        [cell['x'], cell['y'], cell['w'], cell['h']],
            "area":        cell['w'] * cell['h'],
            "iscrowd":     0,
        })
        ann_id[0] += 1
 
    if debug_dir is not None:
        debug_img = draw_debug(img_bgr, h_lines, v_lines, cells)
        cv2.imwrite(str(debug_dir / img_path.name), debug_img)
 
    h_img, w_img = img_bgr.shape[:2]
    return {"id": image_id, "file_name": img_path.name,
            "width": w_img, "height": h_img}
 
 
# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
 
def run(input_dir: str,
        output_dir: str,
        debug: bool = False,
        h_kernel: float = 0.25,
        v_kernel: float = 0.25,
        extensions: tuple[str, ...] = ('.png', '.jpg', '.jpeg')) -> None:
 
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
 
    cells_dir  = out_path / "cells"
    images_dir = out_path / "images"
    debug_dir  = out_path / "debug" if debug else None
 
    cells_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
 
    img_files = sorted(p for p in in_path.iterdir()
                       if p.suffix.lower() in extensions)
    if not img_files:
        print(f"No images found in {input_dir}")
        return
 
    image_records:      list[dict] = []
    annotation_records: list[dict] = []
    ann_id = [0]
 
    for image_id, img_path in enumerate(img_files):
        print(f"[{image_id+1}/{len(img_files)}] {img_path.name}")
        record = process_image(img_path, image_id,
                               cells_dir, debug_dir,
                               annotation_records, ann_id,
                               h_kernel, v_kernel)
        if record is None:
            continue
        image_records.append(record)
        shutil.copy(img_path, images_dir / img_path.name)
 
    coco     = build_coco(image_records, annotation_records)
    ann_path = out_path / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f, indent=2)
 
    print(f"\nDone. {len(image_records)} images, "
          f"{len(annotation_records)} cell annotations.")
    print(f"Annotations : {ann_path}")
    print(f"Cell crops  : {cells_dir}")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract golf scorecard grid cells via morphological line "
                    "detection and produce COCO training data for an RCNN.")
    parser.add_argument("--input",    required=True,
                        help="Folder of raw scorecard PNG/JPG images")
    parser.add_argument("--output",   required=True,
                        help="Output folder for cells + annotations")
    parser.add_argument("--debug",    action="store_true",
                        help="Save debug images with detected grid overlaid")
    parser.add_argument("--h-kernel", type=float, default=0.25, dest="h_kernel",
                        help="Horizontal morphology kernel as fraction of image "
                             "width (default 0.25). Decrease if real lines are "
                             "missed; increase if short strokes are falsely picked up.")
    parser.add_argument("--v-kernel", type=float, default=0.25, dest="v_kernel",
                        help="Vertical morphology kernel as fraction of image "
                             "height (default 0.25).")
    args = parser.parse_args()
 
    run(args.input, args.output,
        debug=args.debug,
        h_kernel=args.h_kernel,
        v_kernel=args.v_kernel)
 
 
if __name__ == "__main__":
    main()

