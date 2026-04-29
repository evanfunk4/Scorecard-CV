"""
evaluate_and_reconstruct.py
===========================
1. Loads RCNN-predicted cell bounding boxes (from predict_cells.py output or
   a COCO annotations.json).
2. Averages the cell edges per row/column to find clean, straight grid lines.
3. Stitches the predicted cells back into a full scorecard PNG.
4. Loads your hand-annotated horizontal and vertical line masks.
5. Extracts ground-truth line positions from those masks.
6. Reports pixel distance between predicted and ground-truth lines.
 
Usage
-----
    python evaluate_and_reconstruct.py \
        --image       original_scorecards/card.png \
        --annotations training_data/annotations.json \
        --h-mask      masks/card_horizontal_mask.png \
        --v-mask      masks/card_vertical_mask.png \
        --output      eval_output/
 
Outputs (in --output folder)
-----------------------------
    card_reconstructed.png   <- cells stitched back together
    card_debug.png           <- original with predicted (blue) and
                                ground-truth (red) lines overlaid
    card_accuracy.json       <- per-line pixel distances + summary stats
    card_accuracy.csv        <- same data, spreadsheet-friendly
"""
 
import argparse
import csv
import json
import os
from pathlib import Path
 
import cv2
import numpy as np
from PIL import Image
 
 
# ---------------------------------------------------------------------------
# Line extraction from mask images
# ---------------------------------------------------------------------------
 
def mask_to_line_positions(mask_path: str, axis: str,
                            cluster_gap: int = 8,
                            target_height: int = None,
                            target_width: int = None) -> list[int]:
    """
    Read a binary (or near-binary) mask PNG and return line positions.
 
    axis='h' -> project columns, return sorted y-coordinates
    axis='v' -> project rows,    return sorted x-coordinates
 
    If target_height/target_width are provided, resize the mask to match
    the target dimensions before extracting lines.
    """
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
 
    # Resize mask if target dimensions are provided
    if target_height is not None and target_width is not None:
        img = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
 
    # Binarise: any non-black pixel is a line pixel
    _, binary = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
 
    if axis == 'h':
        projection = np.sum(binary, axis=1).astype(np.float32)
    else:
        projection = np.sum(binary, axis=0).astype(np.float32)
 
    threshold = projection.max() * 0.25
    candidates = np.where(projection > threshold)[0].tolist()
    return _cluster(candidates, gap=cluster_gap)
 
 
def colour_overlay_to_masks(overlay_path: str,
                             h_colour_bgr: tuple = (180, 210, 210),
                             v_colour_bgr: tuple = (0, 0, 200),
                             tolerance: int = 40
                             ) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a combined colour overlay (red=vertical, beige=horizontal) into
    two separate binary masks.  Only needed if you're using the combined PNG
    instead of separate mask files.
 
    Returns (h_mask, v_mask) as uint8 arrays.
    """
    img = cv2.imread(overlay_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read overlay: {overlay_path}")
 
    def colour_mask(bgr_target):
        lower = np.array([max(0, c - tolerance) for c in bgr_target], np.uint8)
        upper = np.array([min(255, c + tolerance) for c in bgr_target], np.uint8)
        return cv2.inRange(img, lower, upper)
 
    return colour_mask(h_colour_bgr), colour_mask(v_colour_bgr)
 
 
def _cluster(coords: list[int], gap: int) -> list[int]:
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
 
 
# ---------------------------------------------------------------------------
# Extract predicted grid lines from RCNN cell bboxes
# ---------------------------------------------------------------------------
 
def load_predictions_for_image(annotations_path: str,
                                image_filename: str) -> list[dict]:
    """
    Load all cell bboxes for a given image from a COCO annotations.json.
    Returns list of { x, y, w, h } dicts.
    """
    with open(annotations_path) as f:
        coco = json.load(f)
 
    # Find image id
    img_id = None
    for img in coco["images"]:
        if img["file_name"] == image_filename:
            img_id = img["id"]
            break
    if img_id is None:
        raise ValueError(f"Image '{image_filename}' not found in {annotations_path}")
 
    cells = []
    for ann in coco["annotations"]:
        if ann["image_id"] == img_id:
            x, y, w, h = ann["bbox"]
            cells.append({"x": int(x), "y": int(y),
                          "w": int(w), "h": int(h)})
    return cells
 
 
def cells_to_grid_lines(cells: list[dict],
                         cluster_gap: int = 10
                         ) -> tuple[list[int], list[int]]:
    """
    Derive clean grid line positions from cell bounding boxes by:
      1. Collecting all top/bottom edges (for H-lines) and
         left/right edges (for V-lines).
      2. Clustering nearby edges together.
      3. Averaging each cluster to get one representative line position.
 
    This smooths out any per-cell jitter from the RCNN detector.
    """
    h_candidates = []
    v_candidates = []
 
    for c in cells:
        h_candidates.append(c["y"])           # top edge
        h_candidates.append(c["y"] + c["h"])  # bottom edge
        v_candidates.append(c["x"])           # left edge
        v_candidates.append(c["x"] + c["w"])  # right edge
 
    h_lines = _cluster(sorted(h_candidates), gap=cluster_gap)
    v_lines = _cluster(sorted(v_candidates), gap=cluster_gap)
    return h_lines, v_lines
 
 
# ---------------------------------------------------------------------------
# Image reconstruction
# ---------------------------------------------------------------------------
 
def reconstruct_from_cells(source_img: np.ndarray,
                            h_lines: list[int],
                            v_lines: list[int],
                            padding: int = 3) -> np.ndarray:
    """
    Stitch cells back into a full scorecard PNG using the averaged grid lines.
 
    Each cell is cropped from the original image using the clean averaged line
    positions (not the raw RCNN boxes), so any per-cell misalignment is
    corrected.  The result is written into a blank canvas of the same size as
    the bounding box of the grid.
    """
    if len(h_lines) < 2 or len(v_lines) < 2:
        raise ValueError("Need at least 2 H-lines and 2 V-lines to reconstruct.")
 
    grid_h = h_lines[-1] - h_lines[0]
    grid_w = v_lines[-1] - v_lines[0]
    canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 255
 
    img_h, img_w = source_img.shape[:2]
 
    for r, (y0, y1) in enumerate(zip(h_lines, h_lines[1:])):
        for c, (x0, x1) in enumerate(zip(v_lines, v_lines[1:])):
            # Crop from source using averaged lines + inward padding
            src_x0 = min(img_w - 1, x0 + padding)
            src_y0 = min(img_h - 1, y0 + padding)
            src_x1 = max(0, x1 - padding)
            src_y1 = max(0, y1 - padding)
 
            crop = source_img[src_y0:src_y1, src_x0:src_x1]
            if crop.size == 0:
                continue
 
            # Place onto canvas
            dst_x0 = x0 - v_lines[0]
            dst_y0 = y0 - h_lines[0]
            dst_x1 = dst_x0 + (src_x1 - src_x0)
            dst_y1 = dst_y0 + (src_y1 - src_y0)
 
            dst_x1 = min(grid_w, dst_x1)
            dst_y1 = min(grid_h, dst_y1)
 
            canvas[dst_y0:dst_y1, dst_x0:dst_x1] = \
                crop[:dst_y1 - dst_y0, :dst_x1 - dst_x0]
 
    return canvas
 
 
# ---------------------------------------------------------------------------
# Accuracy measurement
# ---------------------------------------------------------------------------
 
def match_lines(predicted: list[int],
                ground_truth: list[int]) -> list[dict]:
    """
    Greedy nearest-neighbour matching of predicted lines to ground-truth lines.
 
    Each ground-truth line is matched to its closest predicted line.
    Returns list of dicts with keys: gt, pred, distance.
    """
    results = []
    used = set()
    for gt in ground_truth:
        best_pred = None
        best_dist = float("inf")
        for i, pred in enumerate(predicted):
            if i in used:
                continue
            d = abs(pred - gt)
            if d < best_dist:
                best_dist = d
                best_pred = (i, pred)
        if best_pred is not None:
            used.add(best_pred[0])
            results.append({"gt": gt, "pred": best_pred[1],
                            "distance": best_dist})
        else:
            results.append({"gt": gt, "pred": None, "distance": None})
    return results
 
 
def compute_accuracy(pred_h: list[int], pred_v: list[int],
                     gt_h: list[int],   gt_v: list[int]
                     ) -> dict:
    """
    Compare predicted vs ground-truth lines and return summary statistics.
    """
    h_matches = match_lines(pred_h, gt_h)
    v_matches = match_lines(pred_v, gt_v)
 
    def stats(matches):
        dists = [m["distance"] for m in matches if m["distance"] is not None]
        if not dists:
            return {"mean_px": None, "max_px": None, "median_px": None,
                    "matched": 0, "unmatched": 0}
        return {
            "mean_px":   round(float(np.mean(dists)),   2),
            "max_px":    round(float(np.max(dists)),    2),
            "median_px": round(float(np.median(dists)), 2),
            "matched":   len(dists),
            "unmatched": sum(1 for m in matches if m["distance"] is None),
        }
 
    return {
        "horizontal": {"stats": stats(h_matches), "matches": h_matches},
        "vertical":   {"stats": stats(v_matches), "matches": v_matches},
    }
 
 
# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------
 
def draw_comparison(img: np.ndarray,
                    pred_h: list[int], pred_v: list[int],
                    gt_h:   list[int], gt_v:   list[int]) -> np.ndarray:
    """
    Draw predicted lines (blue) and ground-truth lines (red) on a copy of img.
    Rotate the image 90° counterclockwise, but keep GT lines in their original
    horizontal/vertical orientation.
    """
    # Rotate the background image
    out = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h_img, w_img = out.shape[:2]
    orig_h, orig_w = img.shape[:2]

    # Transform ground truth lines into rotated image coordinates.
    gt_h_rotated = [orig_w - x - 1 for x in gt_v]
    gt_v_rotated = [y for y in gt_h]

    # Draw transformed ground truth lines on the rotated image.
    for y in gt_h_rotated:
        if 0 <= y < h_img:
            cv2.line(out, (0, y), (w_img, y), (0, 0, 220), 4)   # red  = GT
    for x in gt_v_rotated:
        if 0 <= x < w_img:
            cv2.line(out, (x, 0), (x, h_img), (0, 0, 220), 4)

    # Transform predicted lines into rotated image coordinates.
    pred_h_rotated = [y for y in pred_h]
    pred_v_rotated = [orig_w - x - 1 for x in pred_v]

    for y in pred_v_rotated:
        if 0 <= y < h_img:
            cv2.line(out, (0, y), (w_img, y), (220, 100, 0), 3)  # blue = predicted
    for x in pred_h_rotated:
        if 0 <= x < w_img:
            cv2.line(out, (x, 0), (x, h_img), (220, 100, 0), 3)

    # Legend
    cv2.rectangle(out, (8, 8), (220, 60), (255, 255, 255), -1)
    cv2.putText(out, "Red  = ground truth",  (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 200),   1)
    cv2.putText(out, "Blue = predicted",     (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 100, 0), 1)
    return out
 
 
# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
 
def run(image_path: str,
        annotations_path: str,
        h_mask_path: str,
        v_mask_path: str,
        output_dir: str) -> None:
 
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
 
    print(f"Loading image: {image_path}")
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
 
    # ---- 1. Load predicted cells and derive averaged grid lines ----
    print("Loading RCNN predictions ...")
    image_filename = Path(image_path).name
    cells = load_predictions_for_image(annotations_path, image_filename)
    if not cells:
        print("[WARN] No cells found for this image in annotations.json.")
        return
    print(f"  {len(cells)} predicted cells")
 
    pred_h, pred_v = cells_to_grid_lines(cells)
    print(f"  Averaged to {len(pred_h)} H-lines, {len(pred_v)} V-lines")
 
    # ---- 2. Reconstruct scorecard from averaged predicted lines ----
    print("Reconstructing scorecard ...")
    reconstructed = reconstruct_from_cells(img_bgr, pred_h, pred_v)
    recon_path = out / f"{stem}_reconstructed.png"
    cv2.imwrite(str(recon_path), reconstructed)
    print(f"  Saved: {recon_path}")
 
    # ---- 3. Load ground-truth lines from masks for the reconstructed grid ----
    print("Extracting ground-truth lines from masks ...")
    grid_h, grid_w = reconstructed.shape[:2]
    gt_h = mask_to_line_positions(h_mask_path, axis='h',
                                  target_height=grid_h,
                                  target_width=grid_w)
    gt_v = mask_to_line_positions(v_mask_path, axis='v',
                                  target_height=grid_h,
                                  target_width=grid_w)
    print(f"  Ground truth: {len(gt_h)} H-lines, {len(gt_v)} V-lines")
 
    # Convert predicted lines to reconstructed-grid coordinates
    pred_h_local = [y - pred_h[0] for y in pred_h]
    pred_v_local = [x - pred_v[0] for x in pred_v]
 
    # ---- 4. Accuracy measurement ----
    print("Computing accuracy ...")
    accuracy = compute_accuracy(pred_h_local, pred_v_local, gt_h, gt_v)
 
    h_stats = accuracy["horizontal"]["stats"]
    v_stats = accuracy["vertical"]["stats"]
    print(f"  Horizontal lines — mean: {h_stats['mean_px']}px  "
          f"max: {h_stats['max_px']}px  median: {h_stats['median_px']}px  "
          f"matched: {h_stats['matched']}/{len(gt_h)}")
    print(f"  Vertical lines   — mean: {v_stats['mean_px']}px  "
          f"max: {v_stats['max_px']}px  median: {v_stats['median_px']}px  "
          f"matched: {v_stats['matched']}/{len(gt_v)}")
 
    # Save JSON
    json_path = out / f"{stem}_accuracy.json"
    with open(json_path, "w") as f:
        json.dump(accuracy, f, indent=2)
    print(f"  Saved: {json_path}")
 
    # Save CSV
    csv_path = out / f"{stem}_accuracy.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["axis", "ground_truth_px", "predicted_px", "distance_px"])
        for m in accuracy["horizontal"]["matches"]:
            writer.writerow(["horizontal", m["gt"], m["pred"], m["distance"]])
        for m in accuracy["vertical"]["matches"]:
            writer.writerow(["vertical", m["gt"], m["pred"], m["distance"]])
    print(f"  Saved: {csv_path}")
 
    # ---- 5. Debug overlay ----
    debug_img = draw_comparison(reconstructed, pred_h_local, pred_v_local, gt_h, gt_v)
    debug_path = out / f"{stem}_debug.png"
    cv2.imwrite(str(debug_path), debug_img)
    print(f"  Saved: {debug_path}")
 
    print("\nDone.")
 
 
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct scorecard from RCNN cells and evaluate "
                    "accuracy against hand-annotated line masks.")
    parser.add_argument("--image",       required=True,
                        help="Original scorecard PNG")
    parser.add_argument("--annotations", required=True,
                        help="COCO annotations.json from hough_lines.py or "
                             "predict_cells.py")
    parser.add_argument("--h-mask",      required=True, dest="h_mask",
                        help="PNG mask of horizontal grid lines")
    parser.add_argument("--v-mask",      required=True, dest="v_mask",
                        help="PNG mask of vertical grid lines")
    parser.add_argument("--output",      default="eval_output/",
                        help="Folder for reconstructed image + accuracy report")
    args = parser.parse_args()
 
    run(args.image, args.annotations, args.h_mask, args.v_mask, args.output)
 
 
if __name__ == "__main__":
    main()