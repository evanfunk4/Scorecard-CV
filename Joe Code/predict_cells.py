"""
predict_cells.py  —  Run trained Faster R-CNN on new scorecards,
                      save each detected cell as a separate PNG.
 
Usage:
    python predict_cells.py \
        --model  scorecard_rcnn.pth \
        --input  new_scorecards/ \
        --output outputCells/ \
        --score  0.5
 
Output layout:
    output_cells/
        <image_stem>/
            cell_0001.png
            cell_0002.png
            ...
        <image_stem>_debug.png   <- full image with boxes drawn (optional)
"""
 
import argparse
from pathlib import Path
from xml.parsers.expat import model
 
import cv2
import numpy as np
import torch
from PIL import Image
import torchvision
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
 
 
# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
 
def load_model(weights_path: str, device: torch.device) -> torch.nn.Module:
    model = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    model.rpn.post_nms_top_n_test = 2000
    model.roi_heads.score_thresh = 0.3
    model.roi_heads.nms_thresh = 0.3 
    model.roi_heads.detections_per_img = 750

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
 
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
 
 
# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
 
def predict(model: torch.nn.Module,
            img_tensor: torch.Tensor,
            device: torch.device,
            score_thresh: float) -> list[list[int]]:
    """
    Returns list of [x1, y1, x2, y2] bboxes above score_thresh,
    sorted top-left to bottom-right (row-major order).
    """
    all_boxes, all_scores = [], []
    
    for scale in [1.0, 1.5]:
        if scale != 1.0:
            h, w = img_tensor.shape[1:]
            scaled = TF.resize(img_tensor, [int(h*scale), int(w*scale)])
        else:
            scaled = img_tensor
        
        with torch.no_grad():
            out = model([scaled.to(device)])[0]
        
        boxes = out["boxes"].cpu().numpy() / scale
        scores = out["scores"].cpu().numpy()
        keep = scores >= score_thresh
        all_boxes.extend(boxes[keep].tolist())
        all_scores.extend(scores[keep].tolist())
    
    if not all_boxes:
        return []
    
    boxes_t = torch.tensor(all_boxes, dtype=torch.float32)
    scores_t = torch.tensor(all_scores, dtype=torch.float32)
    keep = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.4)
    result = boxes_t[keep].int().tolist()
    result.sort(key=lambda b: (b[1], b[0]))
    return result
 
 
def crop_box(img_np: np.ndarray, box: list[int], padding: int = 0) -> np.ndarray:
    h, w = img_np.shape[:2]
    x1 = max(0, box[0] - padding)
    y1 = max(0, box[1] - padding)
    x2 = min(w, box[2] + padding)
    y2 = min(h, box[3] + padding)
    return img_np[y1:y2, x1:x2]
 
 
def draw_boxes(img_np: np.ndarray, boxes: list[list[int]]) -> np.ndarray:
    out = img_np.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), 2)
        cv2.putText(out, str(i + 1), (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 220), 1)
    return out
 
 
# ---------------------------------------------------------------------------
# Per-image pipeline
# ---------------------------------------------------------------------------
 
def process_image(img_path: Path,
                  model: torch.nn.Module,
                  device: torch.device,
                  out_dir: Path,
                  score_thresh: float,
                  debug: bool) -> int:
    img_pil = Image.open(img_path).convert("RGB")
    img_tensor = TF.to_tensor(img_pil)
    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
 
    boxes = predict(model, img_tensor, device, score_thresh)
    if not boxes:
        print(f"  [WARN] No cells detected in {img_path.name}")
        return 0
 
    cell_dir = out_dir / img_path.stem
    cell_dir.mkdir(parents=True, exist_ok=True)
 
    for i, box in enumerate(boxes):
        crop = crop_box(img_bgr, box)
        if crop.size == 0:
            continue
        cell_path = cell_dir / f"cell_{i+1:04d}.png"
        cv2.imwrite(str(cell_path), crop)
 
    if debug:
        debug_img = draw_boxes(img_bgr, boxes)
        cv2.imwrite(str(out_dir / f"{img_path.stem}_debug.png"), debug_img)
 
    return len(boxes)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def run(model_path: str,
        input_dir: str,
        output_dir: str,
        score_thresh: float = 0.5,
        debug: bool = False,
        extensions: tuple[str, ...] = ('.png', '.jpg', '.jpeg')) -> None:
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Loading model from {model_path} ...")
    model = load_model(model_path, device)
 
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
 
    img_files = [p for p in sorted(in_path.iterdir())
                 if p.suffix.lower() in extensions]
 
    if not img_files:
        print(f"No images found in {input_dir}")
        return
 
    total_cells = 0
    for i, img_path in enumerate(img_files):
        print(f"[{i+1}/{len(img_files)}] {img_path.name}")
        n = process_image(img_path, model, device, out_path,
                          score_thresh, debug)
        print(f"  -> {n} cells saved to {out_path / img_path.stem}/")
        total_cells += n
 
    print(f"\nDone. {total_cells} total cells saved to {output_dir}")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained RCNN on scorecards; save each cell as PNG.")
    parser.add_argument("--model", required=True,
                        help="Path to scorecard_rcnn.pth weights file")
    parser.add_argument("--input", required=True,
                        help="Directory of new scorecard images")
    parser.add_argument("--output", default="output_cells/",
                        help="Directory to save cropped cell PNGs")
    parser.add_argument("--score", type=float, default=0.3,
                        help="Confidence threshold (default 0.5)")
    parser.add_argument("--debug", action="store_true",
                        help="Save full images with detected boxes drawn")
    args = parser.parse_args()
 
    run(args.model, args.input, args.output,
        score_thresh=args.score, debug=args.debug)
 
 
if __name__ == "__main__":
    main()