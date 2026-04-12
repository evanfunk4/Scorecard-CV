"""
train_rcnn.py  —  Fine-tune Faster R-CNN on golf scorecard grid cells
 
Usage:
    python train_rcnn.py \
        --data  training_data/ \
        --epochs 20 \
        --output scorecard_rcnn.pth
 
Expects training_data/ produced by hough_lines.py:
    training_data/
        images/           <- full scorecard PNGs
        annotations.json  <- COCO-format bbox annotations (class: "cell")
"""
 
import argparse
import json
import os
from pathlib import Path
 
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as TF
 
 
# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
 
class ScorecardCellDataset(Dataset):
    """
    COCO-style dataset for scorecard grid-cell detection.
 
    One sample = one full scorecard image with all its cell bboxes.
    """
 
    def __init__(self, data_dir: str) -> None:
        data_path = Path(data_dir)
        ann_path = data_path / "annotations.json"
        img_dir = data_path / "images"
 
        with open(ann_path) as f:
            coco = json.load(f)
 
        self.img_dir = img_dir
        self.images = coco["images"]
 
        # Build image_id → list of annotations lookup
        self._ann_by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            self._ann_by_image.setdefault(ann["image_id"], []).append(ann)
 
    def __len__(self) -> int:
        return len(self.images)
 
    def __getitem__(self, idx: int, max_side: int = 1333):
        img_meta = self.images[idx]
        img_id = img_meta["id"]
 
        img = Image.open(self.img_dir / img_meta["file_name"]).convert("RGB")
 
        # Resize large images so the longest side <= max_side (default 1333px).
        # This matches Faster R-CNN's built-in resize and prevents OOM errors
        # on high-resolution scans. Bboxes are scaled by the same ratio.
        orig_w, orig_h = img.size
        scale = min(max_side / max(orig_w, orig_h), 1.0)
        if scale < 1.0:
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            img = img.resize((new_w, new_h), Image.BILINEAR)
        else:
            scale = 1.0
 
        img_tensor = TF.to_tensor(img)  # [C, H, W] float in [0, 1]
 
        anns = self._ann_by_image.get(img_id, [])
 
        if anns:
            # COCO bbox is [x, y, w, h] → scale → convert to [x1, y1, x2, y2]
            boxes = torch.tensor(
                [[(a["bbox"][0])          * scale,
                  (a["bbox"][1])          * scale,
                  (a["bbox"][0] + a["bbox"][2]) * scale,
                  (a["bbox"][1] + a["bbox"][3]) * scale] for a in anns],
                dtype=torch.float32,
            )
            labels = torch.ones(len(anns), dtype=torch.int64)  # class 1 = "cell"
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
 
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([img_id]),
            "area": torch.tensor([a["area"] * scale * scale for a in anns],
                                 dtype=torch.float32)
                    if anns else torch.zeros(0),
            "iscrowd": torch.zeros(len(anns), dtype=torch.int64),
        }
        return img_tensor, target
 
 
def collate_fn(batch):
    return tuple(zip(*batch))
 
 
# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
 
def build_model(num_classes: int = 2) -> torch.nn.Module:
    """
    Load pretrained Faster R-CNN (ResNet-50 + FPN) and replace the
    box predictor head with one for num_classes (background + cell).
    """
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
 
 
# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
 
def train_one_epoch(model, optimizer, data_loader, device, epoch: int) -> float:
    model.train()
    total_loss = 0.0
 
    for step, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
 
        loss_dict = model(images, targets)
        losses = sum(loss_dict.values())
 
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
 
        total_loss += losses.item()
        if (step + 1) % 10 == 0 or step == 0:
            print(f"  Epoch {epoch}  step {step+1}/{len(data_loader)}"
                  f"  loss={losses.item():.4f}")
 
    return total_loss / len(data_loader)
 
 
# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
 
def run(data_dir: str,
        epochs: int,
        output_path: str,
        batch_size: int = 2,
        lr: float = 5e-4) -> None:
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
 
    dataset = ScorecardCellDataset(data_dir)
    print(f"Dataset: {len(dataset)} images loaded from {data_dir}")
 
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if os.name == "nt" else min(4, os.cpu_count() or 1),
        collate_fn=collate_fn,
    )
 
    model = build_model(num_classes=2)  # 0=background, 1=cell
    model.to(device)
 
    # Only fine-tune the head + last backbone block
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 3), gamma=0.5
    )
 
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        avg_loss = train_one_epoch(model, optimizer, loader, device, epoch)
        lr_scheduler.step()
        print(f"Epoch {epoch}/{epochs}  avg_loss={avg_loss:.4f}")
 
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_path)
            print(f"  -> Saved best model to {output_path}")
 
    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {output_path}")
 
 
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Faster R-CNN on golf scorecard grid cells.")
    parser.add_argument("--data", required=True,
                        help="training_data/ directory from hough_lines.py")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output", default="scorecard_rcnn.pth",
                        help="Path to save trained model weights")
    args = parser.parse_args()
 
    run(args.data, args.epochs, args.output,
        batch_size=args.batch, lr=args.lr)
 
 
if __name__ == "__main__":
    main()