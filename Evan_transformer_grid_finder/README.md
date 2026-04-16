# Golf Scorecard Grid Extraction (Transformer)

This module detects golf scorecard table grids and exports OCR-ready cell PNGs while preserving table structure.  
It uses a color-aware Pyramid Vision Transformer segmentation model plus a geometric decoder that snaps lines to evidence, handles merged cells, and outputs cells in row-major matrix order (`[0,0]` = top-left).

## Included Files (Minimal Runnable Set)
- `scorecard_transformer_extraction.py` (train/infer/batch pipeline)
- `scorecard_segmentation_extraction.py` (decoder + cell extraction)
- `scorecard_preprocessing.py` (upright correction, denoise, line-friendly preprocessing, PDF->PNG helper)
- `scorecard_row_separator_label_tool.py` (annotation/export tool for row separators)
- `scorecard_corner_label_tool.py` (helper used by LOO script)
- `run_leave_one_out.py` (LOO evaluation runner)
- `checkpoints/scorecard_transformer_rowsep.pt` (trained weights)

## Environment Setup
```bash
cd "Evan_transformer_grid_finder"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- For OCR-based upright checks, install system `tesseract`.
- For PDF conversion, install either Python `pdf2image` (already in requirements) plus Poppler, or just Poppler's `pdftoppm`.

## Run Inference (Single Image)
```bash
python scorecard_transformer_extraction.py infer \
  --weights checkpoints/scorecard_transformer_rowsep.pt \
  --input "../ScoreCards/CleanScans/clean1.png" \
  --output_dir "../ScoreCards/one_image_output" \
  --device auto \
  --min_keep_cols 9
```

## Run Inference (Batch PNG Folder)
```bash
python scorecard_transformer_extraction.py batch \
  --weights checkpoints/scorecard_transformer_rowsep.pt \
  --input_dir "../ScoreCards/CleanScans" \
  --output_dir "../ScoreCards/batch_cells_latest_png" \
  --device auto \
  --min_keep_cols 9
```

## Train / Fine-Tune on Exported Labels
Assumes exported labels are in `../ScoreCards/CleanScansAnnotated` with:
- JSON files at root
- `images/` and `masks/` subfolders

```bash
python scorecard_transformer_extraction.py train \
  --labels_dir "../ScoreCards/CleanScansAnnotated" \
  --out_weights checkpoints/scorecard_transformer_rowsep.pt \
  --device auto \
  --epochs 60 \
  --batch_size 2 \
  --train_size 1024 \
  --val_ratio 0.12
```

## Optional: Add Unlabeled Pretraining
```bash
python scorecard_transformer_extraction.py train \
  --labels_dir "../ScoreCards/CleanScansAnnotated" \
  --out_weights checkpoints/scorecard_transformer_rowsep.pt \
  --device auto \
  --pretrain_images_dir "../ScoreCards/scorecards-online" \
  --pretrain_epochs 20 \
  --pretrain_batch_size 8 \
  --pretrain_max_images 0
```

## Preprocessing Debug (Single Image)
```bash
python scorecard_preprocessing.py \
  --input "../ScoreCards/CleanScans/clean1.png" \
  --debug_dir "./preprocess_debug"
```
