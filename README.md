# Scorecard-CV

A machine learning pipeline for automated digitization of golf scorecards.
Given an image of a golf scorecard, the system extracts structured scoring data
including player scores, hole information, and row labels.

## Pipeline Overview

The system is a two-stage pipeline:

1. **Grid Detection** (handled by partners) — detects the scorecard table structure
   and produces cropped cell images as PNG files.
2. **OCR Engine** (this module) — reads the content of each cell using a combination
   of a CNN digit classifier and TrOCR transformer models.

### Cell Routing

Each cell is routed to the appropriate model based on its predicted content type:

| Cell Type       | Model Used                        | Accuracy |
|----------------|-----------------------------------|----------|
| Handwritten scores | CNN digit classifier (finetuned) | 76.9%   |
| Printed numbers (yardage) | TrOCR printed model      | 86.7%   |
| Row labels (Par, Handicap, etc.) | TrOCR + fuzzy vocab match | 100.0% |
| Empty cells     | Pixel intensity threshold         | 100.0%  |

Overall pipeline accuracy: **77.2%** across 57 labeled cells from 2 real scorecards.

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/evanfunk4/Scorecard-CV.git
cd Scorecard-CV
```

### 2. Create environment
```bash
conda create -n scorecard_ocr python=3.10
conda activate scorecard_ocr
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Usage

### Train the CNN digit classifier
Run this first before evaluating. Downloads MNIST automatically (~11MB),
pretrains for 5 epochs, then fine-tunes on labeled scorecard cells.
```bash
python cnn_digit_classifier.py
```
Trained models are saved to `models/`.

### Run full evaluation (three-way comparison)
Compares TrOCR, CNN, and Hybrid (CNN + TrOCR fallback) across all labeled cells.
```bash
python batch_test.py
```

### Test OCR on a single cell image
```bash
python ocr_engine.py
```
Edit the path at the bottom of `ocr_engine.py` to point at any cell PNG.

### Convert CleanScan PDFs to images
```bash
python pdf_to_image.py
```
Reads from `CleanScans/` and saves PNGs to `data/scans/`.

### Crop cells from a scorecard image
Interactive tool — draw boxes around cells, press `s` to save with a label,
press `q` to quit.
```bash
python crop_tool.py data/scorecard1.png
```
Saved crops go to `data/cells/` with filename format `<label>_<N>.png`.

---

## Project Structure

```
Scorecard-CV/
├── ocr_engine.py            # TrOCR-based OCR with preprocessing pipeline
├── cnn_digit_classifier.py  # CNN digit classifier with MNIST pretraining
├── batch_test.py            # Three-way model comparison and evaluation
├── crop_tool.py             # Interactive cell cropping tool
├── pdf_to_image.py          # PDF to PNG converter for CleanScans
├── requirements.txt
├── models/
│   ├── cnn_mnist.pth        # CNN pretrained on MNIST
│   └── cnn_finetuned.pth    # CNN fine-tuned on scorecard cells
├── data/
│   ├── cells/               # Labeled cell crops for evaluation
│   ├── scans/               # PNG exports from CleanScan PDFs
│   ├── scorecard1.png       # Real scorecard with handwritten scores
│   └── scorecard2.webp      # Real scorecard with handwritten scores
└── CleanScans/              # Clean blank scorecard PDFs (no handwriting)
```

---

## Evaluation Metrics

- **Accuracy** — exact match after digit normalization
- **CER (Character Error Rate)** — Levenshtein edit distance / true label length.
  Standard metric for OCR evaluation. Lower is better (0.0 = perfect).

---

## Known Limitations

- Circled/annotated digits (birdie circles, squares) significantly reduce accuracy
- Very faint pencil marks may be misclassified as empty cells
- Names are not reliably readable — players are identified by column order as fallback
- CNN is trained on single digits only; multi-digit scores (e.g. 10, 11) route to TrOCR

## The Team
- **Grid Detection:** Evan & Joseph
- **Handwriting OCR:** Deniz

---

## Grid Detection Module (Transformer)

The following section is included from `Evan_transformer_grid_finder/README.md`.

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
