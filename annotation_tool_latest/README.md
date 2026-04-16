# Latest Annotation Tool (Row/Column Grid Editor)

This folder contains the latest interactive annotation tool used for scorecard grid supervision.

Tool file:
- `scorecard_row_separator_label_tool.py`

What it labels:
- Table bounding boxes (supports multiple tables per image)
- Vertical separator lines
- Horizontal separators with two classes:
  - `line` = drawn grid line
  - `boundary` = row boundary inferred by color/region change (no explicit dark stroke)

## Setup
From repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install opencv-python numpy
```

## Data Flow
1. (Optional) bootstrap row-separator labels from existing flex labels.
2. edit annotations interactively.
3. render training/export masks and JSON with relative paths.
4. generate overlays for QA.

## Commands
Run from repo root (`Scorecard-CV`):

### 1) Bootstrap from flex labels (optional)
```bash
python annotation_tool_latest/scorecard_row_separator_label_tool.py bootstrap \
  --flex_labels_dir "ScoreCards/CleanScansAnnotated" \
  --rowsep_labels_dir "ml_rowsep_labels" \
  --overwrite
```

### 2) Launch interactive editor
```bash
python annotation_tool_latest/scorecard_row_separator_label_tool.py edit \
  --rowsep_labels_dir "ml_rowsep_labels" \
  --start 0
```

### 3) Render exportable masks + portable JSON
```bash
python annotation_tool_latest/scorecard_row_separator_label_tool.py render \
  --rowsep_labels_dir "ml_rowsep_labels" \
  --out_labels_dir "ml_rowsep_labels_exported" \
  --line_thickness 3
```

### 4) Write overlay previews
```bash
python annotation_tool_latest/scorecard_row_separator_label_tool.py overlay \
  --rowsep_labels_dir "ml_rowsep_labels" \
  --out_dir "ml_rowsep_labels/overlays"
```

## Editor Controls
Modes:
- `1`: table mode
- `2`: horizontal separator mode
- `3`: vertical line mode

Navigation:
- `n` / `p`: next / previous image (auto-saves current)
- `[` / `]`: previous / next table (grid)
- `s`: save current
- `q`: save and quit

Table operations:
- `g`: add a new table (clone + offset)
- `r`: remove current table (if more than one)
- `m` (in table mode): toggle move-table mode, then drag inside table
- drag near corner: move table corners

Separator / vertical operations:
- `a`: toggle add mode, then click to add line
- `x`: toggle delete mode, then click near a line to delete
- drag near point: move one control point
- drag near line: move whole line

Separator-only operations (mode `2`):
- `t`: toggle separator kind (`line` <-> `boundary`) on click
- `b`: split a separator at clicked cell region

## Output Format (`render`)
`out_labels_dir` contains:
- `images/*.png`
- `masks/*_table.png`, `*_v.png`, `*_h.png`, `*_h_line.png`, `*_h_boundary.png`
- per-image JSON records with relative paths
- `index.jsonl`

The JSON paths are written as relative paths, so the export is portable for GitHub and other machines.
