"""
pipeline.py  —  Scorecard digitization pipeline.

Single scorecard
────────────────
    python pipeline.py --input ScoreCards/CleanScans/clean1.pdf

    GT auto-discovered from ScoreCards/CleanScansAnnotated/clean1.json
    Override: --ground_truth path/to/clean1.json

Batch (all scorecards in a folder)
────────────────────────────────────
    python pipeline.py --batch ScoreCards/CleanScans/

    Per-scorecard results → results/<stem>/results.json
    Aggregate summary    → results/summary.json + results/summary.txt

GT JSON format
──────────────
    Custom cells (full eval):
      { "cells": [{"x":10,"y":20,"w":50,"h":30,"text":"4"}, ...] }

    COCO with text (full eval):
      { "annotations": [{"bbox":[10,20,50,30],"text":"4"}, ...] }

    COCO bbox-only (IoU only, no OCR accuracy):
      { "annotations": [{"bbox":[10,20,50,30]}, ...] }

Root causes of previous bugs fixed here
─────────────────────────────────────────
    1. IoU was 0.0 for all cells:
       Evan's cells are named base_r01_c02.png — coordinates were never
       being read from matrix_index.json.  Now reads bbox_xyxy from that
       file directly.  Joe's RCNN now saves boxes.json alongside crops
       so real [x1,y1,x2,y2] coordinates are available for IoU.

    2. CNN == TrOCR accuracy (both 28.8%):
       The old interactive_labels.json had stale entries that capped
       immediately, so every cell used the same wrong label for both
       models.  Interactive labeling removed entirely — evaluation now
       uses GT text from the annotation JSON only.

    3. No interactive prompt shown:
       Same cap issue.  Removed; replaced by GT-JSON driven evaluation.

    4. Missing batch mode: added.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


# ── Colours ───────────────────────────────────────────────────────────────────

class C:
    RESET = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
    GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
    CYAN = "\033[96m"; BLUE = "\033[94m"; MAGENTA = "\033[95m"

def _c(text, *codes):
    return "".join(codes) + str(text) + C.RESET

def log(msg, level="info"):
    col = {"info":C.CYAN,"ok":C.GREEN,"warn":C.YELLOW,
           "error":C.RED,"h":C.BOLD}.get(level,"")
    pre = {"info":"  .","ok":"  v","warn":"  !",
           "error":"  x","h":"  >"}.get(level,"  .")
    print(f"{_c(pre,col)} {msg}", flush=True)

def _banner(t, w=74):
    print()
    print(_c("="*w, C.BOLD))
    print(_c(f"  {t}", C.BOLD))
    print(_c("="*w, C.BOLD))

def _section(t, w=74):
    print()
    print(_c("-"*w, C.DIM))
    print(_c(f"  {t}", C.BOLD+C.CYAN))
    print(_c("-"*w, C.DIM))

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)

def _bar(v, w=18):
    f = round(max(0, min(1, v)) * w)
    return "#"*f + "-"*(w-f)

def _pc(p):
    return C.GREEN if p >= 90 else (C.YELLOW if p >= 70 else C.RED)

def _cc(c):
    return C.GREEN if c <= 0.05 else (C.YELLOW if c <= 0.20 else C.RED)


# ── Ground truth ──────────────────────────────────────────────────────────────

class GTCell:
    def __init__(self, x, y, w, h, text="", row=-1, col=-1):
        self.x=int(x); self.y=int(y); self.w=int(w); self.h=int(h)
        self.text=str(text).strip(); self.row=int(row); self.col=int(col)

    @property
    def bbox(self):
        return (self.x, self.y, self.x+self.w, self.y+self.h)

    @property
    def has_text(self):
        return self.text != ""


def load_ground_truth(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cells = []
    if "cells" in data:
        for c in data["cells"]:
            text = str(c.get("text", c.get("value", c.get("label", "")))).strip()
            cells.append(GTCell(c["x"],c["y"],c["w"],c["h"],
                                text=text,row=c.get("row",-1),col=c.get("col",-1)))
    elif "annotations" in data:
        for ann in data["annotations"]:
            bx,by,bw,bh = ann["bbox"]
            text = (ann.get("text") or ann.get("value")
                    or (ann.get("attributes") or {}).get("text",""))
            cells.append(GTCell(bx,by,bw,bh,
                                text=str(text).strip() if text else ""))
    if not cells:
        log(f"No cells parsed from {Path(path).name} — check format", "warn")
    return cells


def _auto_gt(inp):
    stem = Path(inp).stem
    for cand in [
        Path(inp).parent.parent / "CleanScansAnnotated" / f"{stem}.json",
        Path(inp).parent / f"{stem}.json",
        Path("ScoreCards") / "CleanScansAnnotated" / f"{stem}.json",
        Path("CleanScansAnnotated") / f"{stem}.json",
    ]:
        if cand.exists():
            return cand
    return None


# ── IoU ───────────────────────────────────────────────────────────────────────

def _iou(a, b):
    ax0,ay0,ax1,ay1 = a
    bx0,by0,bx1,by1 = b
    iw = max(0, min(ax1,bx1) - max(ax0,bx0))
    ih = max(0, min(ay1,by1) - max(ay0,by0))
    inter = iw*ih
    if inter <= 0:
        return 0.0
    union = max(1,(ax1-ax0)*(ay1-ay0)) + max(1,(bx1-bx0)*(by1-by0)) - inter
    return inter / union


def iou_metrics(pred_boxes, gt_cells, thr=0.5):
    gt_boxes = [c.bbox for c in gt_cells]
    if not gt_boxes:
        return dict(precision=0.0,recall=0.0,f1=0.0,mean_iou=0.0,
                    tp=0,fp=len(pred_boxes),fn=0,
                    n_pred=len(pred_boxes),n_gt=0,matched_pairs=[])
    if not pred_boxes:
        return dict(precision=0.0,recall=0.0,f1=0.0,mean_iou=0.0,
                    tp=0,fp=0,fn=len(gt_boxes),
                    n_pred=0,n_gt=len(gt_boxes),matched_pairs=[])
    matched=set(); ivals=[]; pairs=[]; tp=0
    for pi,pred in enumerate(pred_boxes):
        bi,bv = 0.0,-1
        for gi,gt in enumerate(gt_boxes):
            if gi in matched: continue
            v = _iou(pred,gt)
            if v > bi: bi,bv = v,gi
        if bi >= thr and bv >= 0:
            tp+=1; matched.add(bv); ivals.append(bi); pairs.append((pi,bv,bi))
    fp=len(pred_boxes)-tp; fn=len(gt_boxes)-len(matched)
    pr=tp/max(1,tp+fp); rc=tp/max(1,tp+fn)
    f1=2*pr*rc/max(1e-9,pr+rc)
    return dict(precision=pr,recall=rc,f1=f1,
                mean_iou=float(np.mean(ivals)) if ivals else 0.0,
                tp=tp,fp=fp,fn=fn,
                n_pred=len(pred_boxes),n_gt=len(gt_boxes),
                matched_pairs=pairs)


# ── Bbox reading ──────────────────────────────────────────────────────────────
#
# This is the section that was broken before.
# Evan's extractor writes matrix_index.json with bbox_xyxy per cell.
# Joe's extractor (our adapter) now writes boxes.json with xyxy per cell.
# Without reading those, every box was (0,0,w,h) -> IoU = 0.

def _fallback_box(cell):
    """Last resort: parse filename or read image size."""
    n = Path(cell).stem
    m = re.search(r'x(\d+)_y(\d+)_w(\d+)_h(\d+)', n)
    if m:
        x,y,w,h = int(m[1]),int(m[2]),int(m[3]),int(m[4])
        return (x,y,x+w,y+h)
    m = re.search(r'x1(\d+)_y1(\d+)_x2(\d+)_y2(\d+)', n)
    if m:
        return (int(m[1]),int(m[2]),int(m[3]),int(m[4]))
    img = cv2.imread(str(cell))
    if img is not None:
        h,w = img.shape[:2]
        return (0,0,w,h)
    return (0,0,1,1)


def _evan_boxes(cells_dir, cells):
    """
    Read bbox_xyxy from matrix_index.json files written by Evan's extractor.
    Each table_XX/cells/ folder has a sibling matrix_index.json with
    base_cells[].bbox_xyxy = [x0,y0,x1,y1].
    """
    bbox_map = {}
    for idx_file in Path(cells_dir).rglob("matrix_index.json"):
        try:
            data = json.loads(idx_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        for rec in data.get("base_cells", []):
            pname = rec.get("path")
            bb = rec.get("bbox_xyxy")
            if pname and bb and len(bb) == 4:
                bbox_map[Path(pname).name] = tuple(int(v) for v in bb)
        for rec in data.get("cells", []):
            pname = rec.get("path")
            bb = rec.get("bbox_xyxy") or (rec.get("merged") or {}).get("bbox_xyxy")
            if pname and bb and len(bb) == 4:
                bbox_map[Path(pname).name] = tuple(int(v) for v in bb)

    boxes = []
    for cell in cells:
        name = Path(cell).name
        if name in bbox_map:
            boxes.append(bbox_map[name])
        else:
            # Try deriving from row/col pattern + x_lines/y_lines in nearest JSON
            m = re.search(r'base_r(\d+)_c(\d+)', Path(cell).stem)
            if m:
                r,c = int(m[1]),int(m[2])
                for idx_file in Path(cell).parent.parent.rglob("matrix_index.json"):
                    try:
                        d = json.loads(idx_file.read_text(encoding="utf-8"))
                        xl = d.get("x_lines",[])
                        yl = d.get("y_lines",[])
                        if c+1 < len(xl) and r+1 < len(yl):
                            boxes.append((int(xl[c]),int(yl[r]),
                                          int(xl[c+1]),int(yl[r+1])))
                            break
                    except Exception:
                        pass
                else:
                    boxes.append(_fallback_box(cell))
            else:
                boxes.append(_fallback_box(cell))
    return boxes


def _hough_boxes(cells_dir, cells):
    """
    Read bounding boxes from boxes.json written by run_hough().
    Falls back to filename patterns.
    """
    bbox_map = {}
    for bj in Path(cells_dir).rglob("boxes.json"):
        try:
            d = json.loads(bj.read_text(encoding="utf-8"))
            for fname,bb in d.items():
                bbox_map[fname] = tuple(int(v) for v in bb)
        except Exception:
            pass
    if bbox_map:
        return [bbox_map.get(Path(c).name, _fallback_box(c)) for c in cells]
    return [_fallback_box(c) for c in cells]


# ── OCR helpers ───────────────────────────────────────────────────────────────

def _ed(a, b):
    m,n = len(a),len(b)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m+1): dp[i][0]=i
    for j in range(n+1): dp[0][j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i][j] = (dp[i-1][j-1] if a[i-1]==b[j-1]
                        else 1+min(dp[i-1][j],dp[i][j-1],dp[i-1][j-1]))
    return dp[m][n]

def cer(true, pred):
    t,p = true.lower().strip(), pred.lower().strip()
    if not t: return 0.0 if not p else 1.0
    return _ed(t,p)/len(t)


# ── PDF -> PNG ────────────────────────────────────────────────────────────────

def to_png(src, dst_dir):
    ensure_dir(dst_dir)
    src = Path(src)
    if src.suffix.lower() == ".pdf":
        log(f"PDF -> PNG: {src.name}")
        try:
            import fitz
            pix = fitz.open(str(src))[0].get_pixmap(
                matrix=fitz.Matrix(300/72,300/72))
            out = Path(dst_dir)/f"{src.stem}_page0.png"
            pix.save(str(out))
            log(f"PNG: {out.name}", "ok")
            return out
        except ImportError:
            raise RuntimeError("pip install pymupdf")
    out = Path(dst_dir)/src.name
    shutil.copy2(str(src), str(out))
    return out


# ── Grid detection adapters ───────────────────────────────────────────────────

def _cuda():
    try:
        import torch; return torch.cuda.is_available()
    except ImportError:
        return False


def run_transformer(image, out):
    ensure_dir(out)
    log("Method A: Transformer/U-Net (Evan)", "h")
    t0 = time.time()
    sys.path.insert(0, str(Path("Evan_transformer_grid_finder").resolve()))
    weights = Path("Evan_transformer_grid_finder")/"scorecard_transformer_weights.pt"
    for mod_name in ["scorecard_transformer_extraction",
                     "scorecard_segmentation_extraction"]:
        try:
            mod = __import__(mod_name)
            mod._run_one(weights=weights, input_image=Path(image),
                         output_dir=Path(out), infer_cfg=mod.InferConfig(),
                         device="cuda" if _cuda() else "cpu")
            log(f"{mod_name} OK", "ok")
            break
        except Exception as e:
            log(f"{mod_name}: {e}", "warn")
    else:
        log("All Method A modules failed", "error")
        return [], time.time()-t0
    cells = sorted(Path(out).rglob("*.png"))
    el = time.time()-t0
    log(f"Method A: {len(cells)} cells in {el:.1f}s", "ok")
    return cells, el


def run_hough(image, out):
    """
    Run Joe's RCNN.  Saves boxes.json so IoU has real coordinates.
    Falls back to hough_lines if RCNN fails.
    """
    ensure_dir(out)
    log("Method B: Hough + RCNN (Joe)", "h")
    t0 = time.time()
    sys.path.insert(0, str(Path("Joe Code").resolve()))
    staging = Path(out).parent/"_hough_input_staging"
    ensure_dir(staging)
    staged = staging/Path(image).name
    if not staged.exists():
        shutil.copy2(str(image), str(staged))
    rcnn = str(Path("models")/"scorecard_rcnn.pth")
    ok = False

    # --- RCNN path ---
    try:
        import predict_cells as _pc
        import torch
        import torchvision.transforms.functional as TF
        from PIL import Image as PILImage

        device = torch.device("cuda" if _cuda() else "cpu")
        model = _pc.load_model(rcnn, device)
        img_pil = PILImage.open(str(staged)).convert("RGB")
        img_tensor = TF.to_tensor(img_pil)
        boxes = _pc.predict(model, img_tensor, device, score_thresh=0.5)

        if boxes:
            img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
            cell_dir = Path(out)/Path(image).stem
            ensure_dir(cell_dir)
            boxes_record = {}
            for i,(x1,y1,x2,y2) in enumerate(boxes):
                crop = img_bgr[max(0,y1):min(img_bgr.shape[0],y2),
                               max(0,x1):min(img_bgr.shape[1],x2)]
                if crop.size == 0: continue
                fname = f"cell_{i+1:04d}.png"
                cv2.imwrite(str(cell_dir/fname), crop)
                boxes_record[fname] = [x1,y1,x2,y2]
            # Save boxes for IoU reading
            (cell_dir/"boxes.json").write_text(
                json.dumps(boxes_record,indent=2), encoding="utf-8")
            (Path(out)/"boxes.json").write_text(
                json.dumps(boxes_record,indent=2), encoding="utf-8")
            log(f"RCNN: {len(boxes_record)} cells saved", "ok")
            ok = True

    except Exception as e:
        log(f"predict_cells: {e}", "warn")

    # --- Hough fallback ---
    if not ok:
        try:
            import hough_lines as _hl
            _hl.run(input_dir=str(staging), output_dir=str(out))
            log("Hough fallback OK", "ok")
            ok = True
        except Exception as e:
            log(f"hough_lines: {e}", "error")

    if not ok:
        return [], time.time()-t0

    cells = [c for c in sorted(Path(out).rglob("*.png"))
             if "debug" not in c.name]
    el = time.time()-t0
    log(f"Method B: {len(cells)} cells in {el:.1f}s", "ok")
    return cells, el


# ── OCR on cells ──────────────────────────────────────────────────────────────

def ocr_cells(cell_paths, gt_cells, pred_boxes, trocr, cnn, iou_thr=0.5):
    """
    Match cells to GT by IoU, run TrOCR / CNN / Hybrid, compute CER + accuracy.
    Returns list of result dicts.
    """
    from batch_test import get_cell_type, is_correct, hybrid_read_score

    vocab = [v.lower() for v in trocr.METRIC_VOCAB]
    has_text = bool(gt_cells) and any(c.has_text for c in gt_cells)

    gt_map = {}
    if gt_cells:
        m = iou_metrics(pred_boxes, gt_cells, iou_thr)
        for pi,gi,_ in m["matched_pairs"]:
            gt_map[pi] = gt_cells[gi]

    results = []
    for pi,img in enumerate(cell_paths):
        matched = gt_map.get(pi)
        if has_text and matched and matched.has_text:
            true = matched.text.lower().strip()
            ct = get_cell_type(true, vocab)
        else:
            true = "unknown"; ct = "score"

        # TrOCR
        if   ct=="empty":   tp=trocr.read_cell(str(img))
        elif ct=="label":   tp=trocr.match_metric_label(str(img))
        elif ct=="printed": tp=str(trocr.read_printed_number(str(img)) or "")
        elif ct=="other":   tp=trocr.read_cell(str(img))
        else:               tp=str(trocr.read_score_cell(str(img)) or "")

        # CNN
        if ct=="score":
            d,conf = cnn.read_digit(str(img)); cp=str(d) if d is not None else ""
        else:
            cp = tp

        # Hybrid
        if ct=="score":
            hp,meth = hybrid_read_score(cnn, trocr, str(img))
        else:
            hp,meth = tp,"trocr"

        row = dict(file=Path(img).name, true=true, type=ct,
                   trocr=tp, cnn=cp, hybrid=hp, hybrid_method=meth,
                   gt_matched=bool(matched))

        if has_text and true != "unknown":
            row["trocr_ok"]   = is_correct(true, tp)
            row["cnn_ok"]     = is_correct(true, cp)
            row["hybrid_ok"]  = is_correct(true, hp)
            row["trocr_cer"]  = cer(true, tp)
            row["cnn_cer"]    = cer(true, cp)
            row["hybrid_cer"] = cer(true, hp)

        results.append(row)
    return results


# ── Printing ──────────────────────────────────────────────────────────────────

def print_grid(cells_a, ta, cells_b, tb, gt_cells, thr):
    _banner("GRID DETECTION RESULTS")
    na,nb = len(cells_a),len(cells_b)
    print(f"\n  {'METHOD':<42} {'CELLS':>7} {'TIME':>7}")
    print(f"  {'-'*58}")
    ma = _c("  < more",C.GREEN) if na>nb else ""
    mb = _c("  < more",C.GREEN) if nb>na else ""
    print(f"  {'Method A - Transformer/U-Net (Evan)':<42} "
          f"{_c(f'{na:>7}',C.CYAN)}  {ta:>5.1f}s{ma}")
    print(f"  {'Method B - Hough + RCNN (Joe)':<42} "
          f"{_c(f'{nb:>7}',C.CYAN)}  {tb:>5.1f}s{mb}")

    if not gt_cells:
        print()
        log("No GT -> IoU skipped", "warn")
        log("Provide ScoreCards/CleanScansAnnotated/<stem>.json to enable IoU")
        return {}, [], []

    gt_text = any(c.has_text for c in gt_cells)
    print(f"\n  GT: {len(gt_cells)} cells "
          f"({'with text' if gt_text else 'bbox-only'})  |  IoU thr: {thr}\n")

    cd_a = cells_a[0].parent if cells_a else Path(".")
    cd_b = cells_b[0].parent if cells_b else Path(".")
    boxes_a = _evan_boxes(cd_a, cells_a)
    boxes_b = _hough_boxes(cd_b, cells_b)

    ia = iou_metrics(boxes_a, gt_cells, thr)
    ib = iou_metrics(boxes_b, gt_cells, thr)

    print(f"  {'METRIC':<20} {'Method A':>12} {'Method B':>12}   WINNER")
    print(f"  {'-'*58}")
    for metric in ["precision","recall","f1","mean_iou"]:
        va,vb = ia[metric],ib[metric]
        if   va>vb: w=_c("  <- A wins",C.GREEN)
        elif vb>va: w=_c("  <- B wins",C.GREEN)
        else:       w=_c("  tie",C.DIM)
        print(f"  {metric.upper():<20} {va:>12.3f} {vb:>12.3f}{w}")
    print(f"\n  {'TP/FP/FN':<20}"
          f"  {ia['tp']}/{ia['fp']}/{ia['fn']:>4}"
          f"   {ib['tp']}/{ib['fp']}/{ib['fn']:>4}")

    return {"method_a":ia,"method_b":ib}, boxes_a, boxes_b


def _row(scored, ck, ck2):
    n = len(scored)
    if n==0: return None,None,0
    acc = sum(r[ck] for r in scored)/n*100
    avg = sum(r[ck2] for r in scored)/n
    return acc,avg,n


def print_ocr(ra, rb):
    _banner("OCR RESULTS")
    has_gt = any("trocr_ok" in r for r in ra+rb)

    if not has_gt:
        log("No text GT -> showing raw predictions","warn")
        log("Add 'text' fields to annotation JSON to enable accuracy metrics")
        for lbl,res in [("Method A cells",ra),("Method B cells",rb)]:
            if not res: continue
            _section(lbl)
            print(f"  {'FILE':<30} {'TrOCR':>14} {'CNN':>10} {'HYBRID':>12}")
            print(f"  {'-'*68}")
            for r in res[:25]:
                print(f"  {r['file']:<30} {str(r['trocr'])[:14]:>14}"
                      f" {str(r['cnn'])[:10]:>10} {str(r['hybrid'])[:12]:>12}")
            if len(res)>25:
                print(f"  ... and {len(res)-25} more")
        return {}

    summary = []
    for glbl,res in [("Method A - Transformer",ra),
                     ("Method B - Hough",rb)]:
        scored = [r for r in res if "trocr_ok" in r]
        if not scored: continue
        _section(f"Grid: {glbl}  ({len(res)} cells, {len(scored)} evaluated)")
        print(f"\n  {'OCR METHOD':<35} {'CORRECT':>8} {'/ TOTAL':>8}"
              f" {'ACCURACY':>10} {'AVG CER':>9}  BAR")
        print(f"  {'-'*82}")
        for lbl,ck,ck2 in [
            ("TrOCR (handwritten model)",    "trocr_ok",  "trocr_cer"),
            ("CNN (digit classifier)",       "cnn_ok",    "cnn_cer"),
            ("Hybrid (CNN + TrOCR fallback)","hybrid_ok", "hybrid_cer"),
        ]:
            acc,avg,n = _row(scored,ck,ck2)
            if acc is None: continue
            correct = round(acc*n/100)
            print(f"  {lbl:<35} {correct:>8} / {n:<6}"
                  f" {_c(f'{acc:>9.1f}%',_pc(acc))}"
                  f" {_c(f'{avg:>9.3f}',_cc(avg))}"
                  f"  {_c(_bar(acc/100),C.BLUE)}")
            summary.append((f"{glbl} + {lbl}", acc, avg))

        types = sorted({r["type"] for r in scored})
        if len(types)>1:
            print(f"\n  {'CELL TYPE':<20} {'N':>4}"
                  f" {'TROCR':>10} {'CNN':>10} {'HYBRID':>10}")
            print(f"  {'-'*58}")
            for ct in types:
                sub = [r for r in scored if r["type"]==ct]; n=len(sub)
                def pf(k,_s=sub,_n=n):
                    return sum(r.get(k,False) for r in _s)/_n*100
                ta,ca,ha=pf("trocr_ok"),pf("cnn_ok"),pf("hybrid_ok")
                print(f"  {ct:<20} {n:>4}"
                      f" {_c(f'{ta:>9.1f}%',_pc(ta))}"
                      f" {_c(f'{ca:>9.1f}%',_pc(ca))}"
                      f" {_c(f'{ha:>9.1f}%',_pc(ha))}")

    if not summary: return {}

    _section("HEAD-TO-HEAD RANKING")
    print(f"\n  {'COMBINATION':<52} {'ACCURACY':>10} {'AVG CER':>10}  BAR")
    print(f"  {'-'*86}")
    summary.sort(key=lambda x: -x[1])
    for i,(lbl,acc,avg) in enumerate(summary):
        star = (_c("  BEST",C.GREEN+C.BOLD) if i==0 else
                _c("  WORST",C.RED+C.DIM)  if i==len(summary)-1 else "")
        print(f"  {lbl:<52}"
              f" {_c(f'{acc:>9.1f}%',_pc(acc))}"
              f" {_c(f'{avg:>10.3f}',_cc(avg))}"
              f"  {_c(_bar(acc/100),C.BLUE)}{star}")
    return {r[0]:{"accuracy_pct":r[1],"avg_cer":r[2]} for r in summary}


def print_verdict(iou, ocr):
    _banner("FINAL VERDICT")
    if iou:
        a=iou.get("method_a",{}); b=iou.get("method_b",{})
        if a and b:
            best="A (Transformer)" if a["f1"]>=b["f1"] else "B (Hough+RCNN)"
            print(f"\n  Grid winner : {_c(best,C.BOLD+C.GREEN)}"
                  f"  (F1  A={a['f1']:.3f}  B={b['f1']:.3f})")
    if ocr:
        best=max(ocr,key=lambda k:ocr[k]["accuracy_pct"])
        ba=ocr[best]["accuracy_pct"]; bc=ocr[best]["avg_cer"]
        print(f"  OCR winner  : {_c(best,C.BOLD+C.GREEN)}")
        print(f"  -> Accuracy {_c(f'{ba:.1f}%',C.GREEN+C.BOLD)}"
              f"  CER {_c(f'{bc:.3f}',C.GREEN+C.BOLD)}")


# ── Matrix builder ────────────────────────────────────────────────────────────

def build_matrix(results):
    rows={}
    for r in results:
        n=Path(r["file"]).stem
        rm=re.search(r'r(?:ow)?[\s_-]?(\d+)',n,re.I)
        cm=re.search(r'c(?:ol)?[\s_-]?(\d+)',n,re.I)
        if rm and cm:
            rows.setdefault(int(rm[1]),{})[int(cm[1])]=r["hybrid"]
    if not rows: return [[r["hybrid"] for r in results]]
    mr=max(rows); mc=max(max(c.keys()) for c in rows.values())
    return [[rows.get(ri,{}).get(ci,"") for ci in range(mc+1)]
            for ri in range(mr+1)]


# ── Single scorecard ──────────────────────────────────────────────────────────

def run_one(inp, out, gt_path=None, skip_a=False, skip_b=False,
            thr=0.5, trocr=None, cnn=None):
    inp=Path(inp); out=Path(out)
    ensure_dir(out)
    _banner(f"SCORECARD: {inp.name}")
    print(f"\n  Input  : {_c(inp,C.CYAN)}\n  Output : {_c(out,C.CYAN)}")

    if gt_path is None:
        gt_path = _auto_gt(inp)
        if gt_path: log(f"Auto GT: {gt_path.name}","ok")
        else:       log("No GT found -- IoU and OCR accuracy disabled","warn")

    gt_cells=[]
    if gt_path and Path(gt_path).exists():
        gt_cells = load_ground_truth(gt_path)
        has_text = any(c.has_text for c in gt_cells)
        log(f"GT: {len(gt_cells)} cells "
            f"({'with text' if has_text else 'bbox-only'})", "ok")
        if not has_text:
            log("GT has no text fields -- add 'text' to enable OCR accuracy","warn")
    elif gt_path:
        log(f"GT file not found: {gt_path}","warn")

    _section("STEP 1 - Image")
    img_png = to_png(inp, out/"input")

    _section("STEP 2 - Grid detection")
    cells_a,ta = [],0.0
    cells_b,tb = [],0.0
    if not skip_a: cells_a,ta = run_transformer(img_png, out/"cells_transformer")
    else: log("Method A skipped","warn")
    if not skip_b: cells_b,tb = run_hough(img_png, out/"cells_hough")
    else: log("Method B skipped","warn")

    ret = print_grid(cells_a,ta,cells_b,tb,gt_cells,thr)
    if isinstance(ret,tuple) and len(ret)==3:
        iou_sum,boxes_a,boxes_b = ret
    else:
        iou_sum = ret if isinstance(ret,dict) else {}
        boxes_a = [_fallback_box(c) for c in cells_a]
        boxes_b = [_fallback_box(c) for c in cells_b]

    _section("STEP 3 - OCR")
    if trocr is None:
        from ocr_engine import OCREngine
        trocr = OCREngine()
    if cnn is None:
        from cnn_digit_classifier import CNNDigitEngine
        mp = ("models/cnn_finetuned.pth" if os.path.exists("models/cnn_finetuned.pth")
              else "models/cnn_mnist.pth")
        cnn = CNNDigitEngine(model_path=mp)
        log(f"CNN: {mp}","ok")

    ra,rb = [],[]
    if cells_a:
        log(f"OCR on {len(cells_a)} Method A cells...")
        ra = ocr_cells(cells_a,gt_cells,boxes_a,trocr,cnn,thr)
        log("Method A OCR done","ok")
    if cells_b:
        log(f"OCR on {len(cells_b)} Method B cells...")
        rb = ocr_cells(cells_b,gt_cells,boxes_b,trocr,cnn,thr)
        log("Method B OCR done","ok")

    ocr_sum = print_ocr(ra,rb)
    print_verdict(iou_sum,ocr_sum)

    _section("STEP 4 - Saving")
    def _clean(d):
        if not d: return {}
        return {k:{m:(round(v,4) if isinstance(v,float) else v)
                   for m,v in dd.items() if m!="matched_pairs"}
                for k,dd in d.items()}

    payload = dict(
        input=str(inp), output=str(out),
        gt=str(gt_path) if gt_path else None,
        grid=dict(
            transformer=dict(n_cells=len(cells_a),time_s=round(ta,2)),
            hough=dict(n_cells=len(cells_b),time_s=round(tb,2)),
        ),
        iou_summary=_clean(iou_sum),
        ocr_summary=ocr_sum,
        ocr_results_transformer=ra,
        ocr_results_hough=rb,
    )
    (out/"results.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    log("Saved: results.json","ok")

    best = ra if len(ra)>=len(rb) else rb
    if best:
        mat = build_matrix(best)
        (out/"scorecard_matrix.json").write_text(
            json.dumps({"matrix":mat},indent=2),encoding="utf-8")
        with open(out/"scorecard_matrix.txt","w",encoding="utf-8") as f:
            for row in mat: f.write("\t".join(str(v) for v in row)+"\n")
        log(f"Saved: scorecard_matrix.txt ({len(mat)} rows)","ok")

    _banner(f"DONE - {inp.name}")
    print(f"\n  Results: {_c(out,C.CYAN+C.BOLD)}\n")
    return payload


# ── Batch mode ────────────────────────────────────────────────────────────────

_EXTS = {".pdf",".png",".jpg",".jpeg",".webp",".tif",".tiff"}

def run_batch(input_dir, out_root, skip_a=False, skip_b=False, thr=0.5):
    input_dir = Path(input_dir); out_root = Path(out_root)
    files = sorted(p for p in input_dir.iterdir()
                   if p.suffix.lower() in _EXTS)
    if not files:
        log(f"No scorecard files in {input_dir}","error"); return

    _banner(f"BATCH MODE -- {len(files)} scorecards in {input_dir.name}/")
    ensure_dir(out_root)

    _section("Loading OCR models (shared across batch)")
    from ocr_engine import OCREngine
    from cnn_digit_classifier import CNNDigitEngine
    trocr = OCREngine()
    mp = ("models/cnn_finetuned.pth" if os.path.exists("models/cnn_finetuned.pth")
          else "models/cnn_mnist.pth")
    cnn = CNNDigitEngine(model_path=mp)
    log(f"Models ready (CNN: {mp})","ok")

    batch=[]
    for i,f in enumerate(files,1):
        print(f"\n{_c(f'[{i}/{len(files)}]  {f.name}',C.BOLD+C.MAGENTA)}")
        try:
            p = run_one(f, out_root/f.stem,
                        skip_a=skip_a, skip_b=skip_b, thr=thr,
                        trocr=trocr, cnn=cnn)
            batch.append({"file":f.name,"status":"ok",**p})
        except Exception as e:
            log(f"Failed on {f.name}: {e}","error")
            batch.append({"file":f.name,"status":"error","error":str(e)})

    _banner("BATCH SUMMARY")
    ok=[r for r in batch if r["status"]=="ok"]
    print(f"\n  Total: {len(batch)}  |  OK: {len(ok)}"
          f"  |  Failed: {len(batch)-len(ok)}\n")

    agg: dict = {}
    for r in ok:
        for combo,m in (r.get("ocr_summary") or {}).items():
            agg.setdefault(combo,[]).append(m["accuracy_pct"])

    if agg:
        print(f"  {'COMBINATION':<52} {'MEAN ACC':>10} {'N':>6}")
        print(f"  {'-'*72}")
        for combo,accs in sorted(agg.items(),
                                  key=lambda x:-float(np.mean(x[1]))):
            avg=float(np.mean(accs))
            print(f"  {combo:<52}"
                  f" {_c(f'{avg:>9.1f}%',_pc(avg))} {len(accs):>6}")

    print(f"\n  {'SCORECARD':<26} {'GT':>5} {'A F1':>8} {'B F1':>8}"
          f" {'A cells':>8} {'B cells':>8}")
    print(f"  {'-'*68}")
    for r in ok:
        gtn=(len(load_ground_truth(r["gt"])) if r.get("gt") else 0)
        af1=r.get("iou_summary",{}).get("method_a",{}).get("f1",float("nan"))
        bf1=r.get("iou_summary",{}).get("method_b",{}).get("f1",float("nan"))
        na=r["grid"]["transformer"]["n_cells"]
        nb=r["grid"]["hough"]["n_cells"]
        fa=f"{af1:.3f}" if af1==af1 else "  n/a"
        fb=f"{bf1:.3f}" if bf1==bf1 else "  n/a"
        print(f"  {r['file']:<26} {gtn:>5} {fa:>8} {fb:>8} {na:>8} {nb:>8}")

    summary=dict(
        n_total=len(batch), n_success=len(ok),
        per_scorecard=batch,
        aggregate_ocr={
            k:{"mean_accuracy_pct":float(np.mean(v)),"n":len(v)}
            for k,v in agg.items()
        })
    (out_root/"summary.json").write_text(
        json.dumps(summary,indent=2),encoding="utf-8")
    with open(out_root/"summary.txt","w",encoding="utf-8") as f:
        f.write(f"Batch -- {len(batch)} scorecards\n{'='*60}\n\n")
        for combo,m in sorted(summary["aggregate_ocr"].items(),
                               key=lambda x:-x[1]["mean_accuracy_pct"]):
            f.write(f"{combo}\n"
                    f"  Mean accuracy : {m['mean_accuracy_pct']:.1f}%\n"
                    f"  Scorecards    : {m['n']}\n\n")
    log(f"Summary: {out_root/'summary.json'}","ok")
    _banner("BATCH COMPLETE")
    print(f"\n  All results: {_c(out_root,C.CYAN+C.BOLD)}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p=argparse.ArgumentParser(
        description="Scorecard digitization pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Single (GT auto-discovered from CleanScansAnnotated/)
  python pipeline.py --input ScoreCards/CleanScans/clean1.pdf

  # Single with explicit GT
  python pipeline.py --input ScoreCards/CleanScans/clean1.pdf \\
                     --ground_truth ScoreCards/CleanScansAnnotated/clean1.json

  # Batch -- every PDF/image in a folder
  python pipeline.py --batch ScoreCards/CleanScans/

  # Batch, custom output root
  python pipeline.py --batch ScoreCards/CleanScans/ --output my_results/

  # Skip one method
  python pipeline.py --input ScoreCards/CleanScans/clean1.pdf --skip_hough
        """)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--input", metavar="FILE",
                      help="Single scorecard (PDF or image)")
    mode.add_argument("--batch", metavar="DIR",
                      help="Folder of scorecards")
    p.add_argument("--output",           metavar="DIR",  default=None)
    p.add_argument("--ground_truth",     metavar="JSON", default=None)
    p.add_argument("--skip_transformer", action="store_true")
    p.add_argument("--skip_hough",       action="store_true")
    p.add_argument("--iou_threshold",    type=float, default=0.5)
    p.add_argument("--no-label", action="store_true")
    args=p.parse_args()

    if args.batch:
        run_batch(Path(args.batch),
                  Path(args.output) if args.output else Path("results"),
                  skip_a=args.skip_transformer, skip_b=args.skip_hough,
                  thr=args.iou_threshold)
    else:
        inp=Path(args.input)
        run_one(inp,
                Path(args.output) if args.output else Path("results")/inp.stem,
                gt_path=Path(args.ground_truth) if args.ground_truth else None,
                skip_a=args.skip_transformer, skip_b=args.skip_hough,
                thr=args.iou_threshold)

if __name__=="__main__":
    main()