"""
batch_test.py  —  TrOCR vs CNN evaluation on labeled scorecard cells.

Usage (from repo root):
    python scripts/batch_test.py

Reads labeled crops from data/cells/ and ground truth from data/cells/labels.json
(written by scripts/crop_tool.py).  Evaluates TrOCR and CNN side-by-side and
prints a per-cell table plus a summary broken down by cell type.

No hybrid comparison, no interactive labeling in this script.
"""

import os
import re
import sys
import json
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

CELLS_DIR   = Path("data/cells")
LABELS_FILE = Path("data/cells/labels.json")

# Allow running from the scripts/ subfolder
if not CELLS_DIR.exists():
    CELLS_DIR   = Path("../data/cells")
    LABELS_FILE = Path("../data/cells/labels.json")


# ── OCR helpers ───────────────────────────────────────────────────────────────

def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j-1] if a[i-1] == b[j-1]
                        else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]))
    return dp[m][n]


def cer(true: str, pred: str) -> float:
    """Character Error Rate = edit_distance / len(true)."""
    t, p = true.lower().strip(), pred.lower().strip()
    if not t:
        return 0.0 if not p else 1.0
    return edit_distance(t, p) / len(t)


def is_correct(true: str, pred: str) -> bool:
    t = true.lower().strip()
    p = pred.lower().strip()
    if t == "empty":
        return p == ""
    if not p:
        return False
    td = re.sub(r'\D', '', t)
    pd = re.sub(r'\D', '', p)
    if td and pd:
        return td == pd
    return t in p or p in t


def get_cell_type(label: str, metric_vocab: list[str]) -> str:
    """Route a cell to the appropriate OCR reader based on its GT label."""
    lab = label.lower()
    if lab == "empty":
        return "empty"
    if lab in metric_vocab or any(
            t in lab for t in ["tees", "hole", "handicap", "par", "yardage"]):
        return "label"
    if any(lab.startswith(p) for p in ["circled", "squared", "triangle"]):
        return "other"
    if lab.startswith("name") or lab.startswith("word:"):
        return "other"
    if len(re.sub(r'\D', '', lab)) == len(lab) and len(lab) >= 3:
        return "printed"
    return "score"


# ── Loading labels ────────────────────────────────────────────────────────────

def load_label_db() -> dict:
    """
    Load labels.json.  Falls back to filename-based parsing for any crop
    that isn't in the JSON (legacy behavior from old crop_tool).
    """
    db = {}
    if LABELS_FILE.exists():
        db = json.loads(LABELS_FILE.read_text(encoding="utf-8"))

    # Also pick up any PNG in CELLS_DIR not in the JSON (filename-parsed label)
    for p in sorted(CELLS_DIR.glob("*.png")):
        if p.name not in db:
            # filename convention: <label>_<N>.png
            stem = p.stem
            m = re.match(r'^(.+)_\d+$', stem)
            if m:
                lbl = m.group(1).replace("_", ":") if m.group(1).startswith("word") else m.group(1)
                db[p.name] = {"label": lbl, "index": -1, "page": "unknown"}

    return db


# ── Printing helpers ──────────────────────────────────────────────────────────

class C:
    RESET  = "\033[0m"; BOLD   = "\033[1m"; DIM = "\033[2m"
    GREEN  = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
    CYAN   = "\033[96m"

def _c(text, *codes):
    return "".join(codes) + str(text) + C.RESET

def _pct(p):
    col = C.GREEN if p >= 80 else (C.YELLOW if p >= 60 else C.RED)
    return _c(f"{p:>6.1f}%", col)

def _cer_fmt(v):
    col = C.GREEN if v <= 0.10 else (C.YELLOW if v <= 0.25 else C.RED)
    return _c(f"{v:>7.3f}", col)

def _mark(ok):
    return _c("  v", C.GREEN) if ok else _c("  x", C.RED)


def print_per_cell_table(results: list[dict]):
    print()
    print(_c("  PER-CELL RESULTS", C.BOLD))
    W = 102
    print(_c("  " + "-"*W, C.DIM))
    print(f"  {'FILE':<28} {'TRUE':<14} {'TYPE':<9} "
          f"{'TROCR PRED':<16} {'CER':>5}  "
          f"{'CNN PRED':<14} {'CER':>5}")
    print(_c("  " + "-"*W, C.DIM))

    for r in results:
        tmark = _c(" v", C.GREEN) if r["trocr_ok"] else _c(" x", C.RED)
        cmark = _c(" v", C.GREEN) if r["cnn_ok"]   else _c(" x", C.RED)
        print(f"  {r['file']:<28} {r['true']:<14} {r['type']:<9} "
              f"{str(r['trocr_pred'])[:14]:<14}{tmark}  {r['trocr_cer']:>5.3f}  "
              f"{str(r['cnn_pred'])[:12]:<12}{cmark}  {r['cnn_cer']:>5.3f}")

    print(_c("  " + "-"*W, C.DIM))


def print_summary_table(results: list[dict]):
    TYPES = ["score", "printed", "label", "empty", "other"]

    print()
    print(_c("  SUMMARY BY CELL TYPE", C.BOLD))
    W = 76
    print(_c("  " + "-"*W, C.DIM))
    print(f"  {'TYPE':<12} {'N':>5}  "
          f"{'TROCR ACC':>10} {'CER':>7}  "
          f"{'CNN ACC':>10} {'CER':>7}")
    print(_c("  " + "-"*W, C.DIM))

    total_n = 0
    total_tok = total_tcer = total_cok = total_ccer = 0

    for ct in TYPES:
        sub = [r for r in results if r["type"] == ct]
        if not sub:
            continue
        n = len(sub)
        tok  = sum(r["trocr_ok"]  for r in sub)
        cok  = sum(r["cnn_ok"]    for r in sub)
        tcer = sum(r["trocr_cer"] for r in sub) / n
        ccer = sum(r["cnn_cer"]   for r in sub) / n
        tacc = tok / n * 100
        cacc = cok / n * 100

        total_n    += n
        total_tok  += tok;  total_tcer += tcer * n
        total_cok  += cok;  total_ccer += ccer * n

        print(f"  {ct:<12} {n:>5}  "
              f"{_pct(tacc)} {_cer_fmt(tcer)}  "
              f"{_pct(cacc)} {_cer_fmt(ccer)}")

    if total_n:
        print(_c("  " + "-"*W, C.DIM))
        ota = total_tok / total_n * 100
        oca = total_cok / total_n * 100
        otc = total_tcer / total_n
        occ = total_ccer / total_n
        print(f"  {'OVERALL':<12} {total_n:>5}  "
              f"{_pct(ota)} {_cer_fmt(otc)}  "
              f"{_pct(oca)} {_cer_fmt(occ)}")
    print(_c("  " + "-"*W, C.DIM))


def print_head_to_head(results: list[dict]):
    score_only = [r for r in results if r["type"] == "score"]
    if not score_only:
        return

    print()
    print(_c("  HEAD-TO-HEAD: SCORE CELLS ONLY", C.BOLD))
    W = 52
    print(_c("  " + "-"*W, C.DIM))
    print(f"  {'MODEL':<30} {'ACCURACY':>10} {'AVG CER':>9}")
    print(_c("  " + "-"*W, C.DIM))

    n = len(score_only)
    for lbl, ok_key, cer_key in [
        ("TrOCR (handwritten model)", "trocr_ok", "trocr_cer"),
        ("CNN  (digit classifier)",   "cnn_ok",   "cnn_cer"),
    ]:
        acc = sum(r[ok_key]  for r in score_only) / n * 100
        avg = sum(r[cer_key] for r in score_only) / n
        print(f"  {lbl:<30} {_pct(acc)} {_cer_fmt(avg)}")

    # Winner
    trocr_acc = sum(r["trocr_ok"] for r in score_only) / n * 100
    cnn_acc   = sum(r["cnn_ok"]   for r in score_only) / n * 100
    if trocr_acc > cnn_acc:
        winner = "TrOCR"
    elif cnn_acc > trocr_acc:
        winner = "CNN"
    else:
        winner = "TIE"
    print(_c("  " + "-"*W, C.DIM))
    print(f"  Winner on score cells: {_c(winner, C.BOLD + C.GREEN)}")
    print(_c("  " + "-"*W, C.DIM))


def print_per_page_summary(results: list[dict], db: dict):
    pages = sorted({db[r["file"]].get("page","unknown")
                    for r in results if r["file"] in db})
    if len(pages) <= 1:
        return

    print()
    print(_c("  PER-PAGE SUMMARY", C.BOLD))
    W = 62
    print(_c("  " + "-"*W, C.DIM))
    print(f"  {'PAGE':<20} {'N':>4}  "
          f"{'TROCR':>10} {'CNN':>10}")
    print(_c("  " + "-"*W, C.DIM))

    for page in pages:
        sub = [r for r in results
               if db.get(r["file"],{}).get("page") == page]
        if not sub: continue
        n  = len(sub)
        ta = sum(r["trocr_ok"] for r in sub) / n * 100
        ca = sum(r["cnn_ok"]   for r in sub) / n * 100
        print(f"  {page:<20} {n:>4}  {_pct(ta)} {_pct(ca)}")
    print(_c("  " + "-"*W, C.DIM))


# ── Main evaluation ───────────────────────────────────────────────────────────

def main():
    # ── Load models ───────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent.parent))  # add repo root to path

    print("Loading TrOCR engine...")
    from ocr_engine import OCREngine
    trocr = OCREngine()

    model_path = ("models/cnn_finetuned.pth"
                  if os.path.exists("models/cnn_finetuned.pth")
                  else "models/cnn_mnist.pth")
    print(f"Loading CNN engine from {model_path}...")
    from cnn_digit_classifier import CNNDigitEngine
    cnn = CNNDigitEngine(model_path=model_path)

    metric_vocab = [v.lower() for v in trocr.METRIC_VOCAB]

    # ── Load labels ───────────────────────────────────────────────────────────
    db = load_label_db()
    if not db:
        print(f"\nNo labeled cells found.")
        print(f"Run:  python scripts/crop_tool.py   first to label cells.")
        sys.exit(1)

    cells = sorted(CELLS_DIR.glob("*.png"))
    usable = [(p, db[p.name]["label"]) for p in cells
              if p.name in db and db[p.name].get("label") not in ("", None)]

    if not usable:
        print("No usable labels found in labels.json.")
        sys.exit(1)

    print(f"\nEvaluating {len(usable)} labeled cells...\n")

    # ── Run OCR ───────────────────────────────────────────────────────────────
    results = []
    for img_path, true_label in usable:
        ct = get_cell_type(true_label, metric_vocab)

        # TrOCR — routed by cell type
        if ct == "empty":
            tp = trocr.read_cell(str(img_path))
        elif ct == "label":
            tp = trocr.match_metric_label(str(img_path))
        elif ct == "printed":
            tp = str(trocr.read_printed_number(str(img_path)) or "")
        elif ct == "other":
            tp = trocr.read_cell(str(img_path))
        else:
            tp = str(trocr.read_score_cell(str(img_path)) or "")

        # CNN — only meaningful for score cells; mirrors TrOCR otherwise
        if ct == "score":
            digit, conf = cnn.read_digit(str(img_path))
            cp = str(digit) if digit is not None else ""
        else:
            cp = tp   # non-score: CNN defers to TrOCR result

        results.append({
            "file":      img_path.name,
            "true":      true_label,
            "type":      ct,
            "trocr_pred": tp,
            "trocr_ok":  is_correct(true_label, tp),
            "trocr_cer": cer(true_label, tp),
            "cnn_pred":  cp,
            "cnn_ok":    is_correct(true_label, cp),
            "cnn_cer":   cer(true_label, cp),
        })

    # ── Print results ─────────────────────────────────────────────────────────
    W = 74
    print()
    print(_c("=" * W, C.BOLD))
    print(_c("  TrOCR vs CNN  —  Scorecard Cell Evaluation", C.BOLD))
    print(_c("=" * W, C.BOLD))

    print_per_cell_table(results)
    print_summary_table(results)
    print_head_to_head(results)
    print_per_page_summary(results, db)

    # ── Save results JSON ─────────────────────────────────────────────────────
    out = CELLS_DIR / "eval_results.json"
    out.write_text(json.dumps({"results": results}, indent=2), encoding="utf-8")
    print(f"\n  Full results saved to: {out}")


if __name__ == "__main__":
    main()