# batch_test.py
# Full three-way comparison: TrOCR vs CNN vs Hybrid (CNN + TrOCR fallback)

from ocr_engine import OCREngine
from cnn_digit_classifier import CNNDigitEngine
from pathlib import Path
import re
import os


# ── Hybrid reader ─────────────────────────────────────────────────────────────

def hybrid_read_score(cnn_engine: CNNDigitEngine,
                      trocr_engine: OCREngine,
                      image_path: str,
                      confidence_threshold: float = 0.6) -> tuple[str, str]:
    """
    Try CNN first. If it returns empty or low confidence, fall back to TrOCR.
    Returns (predicted_value_as_string, method_used).
    """
    digit, conf = cnn_engine.read_digit(image_path)

    if digit is not None and conf >= confidence_threshold:
        return str(digit), "cnn"

    # CNN not confident — fall back to TrOCR
    raw = trocr_engine.read_score_cell(image_path)
    result = str(raw) if raw is not None else ""
    reason = "low_conf" if digit is not None else "empty"
    conf_display = f"{conf:.2f}" if conf is not None else "0.00"
    print(f"  [FALLBACK] CNN {reason} (conf={conf_display}) "
          f"→ TrOCR predicted '{result}' for {Path(image_path).name}")
    return result, "trocr"


# ── Helpers ───────────────────────────────────────────────────────────────────

def is_correct(true: str, predicted: str) -> bool:
    true = true.lower().strip()
    predicted = predicted.lower().strip()

    if true == "empty":
        return predicted == ""
    if predicted == "":
        return False

    true_digits = re.sub(r'\D', '', true)
    pred_digits = re.sub(r'\D', '', predicted)
    if true_digits and pred_digits:
        return true_digits == pred_digits

    return true in predicted or predicted in true


def cer(true: str, predicted: str) -> float:
    """Character Error Rate = edit_distance / len(true)"""
    true = true.lower().strip()
    predicted = predicted.lower().strip()
    if len(true) == 0:
        return 0.0 if len(predicted) == 0 else 1.0
    return edit_distance(true, predicted) / len(true)


def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def get_cell_type(true_label: str, metric_vocab: list[str]) -> str:
    """Classify a cell by its label into one of the routing categories."""
    label = true_label.lower()

    if label == "empty":
        return "empty"
    elif label in metric_vocab or any(
            t in label for t in ["tees", "hole", "handicap", "par", "yardage"]):
        return "label"
    elif any(label.startswith(p) for p in ["circled", "squared", "triangle"]):
        return "other"
    elif label.startswith("name"):
        return "other"
    elif len(re.sub(r'\D', '', label)) == len(label) and len(label) >= 3:
        return "printed"
    else:
        return "score"


def print_summary(label: str, results: list[dict]):
    """Print per-type breakdown and overall summary for one model's results."""
    cell_types = {"score": [], "printed": [], "label": [], "empty": [], "other": []}

    for r in results:
        cell_types[r["type"]].append(r)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"{'TYPE':<12} {'COUNT':>6} {'CORRECT':>8} {'ACCURACY':>10} {'AVG CER':>9}")
    print("-" * 50)

    overall_correct = 0
    overall_total = 0
    overall_cer_vals = []

    for ctype, cresults in cell_types.items():
        if not cresults:
            continue
        n = len(cresults)
        correct = sum(r["correct"] for r in cresults)
        avg_cer = sum(r["cer"] for r in cresults) / n
        accuracy = correct / n * 100
        print(f"{ctype:<12} {n:>6} {correct:>8} {accuracy:>9.1f}% {avg_cer:>9.3f}")
        overall_correct += correct
        overall_total += n
        overall_cer_vals.extend(r["cer"] for r in cresults)

    print("-" * 50)
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0
    overall_avg_cer = (sum(overall_cer_vals) / len(overall_cer_vals)
                       if overall_cer_vals else 0)
    print(f"{'OVERALL':<12} {overall_total:>6} {overall_correct:>8} "
          f"{overall_acc:>9.1f}% {overall_avg_cer:>9.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cells_dir = "data/cells"

    all_cells = sorted(Path(cells_dir).glob("*.png"))
    if not all_cells:
        print(f"No cell images found in {cells_dir}")
        exit(1)

    print(f"Found {len(all_cells)} cell images in {cells_dir}\n")

    # ── Load models ───────────────────────────────────────────────────────────
    print("Loading TrOCR engine...")
    trocr = OCREngine()

    model_path = ("models/cnn_finetuned.pth"
                  if os.path.exists("models/cnn_finetuned.pth")
                  else "models/cnn_mnist.pth")
    print(f"\nLoading CNN engine from {model_path}...")
    cnn = CNNDigitEngine(model_path=model_path)

    metric_vocab = [v.lower() for v in trocr.METRIC_VOCAB]

    # ── Per-file results storage ──────────────────────────────────────────────
    trocr_results = []
    cnn_results = []
    hybrid_results = []

    # ── Per-file table ────────────────────────────────────────────────────────
    print(f"\n{'FILE':<30} {'TRUE':<20} "
          f"{'TROCR':<15} {'CNN':<12} {'HYBRID':<18} {'TYPE':<10}")
    print("-" * 110)

    for img_path in all_cells:
        true_label = img_path.stem.rsplit("_", 1)[0].lower()
        cell_type = get_cell_type(true_label, metric_vocab)

        # ── TrOCR prediction ──────────────────────────────────────────────────
        if cell_type == "empty":
            trocr_pred = trocr.read_cell(str(img_path))
        elif cell_type == "label":
            trocr_pred = trocr.match_metric_label(str(img_path))
        elif cell_type == "printed":
            trocr_pred = str(trocr.read_printed_number(str(img_path)) or "")
        elif cell_type == "other":
            trocr_pred = trocr.read_cell(str(img_path))
        else:
            trocr_pred = str(trocr.read_score_cell(str(img_path)) or "")

        # ── CNN prediction ────────────────────────────────────────────────────
        # CNN only handles score cells — everything else defers to TrOCR
        if cell_type == "score":
            digit, conf = cnn.read_digit(str(img_path))
            cnn_pred = str(digit) if digit is not None else ""
        else:
            cnn_pred = trocr_pred

        # ── Hybrid prediction ─────────────────────────────────────────────────
        if cell_type == "score":
            hybrid_pred, method = hybrid_read_score(cnn, trocr, str(img_path))
        else:
            hybrid_pred = trocr_pred
            method = "trocr"

        # ── Score everything ──────────────────────────────────────────────────
        trocr_ok   = is_correct(true_label, trocr_pred)
        cnn_ok     = is_correct(true_label, cnn_pred)
        hybrid_ok  = is_correct(true_label, hybrid_pred)

        trocr_cer  = cer(true_label, trocr_pred)
        cnn_cer    = cer(true_label, cnn_pred)
        hybrid_cer = cer(true_label, hybrid_pred)

        trocr_results.append({
            "file": img_path.name, "true": true_label,
            "predicted": trocr_pred, "correct": trocr_ok,
            "cer": trocr_cer, "type": cell_type
        })
        cnn_results.append({
            "file": img_path.name, "true": true_label,
            "predicted": cnn_pred, "correct": cnn_ok,
            "cer": cnn_cer, "type": cell_type
        })
        hybrid_results.append({
            "file": img_path.name, "true": true_label,
            "predicted": hybrid_pred, "correct": hybrid_ok,
            "cer": hybrid_cer, "type": cell_type
        })

        # Format columns
        t_mark = "✓" if trocr_ok else "✗"
        c_mark = "✓" if cnn_ok  else "✗"
        h_mark = "✓" if hybrid_ok else "✗"

        trocr_col  = f"{trocr_pred[:12]:<12} {t_mark}"
        cnn_col    = f"{cnn_pred[:8]:<8} {c_mark}"
        hybrid_col = f"{hybrid_pred[:10]:<10} {h_mark} [{method[:3]}]"

        print(f"{img_path.name:<30} {true_label:<20} "
              f"{trocr_col:<15} {cnn_col:<12} {hybrid_col:<20} {cell_type:<10}")

    # ── Summary tables ────────────────────────────────────────────────────────
    print_summary("TROCR SUMMARY", trocr_results)
    print_summary("CNN SUMMARY (score cells defer to TrOCR for other types)",
                  cnn_results)
    print_summary("HYBRID SUMMARY (CNN + TrOCR fallback on score cells)",
                  hybrid_results)

    # ── Head to head on score cells only ─────────────────────────────────────
    score_trocr  = [r for r in trocr_results  if r["type"] == "score"]
    score_cnn    = [r for r in cnn_results    if r["type"] == "score"]
    score_hybrid = [r for r in hybrid_results if r["type"] == "score"]

    print(f"\n{'='*60}")
    print(f"  HEAD-TO-HEAD: SCORE CELLS ONLY")
    print(f"{'='*60}")
    print(f"{'MODEL':<35} {'ACCURACY':>10} {'AVG CER':>10}")
    print("-" * 57)

    for label, results in [
        ("TrOCR (handwritten model)", score_trocr),
        ("CNN (finetuned on scorecard cells)", score_cnn),
        ("Hybrid (CNN + TrOCR fallback)", score_hybrid),
    ]:
        n = len(results)
        if n == 0:
            continue
        acc = sum(r["correct"] for r in results) / n * 100
        avg_cer = sum(r["cer"] for r in results) / n
        print(f"{label:<35} {acc:>9.1f}% {avg_cer:>10.3f}")

    print(f"{'='*60}\n")