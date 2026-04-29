import json
import random
from pathlib import Path
import cv2

RESULTS_DIR = Path("results")
LABELS_FILE = Path("evaluation_labels.json")

SAMPLES_PER_PAGE = 20

# ---------------------------
# Load saved labels
# ---------------------------
if LABELS_FILE.exists():
    with open(LABELS_FILE) as f:
        saved_labels = json.load(f)
else:
    saved_labels = {}

def save_labels():
    with open(LABELS_FILE, "w") as f:
        json.dump(saved_labels, f, indent=2)

# ---------------------------
# Show image (small window)
# ---------------------------
def show_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        print("⚠️ Could not load image:", img_path)
        return

    h, w = img.shape[:2]
    scale = min(400 / h, 400 / w, 1.0)
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))

    cv2.imshow("Cell", img_resized)

    # 👇 THIS FIXES LAG
    for _ in range(5):
        cv2.waitKey(10)

# ---------------------------
# Input label
# ---------------------------
def get_label():
    return input("GT (text, empty, multiple, bad): ").strip().lower()

# ---------------------------
# Metrics
# ---------------------------
total = 0
cnn_correct = 0
trocr_correct = 0

empty_count = 0
multiple_count = 0
bad_count = 0

print("\n==== STARTING EVALUATION ====\n")

for page_dir in sorted(RESULTS_DIR.glob("page_*")):
    print(f"\n===== {page_dir.name} =====")

    # ---------------------------
    # 1. Collect ALL cell images
    # ---------------------------
    cell_images = list(
        page_dir.glob("cells_transformer/**/cells/*.png")
    )

    if len(cell_images) == 0:
        print("No cell images found.")
        continue

    # ---------------------------
    # 2. Load predictions
    # ---------------------------
    results_file = page_dir / "results.json"
    if not results_file.exists():
        print("No results.json, skipping...")
        continue

    with open(results_file) as f:
        data = json.load(f)

    preds = data.get("ocr_results_transformer", [])

    # Filter debug junk
    preds = [p for p in preds if not p["file"].startswith("debug")]

    # ---------------------------
    # 3. Random sampling
    # ---------------------------
    sampled_imgs = random.sample(
        cell_images,
        min(SAMPLES_PER_PAGE, len(cell_images))
    )

    for i, img_path in enumerate(sampled_imgs):
        cell_id = str(img_path)

        # ---------------------------
        # Resume support
        # ---------------------------
        if cell_id in saved_labels:
            entry = saved_labels[cell_id]
            gt = entry["gt"]
            cnn_pred = entry["cnn"]
            trocr_pred = entry["trocr"]
        else:
            print(f"\n--- Cell {i+1}/{len(sampled_imgs)} ---")

            # crude alignment by index
            idx = i % len(preds)
            pred = preds[idx]

            cnn_pred = str(pred.get("cnn", "")).strip()
            trocr_pred = str(pred.get("trocr", "")).strip()

            print(f"CNN:   {cnn_pred}")
            print(f"TrOCR: {trocr_pred}")

            show_image(img_path)

            # Keep window responsive while waiting
            print("Type label in terminal, then press Enter...")

            gt = get_label()

            saved_labels[cell_id] = {
                "gt": gt,
                "cnn": cnn_pred,
                "trocr": trocr_pred
            }

            save_labels()

        # ---------------------------
        # Metrics update
        # ---------------------------
        if gt == "empty":
            empty_count += 1
            continue
        elif gt == "multiple":
            multiple_count += 1
            continue
        elif gt == "bad":
            bad_count += 1
            continue

        total += 1

        if gt == cnn_pred:
            cnn_correct += 1

        if gt == trocr_pred:
            trocr_correct += 1

cv2.destroyAllWindows()

# ---------------------------
# Final results
# ---------------------------
print("\n\n==== FINAL RESULTS ====")

print(f"Total evaluated: {total}")
if total > 0:
    print(f"CNN Accuracy:   {cnn_correct/total:.3f}")
    print(f"TrOCR Accuracy: {trocr_correct/total:.3f}")

print("\n==== LABEL STATS ====")
print(f"Empty:    {empty_count}")
print(f"Multiple: {multiple_count}")
print(f"Bad:      {bad_count}")