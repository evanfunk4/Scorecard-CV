import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps, ImageFilter
import re
import numpy as np
import cv2
 
class OCREngine:
    def __init__(self):
        print("Loading TrOCR handwritten model...")
        self.processor_hw = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
        self.model_hw = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
        self.model_hw.eval()
 
        print("Loading TrOCR printed model...")
        self.processor_pr = TrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
        self.model_pr = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
        self.model_pr.eval()
 
        print("Both models ready.")
 
        self.METRIC_VOCAB = [
            "hole", "par", "handicap", "hcp", "hdcp",
            "yardage", "yards", "score", "name",
            "we +/-", "they +/-", "red", "white", "blue",
            "black", "gold", "yellow", "green",
            "black tees", "blue tees", "white tees",
            "yellow tees", "red tees", "gold tees",
            "mens hdcp", "ladies hdcp",
            "hole number", "total", "out", "in"
        ]
 
    # ── Preprocessing ────────────────────────────────────────────────────────
 
    def is_empty_cell(self, image: Image.Image, threshold=0.95) -> bool:
        """
        Returns True if the cell is essentially blank.
        Threshold lowered to 0.95 — real empty cells score 0.993-1.000,
        cells with content score 0.90 or below, so 0.95 cleanly separates them.
        Debug print kept so you can monitor during development.
        """
        gray = np.array(image.convert("L"))
        light_ratio = np.sum(gray > 180) / gray.size
        # print(f"  [EMPTY CHECK] light_ratio={light_ratio:.3f}, threshold={threshold}")
        return light_ratio > threshold
 
    def remove_grid_lines(self, image: Image.Image, border=4) -> Image.Image:
        """Crop a few pixels from each edge to remove grid line bleed."""
        img = np.array(image.convert("L"))
        h, w = img.shape
        if h > border * 3 and w > border * 3:
            img = img[border:h-border, border:w-border]
        return Image.fromarray(img)
 
    def upscale(self, image: Image.Image, min_height=80) -> Image.Image:
        """
        Upscale small crops so TrOCR has enough pixels to work with.
        Increased min_height to 80 to give more resolution to blurry
        yardage crops like 369, 392, 437.
        """
        w, h = image.size
        if h < min_height:
            scale = min_height / h
            new_w = max(int(w * scale), min_height * 4)
            image = image.resize((new_w, min_height), Image.LANCZOS)
        return image
 
    def deskew(self, pil_image: Image.Image) -> Image.Image:
        """Correct slight skew from non-flat scans."""
        img = np.array(pil_image)
        coords = np.column_stack(np.where(img < 128))
        if len(coords) < 10:
            return pil_image
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = 90 + angle
        if abs(angle) < 0.5:
            return pil_image
        (h, w) = img.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        return Image.fromarray(rotated)
 
    def preprocess_cell(self, image: Image.Image) -> Image.Image:
        """
        Full preprocessing pipeline for a raw cell crop.
        Order: strip borders → grayscale → upscale → enhance → deskew → pad → RGB
        """
        # 1. Strip grid line bleed from edges
        image = self.remove_grid_lines(image)
 
        # 2. Grayscale — handles colored backgrounds (red/yellow tee rows etc.)
        image = image.convert("L")
 
        # 3. Upscale before enhancing — more pixels = better enhancement
        image = self.upscale(image, min_height=80)
 
        # 4. Auto-contrast — stretches histogram, helps faint pencil marks
        image = ImageOps.autocontrast(image, cutoff=2)
 
        # 5. Sharpen twice — helps blurry crops like the yardage cells
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.SHARPEN)
 
        # 6. Deskew — correct slight rotation from non-flat scans
        image = self.deskew(image)
 
        # 7. Pad to consistent landscape aspect ratio for TrOCR
        image = ImageOps.pad(image, (384, 96), color=255)
 
        # 8. Back to RGB — TrOCR requires 3-channel input
        image = image.convert("RGB")
 
        return image
 
    # ── Core OCR ─────────────────────────────────────────────────────────────
 
    def _run_trocr(self, image: Image.Image, printed: bool = False) -> str:
        """Internal: run either the handwritten or printed TrOCR model."""
        processor = self.processor_pr if printed else self.processor_hw
        model = self.model_pr if printed else self.model_hw
 
        with torch.no_grad():
            pixel_values = processor(images=image, return_tensors="pt").pixel_values
            generated_ids = model.generate(
                pixel_values,
                max_new_tokens=20
            )
        return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
 
    def read_cell(self, image_path: str, printed: bool = False) -> str:
        """
        Run OCR on a single cell image. Returns raw predicted string.
        Set printed=True for tee label rows and yardage (non-handwritten content).
        """
        image = Image.open(image_path).convert("RGB")
 
        if self.is_empty_cell(image):
            return ""
 
        image = self.preprocess_cell(image)
        return self._run_trocr(image, printed=printed)
 
    def read_score_cell(self, image_path: str) -> int | None:
        """
        For handwritten score cells.
        Returns an int if a valid number is found, None if empty or unreadable.
        """
        raw = self.read_cell(image_path, printed=False)
 
        if raw == "":
            return None
 
        digits = re.sub(r'\D', '', raw)
 
        if digits == "":
            print(f"  [UNREADABLE] No digits found in raw='{raw}'")
            return None
 
        value = int(digits)
 
        if value > 999:
            print(f"  [WARNING] Suspicious value {value} from raw='{raw}' — check cell")
 
        return value
 
    def read_printed_number(self, image_path: str) -> int | None:
        """
        For printed number cells like yardage (369, 437, etc.).
        Uses the printed model which handles clean typeset numbers better.
        """
        raw = self.read_cell(image_path, printed=True)
 
        if raw == "":
            return None
 
        digits = re.sub(r'\D', '', raw)
 
        if digits == "":
            print(f"  [UNREADABLE] No digits in printed cell, raw='{raw}'")
            return None
 
        return int(digits)
 
    def match_metric_label(self, image_path: str) -> str:
        """
        For row-header label cells (Par, Handicap, Black Tees, etc.).
        Uses printed model + fuzzy vocab matching.
        Truncates to first two words to handle trailing content like
        '68.3/117' (course rating/slope) which appears after tee names.
        Returns best matching label or 'other: <raw>' if nothing fits.
        """
        raw = self.read_cell(image_path, printed=True).lower().strip()
 
        if raw == "":
            return "empty"
 
        # Only use first two words — tee rows have trailing rating/slope numbers
        # e.g. "yellow tees 68.3/117" → we only want "yellow tees"
        truncated = " ".join(raw.split()[:2])
 
        best_match = "other"
        best_score = float('inf')
 
        for label in self.METRIC_VOCAB:
            dist = self._edit_distance(truncated, label)
            if dist < best_score:
                best_score = dist
                best_match = label
 
        if best_score <= 4:
            return best_match
        else:
            return f"other: {raw}"
 
    # ── Evaluation ───────────────────────────────────────────────────────────
 
    def evaluate_batch(self, cells_dir: str = "data/cells") -> dict:
        """
        Run OCR across all labeled cell images in cells_dir and compute
        accuracy and CER broken down by cell type.
 
        Filename convention: <label>_<N>.png
        e.g. 4_9.png, par_3.png, empty_19.png, circled_4_22.png
        """
        from pathlib import Path
 
        results = []
 
        cell_types = {
            "score":   [],
            "printed": [],
            "label":   [],
            "empty":   [],
            "other":   [],
        }
 
        for img_path in sorted(Path(cells_dir).glob("*.png")):
            true_label = img_path.stem.rsplit("_", 1)[0].lower()
 
            # Route to the right reader based on label type
            if true_label == "empty":
                predicted = self.read_cell(str(img_path))
                cell_type = "empty"
 
            elif true_label in [l.lower() for l in self.METRIC_VOCAB] or \
                 any(t in true_label for t in ["tees", "hole", "handicap", "par", "yardage"]):
                predicted = self.match_metric_label(str(img_path))
                cell_type = "label"
 
            elif any(true_label.startswith(p) for p in ["circled", "squared", "triangle"]):
                predicted = self.read_cell(str(img_path))
                cell_type = "other"
 
            elif true_label.startswith("name"):
                predicted = self.read_cell(str(img_path))
                cell_type = "other"
 
            elif len(re.sub(r'\D', '', true_label)) == len(true_label) and len(true_label) >= 3:
                # All digits and 3+ chars = likely a yardage/printed number
                predicted = str(self.read_printed_number(str(img_path)) or "")
                cell_type = "printed"
 
            else:
                # Default: treat as handwritten score
                predicted = str(self.read_score_cell(str(img_path)) or "")
                cell_type = "score"
 
            correct = self._is_correct(true_label, predicted)
            cer = self._cer(true_label, predicted)
 
            result = {
                "file":      img_path.name,
                "true":      true_label,
                "predicted": predicted,
                "correct":   correct,
                "cer":       cer,
                "type":      cell_type,
            }
            results.append(result)
            cell_types[cell_type].append(result)
 
        # ── Print per-result table ───────────────────────────────────────────
        print(f"\n{'FILE':<30} {'TRUE':<20} {'PRED':<25} {'OK':>3} {'CER':>6}")
        print("-" * 90)
        for r in results:
            mark = "✓" if r["correct"] else "✗"
            print(f"{r['file']:<30} {r['true']:<20} {r['predicted']:<25} {mark:>3} {r['cer']:>6.3f}")
 
        # ── Summary stats ────────────────────────────────────────────────────
        print(f"\n{'='*90}")
        print(f"{'SUMMARY BY CELL TYPE':^90}")
        print(f"{'='*90}")
        print(f"{'TYPE':<12} {'COUNT':>6} {'CORRECT':>8} {'ACCURACY':>10} {'AVG CER':>9}")
        print("-" * 50)
 
        overall_correct = 0
        overall_total = 0
        overall_cer = []
 
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
            overall_cer.extend(r["cer"] for r in cresults)
 
        print("-" * 50)
        overall_acc = overall_correct / overall_total * 100 if overall_total else 0
        overall_avg_cer = sum(overall_cer) / len(overall_cer) if overall_cer else 0
        print(f"{'OVERALL':<12} {overall_total:>6} {overall_correct:>8} {overall_acc:>9.1f}% {overall_avg_cer:>9.3f}")
        print(f"{'='*90}\n")
 
        return results
 
    # ── Helpers ──────────────────────────────────────────────────────────────
 
    def _is_correct(self, true: str, predicted: str) -> bool:
        """
        Flexible correctness check:
        - For numbers, compare digit-only versions
        - For labels, check if true label appears in predicted
        - For empty, correct only if predicted is also empty
        """
        true = true.lower().strip()
        predicted = predicted.lower().strip()
 
        if true == "empty":
            return predicted == ""
 
        # If predicted is empty but true isn't, that's always wrong
        if predicted == "":
            return False
 
        true_digits = re.sub(r'\D', '', true)
        pred_digits = re.sub(r'\D', '', predicted)
        if true_digits and pred_digits:
            return true_digits == pred_digits
 
        return true in predicted or predicted in true
 
    def _cer(self, true: str, predicted: str) -> float:
        """
        Character Error Rate = edit_distance(true, predicted) / len(true)
        Standard metric for OCR evaluation (referenced in the project report).
        """
        true = true.lower().strip()
        predicted = predicted.lower().strip()
        if len(true) == 0:
            return 0.0 if len(predicted) == 0 else 1.0
        return self._edit_distance(true, predicted) / len(true)
 
    def _edit_distance(self, a: str, b: str) -> int:
        """Standard Levenshtein edit distance."""
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
 
 
# ── Entry points ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    engine = OCREngine()
 
    # Quick single cell test
    score = engine.read_score_cell("data/test_digit1.png")
    print(f"\nSingle cell test — Score: {score}")
 
    # Full batch evaluation with metrics
    engine.evaluate_batch("data/cells")
 