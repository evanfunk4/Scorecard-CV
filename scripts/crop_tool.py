"""
crop_tool.py  —  Interactive cell labeler for filled scorecard pages.

Usage (from repo root):
    python scripts/crop_tool.py

    Processes every page_000.png … page_015.png in ScoreCards/FilledCards/pages/.
    For each page you are asked to crop and label LABELS_PER_PAGE cells.
    Labeled crops are saved to data/cells/<label>_<N>.png

Controls while the window is open:
    Click + drag   Draw a crop box around a cell
    s              Save the current box — you will be prompted for a label
    r              Reset / discard the current box without saving
    n              Skip to the next page (if you've labeled enough)
    q              Quit entirely

Label conventions:
    Single digit   →  4          (handwritten score)
    Multi-digit    →  10         (score >= 10)
    Empty cell     →  empty
    Par row        →  par
    Handicap row   →  handicap
    Yardage        →  369        (3+ digit number = printed)
    Player name    →  name
    Other text     →  word:<text>   e.g.  word:total

Labels are appended to data/cells/labels.json so batch_test.py can
always find the ground truth without relying on filename parsing.
"""

import cv2
import json
import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

PAGES_DIR    = Path("ScoreCards/FilledCards/pages")
CELLS_DIR    = Path("data/cells")
LABELS_FILE  = Path("data/cells/labels.json")
LABELS_PER_PAGE = 20

# Display window will be scaled to fit this height (px) — adjust if needed
MAX_DISPLAY_H = 900


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dirs():
    CELLS_DIR.mkdir(parents=True, exist_ok=True)


def load_labels() -> dict:
    if LABELS_FILE.exists():
        return json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    return {}


def save_labels(db: dict):
    LABELS_FILE.write_text(json.dumps(db, indent=2), encoding="utf-8")


def next_crop_index(db: dict) -> int:
    """Return the next available global crop index."""
    if not db:
        return 0
    return max(int(v["index"]) for v in db.values()) + 1


def scale_factor(img_h: int) -> float:
    if img_h <= MAX_DISPLAY_H:
        return 1.0
    return MAX_DISPLAY_H / img_h


def scale_box(box, sf):
    """Scale a box back to original image coordinates."""
    (x1, y1), (x2, y2) = box
    return (round(x1 / sf), round(y1 / sf),
            round(x2 / sf), round(y2 / sf))


def normalise_box(x1, y1, x2, y2):
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def already_labeled_for_page(db: dict, page_name: str) -> int:
    return sum(1 for v in db.values()
               if v.get("page") == page_name and v.get("label") not in ("", None))


# ── Interactive labeler ───────────────────────────────────────────────────────

class CropTool:
    def __init__(self):
        self.drawing = False
        self.start   = (0, 0)
        self.rect    = None   # ((x1,y1),(x2,y2)) in display coords

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start   = (x, y)
            self.rect    = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.rect = (self.start, (x, y))
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rect    = (self.start, (x, y))

    def run_page(self, img_path: Path, db: dict, crop_idx: int,
                 target: int) -> tuple[dict, int]:
        """
        Show one page.  Collect up to `target` new labeled cells.
        Returns updated (db, crop_idx).
        """
        img_orig = cv2.imread(str(img_path))
        if img_orig is None:
            print(f"  [!] Cannot read {img_path} — skipping")
            return db, crop_idx

        h, w = img_orig.shape[:2]
        sf   = scale_factor(h)
        disp_w, disp_h = round(w * sf), round(h * sf)
        img_disp = cv2.resize(img_orig, (disp_w, disp_h)) if sf < 1.0 else img_orig.copy()

        page_name  = img_path.name
        already    = already_labeled_for_page(db, page_name)
        still_need = max(0, target - already)

        print(f"\n{'='*60}")
        print(f"  Page: {page_name}")
        print(f"  Already labeled: {already}  |  Need {still_need} more")
        print(f"  Controls: drag=draw box  s=save  r=reset  n=next  q=quit")
        print(f"{'='*60}")

        if still_need == 0:
            print("  All labels collected for this page — skipping.")
            return db, crop_idx

        win = f"Crop Tool — {page_name}  (need {still_need} more labels)"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, disp_w, disp_h)
        cv2.setMouseCallback(win, self.mouse_callback)
        self.rect = None

        labeled_this_page = 0

        while labeled_this_page < still_need:
            canvas = img_disp.copy()

            # Draw current box
            if self.rect:
                cv2.rectangle(canvas, self.rect[0], self.rect[1], (0, 255, 0), 2)

            # Status overlay
            status = (f"Labeled {already + labeled_this_page} / "
                      f"{already + still_need}  |  s=save  r=reset  n=next  q=quit")
            cv2.putText(canvas, status, (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow(win, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                cv2.destroyAllWindows()
                print("\nQuitting crop tool.")
                return db, crop_idx

            elif key == ord('n'):
                print(f"  Skipping to next page (labeled {labeled_this_page} this session).")
                break

            elif key == ord('r'):
                self.rect = None
                print("  Box reset.")

            elif key == ord('s') and self.rect:
                rx1, ry1, rx2, ry2 = normalise_box(
                    self.rect[0][0], self.rect[0][1],
                    self.rect[1][0], self.rect[1][1])

                if abs(rx2 - rx1) < 5 or abs(ry2 - ry1) < 5:
                    print("  Box too small — draw a bigger area.")
                    continue

                # Scale back to original coords
                ox1, oy1, ox2, oy2 = scale_box(
                    ((rx1, ry1), (rx2, ry2)), sf)
                crop = img_orig[oy1:oy2, ox1:ox2]

                if crop.size == 0:
                    print("  Empty crop — try again.")
                    continue

                # Show close-up for confirmation
                MAX_PREVIEW = 300
                ph, pw = crop.shape[:2]
                psf = min(MAX_PREVIEW / max(pw, 1), MAX_PREVIEW / max(ph, 1), 3.0)
                preview = cv2.resize(crop, (round(pw*psf), round(ph*psf)))
                cv2.imshow("Preview — press any key", preview)
                cv2.waitKey(500)          # brief flash; user types label below
                cv2.destroyWindow("Preview — press any key")

                print()
                print("  Label this cell:")
                print("    digit  → just type it:  4   (or  10  for two-digit)")
                print("    empty  → empty")
                print("    label  → par / handicap / hole / yardage / name")
                print("    other  → word:<text>  e.g.  word:total")
                print("    skip   → (press Enter with no label to skip)")
                label = input("  Label: ").strip().lower()

                if not label:
                    print("  Skipped.")
                    self.rect = None
                    continue

                # Save crop
                safe_label = label.replace(":", "_").replace("/", "_")
                fname = f"{safe_label}_{crop_idx}.png"
                save_path = CELLS_DIR / fname
                cv2.imwrite(str(save_path), crop)

                db[fname] = {
                    "label":  label,
                    "index":  crop_idx,
                    "page":   page_name,
                    "bbox":   [ox1, oy1, ox2, oy2],
                }
                save_labels(db)

                print(f"  Saved: {save_path}  (label={label})")
                crop_idx         += 1
                labeled_this_page += 1
                self.rect = None

        cv2.destroyAllWindows()
        return db, crop_idx


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    pages = sorted(PAGES_DIR.glob("page_*.png"))
    if not pages:
        print(f"No pages found in {PAGES_DIR}")
        print("Expected: ScoreCards/FilledCards/pages/page_000.png ... page_015.png")
        sys.exit(1)

    print(f"Found {len(pages)} pages in {PAGES_DIR}")
    print(f"Will ask for {LABELS_PER_PAGE} labels per page.")
    print(f"Crops saved to: {CELLS_DIR}/")
    print(f"Labels saved to: {LABELS_FILE}")

    db        = load_labels()
    crop_idx  = next_crop_index(db)
    tool      = CropTool()

    for page in pages:
        db, crop_idx = tool.run_page(page, db, crop_idx,
                                     target=LABELS_PER_PAGE)

    total = sum(1 for v in db.values() if v.get("label") not in ("", None))
    print(f"\nDone. Total labeled cells: {total}")
    print(f"Run:  python scripts/batch_test.py  to evaluate TrOCR vs CNN")


if __name__ == "__main__":
    main()