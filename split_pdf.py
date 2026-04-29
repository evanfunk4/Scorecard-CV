from pdf2image import convert_from_path
from pathlib import Path

pdf_path = Path("ScoreCards/FilledCards/TOURNAMENT.pdf")
out_dir = Path("ScoreCards/FilledCards/pages")
out_dir.mkdir(parents=True, exist_ok=True)

pages = convert_from_path(pdf_path, dpi=300)

for i, page in enumerate(pages):
    out_path = out_dir / f"page_{i:03d}.png"
    page.save(out_path, "PNG")
    print(f"[saved] {out_path}")