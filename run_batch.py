import subprocess
from pathlib import Path

input_dir = Path("ScoreCards/FilledCards/pages")

images = sorted(input_dir.glob("*.png"))

for img in images:
    print(f"\n=== Running: {img.name} ===")

    subprocess.run([
        "python",
        "pipeline.py",
        "--input",
        str(img),
        "--no-label"   # IMPORTANT (skip popups overnight)
    ])