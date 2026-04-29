# pdf_to_image.py
import fitz  # pip install pymupdf
import os
from pathlib import Path

def convert_pdf_folder(pdf_folder="CleanScans", output_folder="data/scans"):
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    for pdf_file in Path(pdf_folder).glob("*.pdf"):
        doc = fitz.open(pdf_file)
        for page_num in range(len(doc)):
            page = doc[page_num]
            # 300 DPI - high enough for OCR
            mat = fitz.Matrix(300/72, 300/72)
            pix = page.get_pixmap(matrix=mat)
            out_path = f"{output_folder}/{pdf_file.stem}_page{page_num}.png"
            pix.save(out_path)
            print(f"Saved {out_path}")

if __name__ == "__main__":
    convert_pdf_folder()