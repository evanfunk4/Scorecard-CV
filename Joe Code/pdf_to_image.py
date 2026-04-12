from pdf2image import convert_from_path
import os

input_folder = "CleanScans"
output_folder = "images"

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith(".pdf"):
        pdf_path = os.path.join(input_folder, file)
        images = convert_from_path(
            pdf_path,
            dpi=300,
            poppler_path=r"C:\poppler\Library\bin"
        )

        for i, img in enumerate(images):
            out_path = os.path.join(output_folder, f"{file[:-4]}_page{i}.png")
            img.save(out_path, "PNG")