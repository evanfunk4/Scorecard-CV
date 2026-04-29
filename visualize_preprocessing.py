"""
visualize_preprocessing.py

Shows each preprocessing step for CNN and TrOCR side-by-side.
Saves intermediate images at each step for presentation slides.

Usage:
    python visualize_preprocessing.py <input_cell_image>

Example:
    python visualize_preprocessing.py data/test_digit1.png
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
import torch
from torchvision import transforms


# ══════════════════════════════════════════════════════════════════════════════
# CNN PREPROCESSING STEPS
# ══════════════════════════════════════════════════════════════════════════════

def cnn_step_0_original(image_path):
    """Step 0: Load original image"""
    image = Image.open(image_path).convert("RGB")
    return image, "0. Original Image"


def cnn_step_1_grayscale(image):
    """Step 1: Convert to grayscale"""
    image = image.convert("L")
    return image, "1. Grayscale"


def cnn_step_2_remove_borders(image):
    """Step 2: Remove grid line bleed from edges"""
    img = np.array(image)
    h, w = img.shape
    border = 4
    if h > border * 3 and w > border * 3:
        img = img[border:h-border, border:w-border]
    image = Image.fromarray(img)
    return image, "2. Remove Grid Borders (4px)"


def cnn_step_3_autocontrast(image):
    """Step 3: Aggressive contrast boost"""
    image = ImageOps.autocontrast(image, cutoff=5)
    return image, "3. Auto-contrast (cutoff=5)"


def cnn_step_4_sharpen(image):
    """Step 4: Sharpen twice for faint pencil marks"""
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.SHARPEN)
    return image, "4. Sharpen 2x"


def cnn_step_5_morph_open(image):
    """Step 5: Remove annotation circles/marks using morphological opening"""
    img = np.array(image)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    image = Image.fromarray(img)
    return image, "5. Morphological Opening\n(removes circles)"


def cnn_step_6_resize_mnist(image):
    """Step 6: Resize to 28x28 for MNIST-trained model"""
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    return image, "6. Resize to 28×28"


def cnn_step_7_normalize(image):
    """Step 7: Normalize using MNIST mean/std (visualization only)"""
    # For visualization, we'll just convert to tensor and back
    # The actual normalization changes pixel values to negative/positive floats
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    tensor = transform(image)
    
    # Convert back to displayable image (denormalize for visualization)
    denorm_tensor = tensor * 0.3081 + 0.1307
    denorm_tensor = torch.clamp(denorm_tensor, 0, 1)
    
    img_array = (denorm_tensor.squeeze().numpy() * 255).astype(np.uint8)
    image = Image.fromarray(img_array, mode='L')
    
    return image, "7. Normalize\n(MNIST mean=0.13, std=0.31)"


def get_cnn_pipeline(image_path):
    """Get all CNN preprocessing steps"""
    steps = []
    
    # Step 0: Original
    img, desc = cnn_step_0_original(image_path)
    steps.append((img, desc))
    
    # Step 1: Grayscale
    img, desc = cnn_step_1_grayscale(img)
    steps.append((img, desc))
    
    # Step 2: Remove borders
    img, desc = cnn_step_2_remove_borders(img)
    steps.append((img, desc))
    
    # Step 3: Auto-contrast
    img, desc = cnn_step_3_autocontrast(img)
    steps.append((img, desc))
    
    # Step 4: Sharpen
    img, desc = cnn_step_4_sharpen(img)
    steps.append((img, desc))
    
    # Step 5: Morphological opening
    img, desc = cnn_step_5_morph_open(img)
    steps.append((img, desc))
    
    # Step 6: Resize to 28x28
    img, desc = cnn_step_6_resize_mnist(img)
    steps.append((img, desc))
    
    # Step 7: Normalize
    img, desc = cnn_step_7_normalize(img)
    steps.append((img, desc))
    
    return steps


# ══════════════════════════════════════════════════════════════════════════════
# TrOCR PREPROCESSING STEPS
# ══════════════════════════════════════════════════════════════════════════════

def trocr_step_0_original(image_path):
    """Step 0: Load original image"""
    image = Image.open(image_path).convert("RGB")
    return image, "0. Original Image"


def trocr_step_1_remove_borders(image):
    """Step 1: Remove grid line bleed from edges"""
    img = np.array(image.convert("L"))
    h, w = img.shape
    border = 4
    if h > border * 3 and w > border * 3:
        img = img[border:h-border, border:w-border]
    image = Image.fromarray(img)
    return image, "1. Remove Grid Borders (4px)"


def trocr_step_2_grayscale(image):
    """Step 2: Convert to grayscale"""
    image = image.convert("L")
    return image, "2. Grayscale"


def trocr_step_3_upscale(image):
    """Step 3: Upscale small crops for better resolution"""
    w, h = image.size
    min_height = 80
    if h < min_height:
        scale = min_height / h
        new_w = max(int(w * scale), min_height * 4)
        image = image.resize((new_w, min_height), Image.Resampling.LANCZOS)
    return image, f"3. Upscale\n(min height=80px)\n{image.size[0]}×{image.size[1]}"


def trocr_step_4_autocontrast(image):
    """Step 4: Auto-contrast with gentler cutoff than CNN"""
    image = ImageOps.autocontrast(image, cutoff=2)
    return image, "4. Auto-contrast (cutoff=2)"


def trocr_step_5_sharpen(image):
    """Step 5: Sharpen twice for blurry crops"""
    image = image.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.SHARPEN)
    return image, "5. Sharpen 2x"


def trocr_step_6_deskew(image):
    """Step 6: Correct slight skew from non-flat scans"""
    img = np.array(image)
    coords = np.column_stack(np.where(img < 128))
    
    if len(coords) < 10:
        return image, "6. Deskew\n(no skew detected)"
    
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    
    if abs(angle) < 0.5:
        return image, f"6. Deskew\n(angle={angle:.2f}° - skipped)"
    
    (h, w) = img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    image = Image.fromarray(rotated)
    
    return image, f"6. Deskew\n(rotated {angle:.2f}°)"


def trocr_step_7_pad(image):
    """Step 7: Pad to consistent landscape aspect ratio"""
    image = ImageOps.pad(image, (384, 96), color=255)
    return image, "7. Pad to 384×96\n(landscape ratio)"


def trocr_step_8_rgb(image):
    """Step 8: Convert back to RGB (TrOCR requires 3 channels)"""
    image = image.convert("RGB")
    return image, "8. Convert to RGB\n(TrOCR input format)"


def get_trocr_pipeline(image_path):
    """Get all TrOCR preprocessing steps"""
    steps = []
    
    # Step 0: Original
    img, desc = trocr_step_0_original(image_path)
    steps.append((img, desc))
    
    # Step 1: Remove borders
    img, desc = trocr_step_1_remove_borders(img)
    steps.append((img, desc))
    
    # Step 2: Grayscale
    img, desc = trocr_step_2_grayscale(img)
    steps.append((img, desc))
    
    # Step 3: Upscale
    img, desc = trocr_step_3_upscale(img)
    steps.append((img, desc))
    
    # Step 4: Auto-contrast
    img, desc = trocr_step_4_autocontrast(img)
    steps.append((img, desc))
    
    # Step 5: Sharpen
    img, desc = trocr_step_5_sharpen(img)
    steps.append((img, desc))
    
    # Step 6: Deskew
    img, desc = trocr_step_6_deskew(img)
    steps.append((img, desc))
    
    # Step 7: Pad
    img, desc = trocr_step_7_pad(img)
    steps.append((img, desc))
    
    # Step 8: RGB
    img, desc = trocr_step_8_rgb(img)
    steps.append((img, desc))
    
    return steps


# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION & SAVING
# ══════════════════════════════════════════════════════════════════════════════

def add_label(image, text, position='bottom'):
    """Add a text label to an image"""
    # Convert to RGB if grayscale
    if image.mode == 'L':
        image = image.convert('RGB')
    
    # Create a copy to draw on
    img_with_label = image.copy()
    draw = ImageDraw.Draw(img_with_label)
    
    # Try to use a nicer font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Add white background for text
    img_width, img_height = img_with_label.size
    
    if position == 'bottom':
        # Add padding at bottom
        new_height = img_height + text_height + 20
        new_img = Image.new('RGB', (img_width, new_height), 'white')
        new_img.paste(img_with_label, (0, 0))
        draw = ImageDraw.Draw(new_img)
        text_x = (img_width - text_width) // 2
        text_y = img_height + 10
    else:  # top
        new_height = img_height + text_height + 20
        new_img = Image.new('RGB', (img_width, new_height), 'white')
        new_img.paste(img_with_label, (0, text_height + 20))
        draw = ImageDraw.Draw(new_img)
        text_x = (img_width - text_width) // 2
        text_y = 10
    
    # Draw text
    draw.text((text_x, text_y), text, fill='black', font=font)
    
    return new_img


def save_pipeline_steps(steps, output_dir, prefix):
    """Save each step as a separate image"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    
    for i, (img, desc) in enumerate(steps):
        # Upscale small images for visibility
        if max(img.size) < 200:
            scale = 200 / max(img.size)
            new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
            img = img.resize(new_size, Image.Resampling.NEAREST)
        
        # Add label
        img_labeled = add_label(img, desc, position='bottom')
        
        # Save
        filename = f"{prefix}_step{i}_{desc.split('.')[1].split()[0].lower()}.png"
        filepath = output_dir / filename
        img_labeled.save(filepath)
        saved_files.append(filepath)
        
        print(f"  ✓ Saved: {filename}")
    
    return saved_files


def create_comparison_grid(cnn_steps, trocr_steps, output_path):
    """Create a side-by-side comparison image of all steps"""
    
    # Upscale and prepare images
    max_steps = max(len(cnn_steps), len(trocr_steps))
    
    cell_width = 250
    cell_height = 200
    
    # Create large canvas
    margin = 20
    header_height = 50
    total_width = 2 * cell_width + 3 * margin
    total_height = header_height + max_steps * (cell_height + margin) + margin
    
    canvas = Image.new('RGB', (total_width, total_height), 'white')
    draw = ImageDraw.Draw(canvas)
    
    # Try to load a font
    try:
        title_font = ImageFont.truetype("arial.ttf", 20)
        step_font = ImageFont.truetype("arial.ttf", 12)
    except:
        title_font = ImageFont.load_default()
        step_font = ImageFont.load_default()
    
    # Draw headers
    draw.text((margin + cell_width//2 - 30, 15), "CNN Pipeline", fill='black', font=title_font)
    draw.text((2*margin + cell_width + cell_width//2 - 50, 15), "TrOCR Pipeline", fill='black', font=title_font)
    
    # Place each step
    for i in range(max_steps):
        y_offset = header_height + i * (cell_height + margin)
        
        # CNN step
        if i < len(cnn_steps):
            img, desc = cnn_steps[i]
            # Resize to fit cell
            img = img.convert('RGB')
            img.thumbnail((cell_width - 20, cell_height - 40), Image.Resampling.LANCZOS)
            
            # Center in cell
            x = margin + (cell_width - img.width) // 2
            y = y_offset + (cell_height - img.height) // 2
            canvas.paste(img, (x, y))
            
            # Draw step label
            draw.text((margin + 10, y_offset + 10), desc, fill='blue', font=step_font)
        
        # TrOCR step
        if i < len(trocr_steps):
            img, desc = trocr_steps[i]
            # Resize to fit cell
            img = img.convert('RGB')
            img.thumbnail((cell_width - 20, cell_height - 40), Image.Resampling.LANCZOS)
            
            # Center in cell
            x = 2 * margin + cell_width + (cell_width - img.width) // 2
            y = y_offset + (cell_height - img.height) // 2
            canvas.paste(img, (x, y))
            
            # Draw step label
            draw.text((2*margin + cell_width + 10, y_offset + 10), desc, fill='green', font=step_font)
        
        # Draw separator line
        if i < max_steps - 1:
            line_y = y_offset + cell_height + margin // 2
            draw.line([(margin, line_y), (total_width - margin, line_y)], fill='lightgray', width=1)
    
    canvas.save(output_path)
    print(f"\n  ✓ Saved comparison grid: {output_path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_preprocessing.py <input_image>")
        print("\nExample:")
        print("  python visualize_preprocessing.py data/test_digit1.png")
        sys.exit(1)
    
    input_path = sys.argv[1]
    
    if not Path(input_path).exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path("preprocessing_visualization")
    stem = Path(input_path).stem
    
    print(f"\n{'='*70}")
    print(f"Preprocessing Visualization: {input_path}")
    print(f"{'='*70}\n")
    
    # Get CNN pipeline steps
    print("CNN Preprocessing Pipeline:")
    cnn_steps = get_cnn_pipeline(input_path)
    cnn_files = save_pipeline_steps(cnn_steps, output_dir / "cnn", f"cnn_{stem}")
    
    print(f"\nTrOCR Preprocessing Pipeline:")
    trocr_steps = get_trocr_pipeline(input_path)
    trocr_files = save_pipeline_steps(trocr_steps, output_dir / "trocr", f"trocr_{stem}")
    
    # Create comparison grid
    print(f"\nCreating comparison grid...")
    comparison_path = output_dir / f"comparison_{stem}.png"
    create_comparison_grid(cnn_steps, trocr_steps, comparison_path)
    
    print(f"\n{'='*70}")
    print(f"✓ All visualizations saved to: {output_dir}/")
    print(f"{'='*70}\n")
    print(f"Individual steps:")
    print(f"  - CNN steps:    {output_dir}/cnn/")
    print(f"  - TrOCR steps:  {output_dir}/trocr/")
    print(f"\nComparison grid:")
    print(f"  - {comparison_path}")
    print()


if __name__ == "__main__":
    main()