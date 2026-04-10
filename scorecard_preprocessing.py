"""Preprocessing for golf scorecard grid extraction by Hough transform.

This module focuses on improving grid-line visibility under typical
scorecard conditions:
- variable card size/aspect ratio
- textured or noisy backgrounds
- booklet folds and shadows
- weakly printed table lines
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import subprocess


@dataclass
class PreprocessConfig:
    """Configuration for scorecard preprocessing."""

    target_long_edge: int = 1800
    ensure_upright: bool = True
    upright_use_ocr: bool = True
    upright_use_osd: bool = True
    upright_osd_min_confidence: float = 2.0
    upright_score_long_edge: int = 1200
    upright_landscape_bonus: float = 2.0
    upright_portrait_penalty: float = 5.0
    upright_force_landscape_for_portrait_input: bool = True
    upright_portrait_aspect_trigger: float = 1.06
    upright_grid_score_weight: float = 2.8
    deskew: bool = True
    max_skew_degrees: float = 8.0
    clahe_clip_limit: float = 2.5
    clahe_grid_size: int = 8
    adaptive_block_size: int = 41
    adaptive_c: int = 9
    line_boost_scale: float = 0.03
    line_boost_scale_small: float = 0.012
    edge_rescue_enabled: bool = True
    edge_rescue_canny_low: int = 40
    edge_rescue_canny_high: int = 130
    remove_center_fold: bool = True


@dataclass
class PreprocessResult:
    """Intermediate products passed into line extraction."""

    image_bgr: np.ndarray
    gray: np.ndarray
    binary_inv: np.ndarray
    edge: np.ndarray
    line_mask: np.ndarray
    scale_factor: float
    rotation_degrees: float
    upright_rotation_degrees: int = 0


def load_image(image_path: str | Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to load image: {image_path}")
    return image


def convert_pdf_to_png(
    pdf_path: str | Path,
    output_dir: str | Path,
    dpi: int = 300,
    page_number: Optional[int] = None,
    prefix: str = "scorecard_page",
) -> list[Path]:
    """Convert a PDF (single or multi-page) into PNG image(s).

    Args:
        pdf_path: Path to input PDF file.
        output_dir: Directory where PNG files will be written.
        dpi: Rasterization DPI.
        page_number: Optional 1-based page index. If None, convert all pages.
        prefix: Prefix for generated PNG names.

    Returns:
        List of written PNG paths in page order.
    """

    src = Path(pdf_path)
    if not src.exists():
        raise FileNotFoundError(f"PDF not found: {src}")
    if src.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF input, got: {src}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    first_page = page_number if page_number is not None else None
    last_page = page_number if page_number is not None else None

    # Preferred path: pdf2image if available.
    try:
        from pdf2image import convert_from_path

        pages = convert_from_path(
            str(src),
            dpi=dpi,
            first_page=first_page,
            last_page=last_page,
        )
        written: list[Path] = []
        start_idx = page_number if page_number is not None else 1
        for i, page_img in enumerate(pages, start=start_idx):
            out_path = out_dir / f"{prefix}_{i:03d}.png"
            page_img.save(str(out_path), format="PNG")
            written.append(out_path)
        if written:
            return written
    except ImportError:
        pass

    # Fallback path: poppler's pdftoppm command if installed.
    base = out_dir / prefix
    cmd = ["pdftoppm", "-png", "-r", str(dpi)]
    if page_number is not None:
        cmd.extend(["-f", str(page_number), "-l", str(page_number), "-singlefile"])
    cmd.extend([str(src), str(base)])
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "PDF conversion requires either `pdf2image` (Python package) "
            "or `pdftoppm` (poppler). Neither is available."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"pdftoppm failed while converting {src}: {exc.stderr.strip()}"
        ) from exc

    if page_number is not None:
        out = out_dir / f"{prefix}.png"
        if not out.exists():
            raise RuntimeError("PDF conversion did not produce an output PNG.")
        renamed = out_dir / f"{prefix}_{page_number:03d}.png"
        out.replace(renamed)
        return [renamed]

    written = sorted(out_dir.glob(f"{prefix}-*.png"))
    if not written:
        raise RuntimeError("PDF conversion did not produce any PNG pages.")

    normalized: list[Path] = []
    for idx, old in enumerate(written, start=1):
        new = out_dir / f"{prefix}_{idx:03d}.png"
        if old != new:
            old.replace(new)
        normalized.append(new)
    return normalized


def _resize_long_edge(image: np.ndarray, target_long_edge: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    long_edge = max(h, w)
    if long_edge <= target_long_edge:
        return image, 1.0
    scale = target_long_edge / float(long_edge)
    resized = cv2.resize(
        image,
        (int(round(w * scale)), int(round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _estimate_skew(gray: np.ndarray, max_skew_degrees: float) -> float:
    edge = cv2.Canny(gray, 60, 180)
    lines = cv2.HoughLinesP(
        edge,
        rho=1,
        theta=np.pi / 180.0,
        threshold=120,
        minLineLength=int(0.3 * max(gray.shape[:2])),
        maxLineGap=20,
    )
    if lines is None:
        return 0.0

    offsets = []
    for ln in lines[:, 0]:
        x1, y1, x2, y2 = ln
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0 and dy == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Normalize to [-90, 90)
        angle = ((angle + 90.0) % 180.0) - 90.0
        # Keep near-horizontal and near-vertical lines.
        if abs(angle) <= 25.0:
            offsets.append(angle)
        elif abs(abs(angle) - 90.0) <= 25.0:
            offsets.append(angle - np.sign(angle) * 90.0)

    if not offsets:
        return 0.0

    skew = float(np.median(offsets))
    if abs(skew) > max_skew_degrees:
        return 0.0
    return skew


def _rotate_image(image: np.ndarray, degrees: float) -> np.ndarray:
    if abs(degrees) < 0.05:
        return image
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, degrees, 1.0)
    return cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rotate_90_clockwise_k(image: np.ndarray, k: int) -> np.ndarray:
    k = int(k) % 4
    if k == 0:
        return image
    if k == 1:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if k == 2:
        return cv2.rotate(image, cv2.ROTATE_180)
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def _text_like_mask(gray: np.ndarray) -> np.ndarray:
    """Extract likely text/number strokes while suppressing long table lines."""

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    blk = max(31, ((min(gray.shape[:2]) // 12) | 1))
    binary_inv = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blk,
        9,
    )
    h, w = binary_inv.shape[:2]
    hk = max(21, int(round(0.05 * w)))
    vk = max(21, int(round(0.05 * h)))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk))
    long_h = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, h_kernel, iterations=1)
    long_v = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, v_kernel, iterations=1)
    long_lines = cv2.bitwise_or(long_h, long_v)
    text = cv2.bitwise_and(binary_inv, cv2.bitwise_not(long_lines))
    text = cv2.morphologyEx(
        text, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    return text


def _smooth_1d(signal: np.ndarray, window: int) -> np.ndarray:
    window = max(3, int(window) | 1)
    if signal.size < window:
        return signal.astype(np.float32, copy=True)
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(signal.astype(np.float32), kernel, mode="same")


def _count_projection_peaks(
    proj: np.ndarray, rel_thresh: float = 0.18, min_gap: int = 8
) -> int:
    if proj.size < 3:
        return 0
    sm = _smooth_1d(proj.astype(np.float32), max(5, int(proj.size * 0.015) | 1))
    mx = float(sm.max()) if sm.size else 0.0
    if mx <= 0:
        return 0
    thr = rel_thresh * mx
    peaks: list[int] = []
    for i in range(1, sm.size - 1):
        if sm[i] < thr:
            continue
        if sm[i] >= sm[i - 1] and sm[i] >= sm[i + 1]:
            if not peaks or (i - peaks[-1] >= min_gap):
                peaks.append(i)
            elif sm[i] > sm[peaks[-1]]:
                peaks[-1] = i
    return len(peaks)


def _grid_orientation_score(gray: np.ndarray) -> float:
    """Score how plausible this orientation is for a scorecard grid layout."""

    h, w = gray.shape[:2]
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    block = max(31, ((min(h, w) // 12) | 1))
    bin_inv = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        9,
    )
    hk = max(15, int(round(0.03 * w)))
    vk = max(15, int(round(0.03 * h)))
    h_lines = cv2.morphologyEx(
        bin_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (hk, 1)), iterations=1
    )
    v_lines = cv2.morphologyEx(
        bin_inv, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, vk)), iterations=1
    )
    line_mask = cv2.bitwise_or(h_lines, v_lines)

    n, _, stats, _ = cv2.connectedComponentsWithStats(line_mask, connectivity=8)
    best = None
    best_area = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.01 * (h * w):
            continue
        if bw < 0.20 * w or bh < 0.10 * h:
            continue
        if area > best_area:
            best_area = area
            best = (x, y, bw, bh)

    if best is None:
        return 0.0

    x, y, bw, bh = best
    ar = bw / float(max(1, bh))
    ar_score = 2.2 * np.clip(ar - 1.0, -1.0, 3.0)
    wide_bonus = 1.0 if bw >= bh else -0.8

    crop = line_mask[y : y + bh, x : x + bw]
    x_peaks = _count_projection_peaks(crop.sum(axis=0).astype(np.float32), rel_thresh=0.15, min_gap=max(6, bw // 80))
    y_peaks = _count_projection_peaks(crop.sum(axis=1).astype(np.float32), rel_thresh=0.15, min_gap=max(6, bh // 60))
    # Scorecards usually have more vertical separators than horizontal row separators.
    ratio = (x_peaks + 1.0) / (y_peaks + 1.0)
    ratio_score = 1.4 * np.clip(ratio - 1.0, -1.0, 3.0)

    return float(ar_score + wide_bonus + ratio_score)


def _layout_orientation_score(gray: np.ndarray, landscape_bonus: float) -> float:
    """Heuristic orientation score for golf scorecards when OCR is unavailable."""

    h, w = gray.shape[:2]
    text = _text_like_mask(gray)
    top = float(text[: int(0.35 * h), :].sum()) / 255.0
    bottom = float(text[int(0.65 * h) :, :].sum()) / 255.0
    left = float(text[:, : int(0.35 * w)].sum()) / 255.0
    right = float(text[:, int(0.65 * w) :].sum()) / 255.0

    # Typical scorecards: more header/hole-number text toward top,
    # and metric labels ("Hole/Par/Score...") toward the left.
    top_bias = 3.0 * (top - bottom) / (top + bottom + 1.0)
    left_bias = 2.0 * (left - right) / (left + right + 1.0)

    # Peak position priors further disambiguate 180-degree flips.
    y_proj = text.sum(axis=1).astype(np.float32)
    x_proj = text.sum(axis=0).astype(np.float32)
    if y_proj.size > 0:
        wy = max(9, int(round(h * 0.03)) | 1)
        y_sm = _smooth_1d(y_proj, wy)
        y_peak = float(np.argmax(y_sm))
        y_target = 0.18 * h
        y_peak_bias = 2.5 * (1.0 - min(1.0, abs(y_peak - y_target) / max(1.0, 0.50 * h)))
    else:
        y_peak_bias = 0.0
    if x_proj.size > 0:
        wx = max(9, int(round(w * 0.03)) | 1)
        x_sm = _smooth_1d(x_proj, wx)
        x_peak = float(np.argmax(x_sm))
        x_target = 0.12 * w
        x_peak_bias = 1.8 * (1.0 - min(1.0, abs(x_peak - x_target) / max(1.0, 0.50 * w)))
    else:
        x_peak_bias = 0.0

    landscape = landscape_bonus if w >= h else 0.0
    return top_bias + left_bias + y_peak_bias + x_peak_bias + landscape


def _ocr_readability_score(gray: np.ndarray) -> Optional[float]:
    """Return OCR readability score; None if OCR stack is unavailable."""

    try:
        import pytesseract
    except Exception:
        return None

    try:
        data = pytesseract.image_to_data(
            gray,
            output_type=pytesseract.Output.DICT,
            config="--psm 6",
        )
    except Exception:
        return None

    text_vals = data.get("text", [])
    conf_vals = data.get("conf", [])
    confs: list[float] = []
    n_alnum = 0
    vocab = {
        "hole",
        "par",
        "score",
        "out",
        "in",
        "total",
        "yardage",
        "player",
        "hdcp",
        "handicap",
        "net",
    }
    keyword_hits = 0
    for txt, conf in zip(text_vals, conf_vals):
        token = str(txt).strip()
        if not token:
            continue
        if any(ch.isalnum() for ch in token):
            n_alnum += 1
            t = "".join(ch.lower() for ch in token if ch.isalnum())
            if t in vocab:
                keyword_hits += 1
        try:
            c = float(conf)
        except Exception:
            continue
        if c >= 0:
            confs.append(c)
    mean_conf = (float(np.mean(confs)) / 100.0) if confs else 0.0
    # Higher OCR confidence and more alphanumeric tokens => more likely upright.
    keyword_bonus = 0.9 * min(8, keyword_hits)
    return (8.0 * mean_conf) + (0.25 * min(30, n_alnum)) + keyword_bonus


def _ocr_osd_rotation_quadrants(
    image_bgr: np.ndarray, min_conf: float
) -> Optional[int]:
    """Return clockwise 90-degree steps from Tesseract OSD, if reliable."""

    try:
        import pytesseract
    except Exception:
        return None

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    try:
        osd = pytesseract.image_to_osd(gray)
    except Exception:
        return None

    rotate_deg: Optional[int] = None
    conf_val: Optional[float] = None
    for line in osd.splitlines():
        line = line.strip()
        if line.startswith("Rotate:"):
            try:
                rotate_deg = int(float(line.split(":", 1)[1].strip()))
            except Exception:
                rotate_deg = None
        elif line.startswith("Orientation confidence:"):
            try:
                conf_val = float(line.split(":", 1)[1].strip())
            except Exception:
                conf_val = None

    if rotate_deg is None:
        return None
    if conf_val is not None and conf_val < min_conf:
        return None
    rotate_deg = int(rotate_deg) % 360
    if rotate_deg % 90 != 0:
        return None
    return (rotate_deg // 90) % 4


def _estimate_upright_rotation_quadrants(image_bgr: np.ndarray, cfg: PreprocessConfig) -> int:
    """Return clockwise 90-degree steps to make image upright."""

    probe, _ = _resize_long_edge(image_bgr, cfg.upright_score_long_edge)

    if cfg.upright_use_ocr and cfg.upright_use_osd:
        osd_k = _ocr_osd_rotation_quadrants(probe, cfg.upright_osd_min_confidence)
        if osd_k is not None:
            return osd_k

    candidate_k = (0, 1, 2, 3)
    if (
        cfg.upright_force_landscape_for_portrait_input
        and probe.shape[0] > cfg.upright_portrait_aspect_trigger * probe.shape[1]
    ):
        # Portrait scans in this pipeline are typically sideways scorecards.
        candidate_k = (1, 3)

    best_k = 0
    best_score = -1e9
    for k in candidate_k:
        cand = _rotate_90_clockwise_k(probe, k)
        gray = cv2.cvtColor(cand, cv2.COLOR_BGR2GRAY)
        score = _layout_orientation_score(gray, cfg.upright_landscape_bonus)
        score += cfg.upright_grid_score_weight * _grid_orientation_score(gray)
        if cand.shape[1] < cand.shape[0]:
            score -= cfg.upright_portrait_penalty
        if cfg.upright_use_ocr:
            ocr = _ocr_readability_score(gray)
            if ocr is not None:
                score += ocr
        # Tie-breaker: prefer less rotation when scores are effectively equal.
        if score > best_score + 1e-6:
            best_k = k
            best_score = score
    return best_k


def _illumination_normalize(gray: np.ndarray) -> np.ndarray:
    # Large blur approximates slow-varying background illumination.
    k = max(31, ((min(gray.shape[:2]) // 10) | 1))
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    bg = np.maximum(bg, 1)
    norm = cv2.divide(gray, bg, scale=255)
    return norm


def _remove_single_center_fold(line_mask: np.ndarray) -> np.ndarray:
    """Suppress a dominant fold line that runs through card center."""

    out = line_mask.copy()
    h, w = out.shape[:2]

    # Vertical fold candidate from x-projection.
    proj_x = out.sum(axis=0).astype(np.float32)
    if proj_x.size > 0:
        px = int(np.argmax(proj_x))
        pval = float(proj_x[px])
        med = float(np.median(proj_x))
        if 0.25 * w <= px <= 0.75 * w and pval > 4.0 * max(1.0, med):
            band = max(2, w // 300)
            out[:, max(0, px - band) : min(w, px + band + 1)] = 0

    # Horizontal fold candidate from y-projection.
    proj_y = out.sum(axis=1).astype(np.float32)
    if proj_y.size > 0:
        py = int(np.argmax(proj_y))
        pval = float(proj_y[py])
        med = float(np.median(proj_y))
        if 0.25 * h <= py <= 0.75 * h and pval > 4.0 * max(1.0, med):
            band = max(2, h // 300)
            out[max(0, py - band) : min(h, py + band + 1), :] = 0

    return out


def preprocess_scorecard(
    image: np.ndarray,
    config: Optional[PreprocessConfig] = None,
) -> PreprocessResult:
    """Preprocess a scorecard image for robust Hough line extraction."""

    cfg = config or PreprocessConfig()
    upright_rotation_deg = 0
    oriented = image
    if cfg.ensure_upright:
        # Estimate orientation on original image to avoid sensitivity to
        # intermediate resampling artifacts.
        k = _estimate_upright_rotation_quadrants(image, cfg)
        if k != 0:
            oriented = _rotate_90_clockwise_k(image, k)
            upright_rotation_deg = 90 * k
    resized, scale = _resize_long_edge(oriented, cfg.target_long_edge)

    rotation = 0.0
    if cfg.deskew:
        gray_for_skew = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        rotation = _estimate_skew(gray_for_skew, cfg.max_skew_degrees)
        resized = _rotate_image(resized, -rotation)

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    norm = _illumination_normalize(gray)

    clahe = cv2.createCLAHE(
        clipLimit=cfg.clahe_clip_limit,
        tileGridSize=(cfg.clahe_grid_size, cfg.clahe_grid_size),
    )
    enh = clahe.apply(norm)
    denoise = cv2.bilateralFilter(enh, d=5, sigmaColor=35, sigmaSpace=35)

    block_size = cfg.adaptive_block_size | 1
    binary_inv = cv2.adaptiveThreshold(
        denoise,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        cfg.adaptive_c,
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Strengthen long vertical/horizontal structures to favor table lines.
    h, w = binary_inv.shape[:2]
    kx_long = max(15, int(round(w * cfg.line_boost_scale)))
    ky_long = max(15, int(round(h * cfg.line_boost_scale)))
    kx_small = max(9, int(round(w * cfg.line_boost_scale_small)))
    ky_small = max(9, int(round(h * cfg.line_boost_scale_small)))

    h_kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (kx_long, 1))
    v_kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky_long))
    h_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (kx_small, 1))
    v_kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ky_small))

    h_lines_long = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, h_kernel_long, iterations=1)
    v_lines_long = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, v_kernel_long, iterations=1)
    h_lines_small = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, h_kernel_small, iterations=1)
    v_lines_small = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, v_kernel_small, iterations=1)

    line_mask = cv2.bitwise_or(h_lines_long, v_lines_long)
    line_mask = cv2.bitwise_or(line_mask, h_lines_small)
    line_mask = cv2.bitwise_or(line_mask, v_lines_small)

    if cfg.edge_rescue_enabled:
        # Rescue thin, low-contrast grid lines with directional edge extraction.
        edge_rescue = cv2.Canny(
            denoise,
            int(cfg.edge_rescue_canny_low),
            int(cfg.edge_rescue_canny_high),
        )
        h_edge_long = cv2.morphologyEx(
            edge_rescue, cv2.MORPH_OPEN, h_kernel_long, iterations=1
        )
        v_edge_long = cv2.morphologyEx(
            edge_rescue, cv2.MORPH_OPEN, v_kernel_long, iterations=1
        )
        edge_lines = cv2.bitwise_or(h_edge_long, v_edge_long)
        line_mask = cv2.bitwise_or(line_mask, edge_lines)

    line_mask = cv2.morphologyEx(
        line_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1
    )
    if cfg.remove_center_fold:
        line_mask = _remove_single_center_fold(line_mask)

    edge = cv2.Canny(denoise, 50, 160)

    return PreprocessResult(
        image_bgr=resized,
        gray=gray,
        binary_inv=binary_inv,
        edge=edge,
        line_mask=line_mask,
        scale_factor=scale,
        rotation_degrees=rotation,
        upright_rotation_degrees=upright_rotation_deg,
    )


def _save_debug_outputs(result: PreprocessResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "01_image.png"), result.image_bgr)
    cv2.imwrite(str(out_dir / "02_gray.png"), result.gray)
    cv2.imwrite(str(out_dir / "03_binary_inv.png"), result.binary_inv)
    cv2.imwrite(str(out_dir / "04_edge.png"), result.edge)
    cv2.imwrite(str(out_dir / "05_line_mask.png"), result.line_mask)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a golf scorecard image.")
    parser.add_argument("--input", required=True, help="Path to scorecard image.")
    parser.add_argument(
        "--debug_dir",
        default="preprocess_debug",
        help="Directory for intermediate images.",
    )
    parser.add_argument(
        "--no_upright",
        action="store_true",
        help="Disable automatic upright orientation normalization.",
    )
    args = parser.parse_args()

    img = load_image(args.input)
    result = preprocess_scorecard(
        img,
        PreprocessConfig(ensure_upright=not args.no_upright),
    )
    _save_debug_outputs(result, Path(args.debug_dir))
    print(f"Upright rotation applied: {result.upright_rotation_degrees} degrees clockwise")
    print(f"Deskew correction applied: {result.rotation_degrees:.2f} degrees")
    print(f"Saved preprocessing debug images to: {args.debug_dir}")
