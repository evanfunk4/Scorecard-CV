from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from scorecard_corner_label_tool import export_labels
from scorecard_segmentation_extraction import (
    GridDecoder,
    InferConfig,
    TrainConfig,
    _derive_junction_mask,
    _load_model,
    _predict_maps,
    extract_cells_from_decoded,
    train_model,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_label_jsons(labels_dir: Path) -> list[Path]:
    out: list[Path] = []
    for js in sorted(labels_dir.glob("*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and all(k in data for k in ("image", "table_mask", "v_mask", "h_mask")):
            out.append(js)
    return out


def _copy_subset_jsons(all_jsons: list[Path], holdout_stem: str, out_dir: Path) -> list[Path]:
    _ensure_dir(out_dir)
    kept: list[Path] = []
    for js in all_jsons:
        if js.stem == holdout_stem:
            continue
        dst = out_dir / js.name
        shutil.copy2(js, dst)
        kept.append(dst)
    return kept


def _binarize(mask: np.ndarray) -> np.ndarray:
    return (mask > 127).astype(np.uint8)


def _metrics(pred_u8: np.ndarray, gt_u8: np.ndarray) -> dict[str, float]:
    pred = pred_u8.astype(bool)
    gt = gt_u8.astype(bool)
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())
    tn = int(np.logical_and(np.logical_not(pred), np.logical_not(gt)).sum())
    iou = tp / float(max(1, tp + fp + fn))
    dice = (2 * tp) / float(max(1, 2 * tp + fp + fn))
    prec = tp / float(max(1, tp + fp))
    rec = tp / float(max(1, tp + fn))
    acc = (tp + tn) / float(max(1, tp + tn + fp + fn))
    return {
        "iou": float(iou),
        "dice": float(dice),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(acc),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _write_prob(path: Path, prob: np.ndarray) -> None:
    cv2.imwrite(str(path), np.clip(prob * 255.0, 0, 255).astype(np.uint8))


def _infer_on_exported_image(
    weights: Path,
    sample_json: Path,
    out_dir: Path,
    infer_cfg: InferConfig,
    device: str,
) -> dict:
    _ensure_dir(out_dir)
    rec = json.loads(sample_json.read_text(encoding="utf-8"))
    img = cv2.imread(str(rec["image"]), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to load image from {sample_json}")

    model, torch, F, _ = _load_model(weights=weights, device=device)
    table_prob, v_prob, h_prob, j_prob = _predict_maps(model, torch, F, img, device=device)

    _write_prob(out_dir / "debug_table_prob.png", table_prob)
    _write_prob(out_dir / "debug_v_prob.png", v_prob)
    _write_prob(out_dir / "debug_h_prob.png", h_prob)
    _write_prob(out_dir / "debug_junction_prob.png", j_prob)
    cv2.imwrite(str(out_dir / "debug_input_image.png"), img)

    decoder = GridDecoder(infer_cfg)
    decoded = decoder.decode_all(table_prob, v_prob, h_prob, j_prob)

    overlay = img.copy()
    for dec in decoded:
        x0, y0, x1, y1 = dec.bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 0), 2)
        for x in dec.x_lines:
            cv2.line(overlay, (x, y0), (x, y1), (255, 0, 0), 1)
        for y in dec.y_lines:
            cv2.line(overlay, (x0, y), (x1, y), (0, 0, 255), 1)
    cv2.imwrite(str(out_dir / "debug_grid_overlay.png"), overlay)

    table_results = []
    for dec in decoded:
        tdir = out_dir / f"table_{dec.table_id:02d}" / "cells"
        table_results.append(extract_cells_from_decoded(img, dec, tdir, infer_cfg))

    payload = {
        "image": rec["image"],
        "table_count": len(table_results),
        "tables": table_results,
    }
    (out_dir / "image_index.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    gt_table = cv2.imread(str(rec["table_mask"]), cv2.IMREAD_GRAYSCALE)
    gt_v = cv2.imread(str(rec["v_mask"]), cv2.IMREAD_GRAYSCALE)
    gt_h = cv2.imread(str(rec["h_mask"]), cv2.IMREAD_GRAYSCALE)
    if gt_table is None or gt_v is None or gt_h is None:
        raise RuntimeError(f"Failed to load GT masks for {sample_json}")
    gt_j = _derive_junction_mask(gt_v, gt_h, dilate_px=4)

    if gt_table.shape != table_prob.shape:
        raise RuntimeError(
            f"Shape mismatch for {sample_json.name}: "
            f"pred={table_prob.shape} gt={gt_table.shape}. "
            "Use exported labels/images from the same preprocessing path."
        )

    pred_table = (table_prob >= infer_cfg.table_thresh).astype(np.uint8) * 255
    pred_v = (v_prob >= infer_cfg.line_thresh).astype(np.uint8) * 255
    pred_h = (h_prob >= infer_cfg.line_thresh).astype(np.uint8) * 255
    pred_j = (j_prob >= infer_cfg.junction_thresh).astype(np.uint8) * 255

    m = {
        "table": _metrics(_binarize(pred_table), _binarize(gt_table)),
        "v": _metrics(_binarize(pred_v), _binarize(gt_v)),
        "h": _metrics(_binarize(pred_h), _binarize(gt_h)),
        "junction": _metrics(_binarize(pred_j), _binarize(gt_j)),
    }
    return {
        "metrics": m,
        "table_count_pred": int(len(decoded)),
        "rows_cols_pred": [
            {"table_id": int(t["table_id"]), "rows_base": int(t["rows_base"]), "cols_base": int(t["cols_base"])}
            for t in table_results
        ],
    }


def _mean(vals: list[float]) -> float:
    return float(sum(vals) / max(1, len(vals)))


def run_leave_one_out(
    intersection_labels_dir: Path,
    exported_labels_dir: Path,
    work_dir: Path,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    train_size: int,
    val_ratio: float,
    refresh_export: bool,
    max_folds: int,
    include_stems: list[str],
    infer_only: bool,
    pretrain_images_dir: Path | None,
    pretrain_epochs: int,
    pretrain_batch_size: int,
    pretrain_lr: float,
    pretrain_weight_decay: float,
    pretrain_max_images: int,
) -> None:
    if refresh_export or not exported_labels_dir.exists():
        _ensure_dir(exported_labels_dir)
        export_labels(
            labels_dir=intersection_labels_dir,
            out_dir=exported_labels_dir,
            line_thickness=3,
            cell_inset=2,
        )

    all_jsons = _collect_label_jsons(exported_labels_dir)
    if len(all_jsons) < 2:
        raise RuntimeError(f"Need at least 2 exported label JSONs in {exported_labels_dir}")

    if include_stems:
        include = set(include_stems)
        all_jsons = [js for js in all_jsons if js.stem in include]
        if len(all_jsons) < 2:
            raise RuntimeError(
                f"After --include_stems filter, need at least 2 samples. Got {len(all_jsons)}."
            )
    if max_folds > 0:
        all_jsons = all_jsons[: int(max_folds)]

    _ensure_dir(work_dir)
    folds_dir = work_dir / "folds"
    _ensure_dir(folds_dir)

    infer_cfg = InferConfig()
    fold_rows: list[dict[str, object]] = []

    for holdout in all_jsons:
        stem = holdout.stem
        print(f"\n=== LOO fold: holdout={stem} ===")
        fold_dir = folds_dir / stem
        train_json_dir = fold_dir / "train_labels_jsons"
        infer_out = fold_dir / "infer"
        _ensure_dir(fold_dir)
        if train_json_dir.exists():
            shutil.rmtree(train_json_dir, ignore_errors=True)
        _ensure_dir(train_json_dir)

        copied = _copy_subset_jsons(all_jsons, holdout_stem=stem, out_dir=train_json_dir)
        print(f"train_samples={len(copied)} holdout=1")

        weights = fold_dir / "weights.pt"
        if not infer_only:
            cfg = TrainConfig(
                labels_dir=train_json_dir,
                out_weights=weights,
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                weight_decay=float(weight_decay),
                train_size=int(train_size),
                val_ratio=float(val_ratio),
                seed=7,
                junction_dilate_px=4,
                num_workers=0,
                pretrain_images_dir=pretrain_images_dir,
                pretrain_epochs=int(pretrain_epochs),
                pretrain_batch_size=int(pretrain_batch_size),
                pretrain_lr=float(pretrain_lr),
                pretrain_weight_decay=float(pretrain_weight_decay),
                pretrain_max_images=int(pretrain_max_images),
            )
            train_model(cfg)
        else:
            if not weights.exists():
                raise RuntimeError(
                    f"--infer_only requested but weights missing for fold '{stem}': {weights}"
                )

        eval_payload = _infer_on_exported_image(
            weights=weights,
            sample_json=holdout,
            out_dir=infer_out,
            infer_cfg=infer_cfg,
            device=device,
        )

        m = eval_payload["metrics"]
        row = {
            "holdout": stem,
            "table_iou": float(m["table"]["iou"]),
            "table_dice": float(m["table"]["dice"]),
            "v_iou": float(m["v"]["iou"]),
            "v_dice": float(m["v"]["dice"]),
            "h_iou": float(m["h"]["iou"]),
            "h_dice": float(m["h"]["dice"]),
            "j_iou": float(m["junction"]["iou"]),
            "j_dice": float(m["junction"]["dice"]),
            "pred_table_count": int(eval_payload["table_count_pred"]),
        }
        fold_rows.append(row)
        (fold_dir / "eval_metrics.json").write_text(json.dumps(eval_payload, indent=2), encoding="utf-8")
        print(
            f"metrics table_dice={row['table_dice']:.4f} v_dice={row['v_dice']:.4f} "
            f"h_dice={row['h_dice']:.4f} j_dice={row['j_dice']:.4f}"
        )

    summary = {
        "n_folds": len(fold_rows),
        "mean_table_iou": _mean([float(r["table_iou"]) for r in fold_rows]),
        "mean_table_dice": _mean([float(r["table_dice"]) for r in fold_rows]),
        "mean_v_iou": _mean([float(r["v_iou"]) for r in fold_rows]),
        "mean_v_dice": _mean([float(r["v_dice"]) for r in fold_rows]),
        "mean_h_iou": _mean([float(r["h_iou"]) for r in fold_rows]),
        "mean_h_dice": _mean([float(r["h_dice"]) for r in fold_rows]),
        "mean_j_iou": _mean([float(r["j_iou"]) for r in fold_rows]),
        "mean_j_dice": _mean([float(r["j_dice"]) for r in fold_rows]),
        "folds": fold_rows,
        "train_config": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "train_size": int(train_size),
            "val_ratio": float(val_ratio),
            "device": str(device),
            "pretrain_images_dir": str(pretrain_images_dir) if pretrain_images_dir is not None else "",
            "pretrain_epochs": int(pretrain_epochs),
            "pretrain_batch_size": int(pretrain_batch_size),
            "pretrain_lr": float(pretrain_lr),
            "pretrain_weight_decay": float(pretrain_weight_decay),
            "pretrain_max_images": int(pretrain_max_images),
        },
        "infer_config": asdict(infer_cfg),
    }

    (work_dir / "loo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (work_dir / "loo_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "holdout",
                "table_iou",
                "table_dice",
                "v_iou",
                "v_dice",
                "h_iou",
                "h_dice",
                "j_iou",
                "j_dice",
                "pred_table_count",
            ],
        )
        writer.writeheader()
        writer.writerows(fold_rows)

    print("\n=== LOO done ===")
    print(f"summary: {work_dir / 'loo_summary.json'}")
    print(
        f"mean dice table={summary['mean_table_dice']:.4f} "
        f"v={summary['mean_v_dice']:.4f} h={summary['mean_h_dice']:.4f} j={summary['mean_j_dice']:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Leave-one-out training/evaluation for scorecard segmentation")
    p.add_argument("--intersection_labels_dir", default="ml_intersection_labels")
    p.add_argument("--exported_labels_dir", default="ml_intersection_labels_exported")
    p.add_argument("--work_dir", default="loo_seg_results")
    p.add_argument("--device", default="cpu")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--train_size", type=int, default=768)
    p.add_argument("--val_ratio", type=float, default=0.12)
    p.add_argument("--refresh_export", action="store_true")
    p.add_argument("--max_folds", type=int, default=0, help="0 means all folds")
    p.add_argument(
        "--include_stems",
        nargs="*",
        default=None,
        help="Optional subset of sample stems (e.g., clean1 clean2 ...)",
    )
    p.add_argument(
        "--infer_only",
        action="store_true",
        help="Skip training and run inference/evaluation using existing fold weights.",
    )
    p.add_argument("--pretrain_images_dir", default="")
    p.add_argument("--pretrain_epochs", type=int, default=0)
    p.add_argument("--pretrain_batch_size", type=int, default=4)
    p.add_argument("--pretrain_lr", type=float, default=3e-4)
    p.add_argument("--pretrain_weight_decay", type=float, default=1e-5)
    p.add_argument("--pretrain_max_images", type=int, default=0, help="0 means all pretraining images")
    args = p.parse_args()

    pretrain_images_dir = Path(args.pretrain_images_dir) if args.pretrain_images_dir else None

    run_leave_one_out(
        intersection_labels_dir=Path(args.intersection_labels_dir),
        exported_labels_dir=Path(args.exported_labels_dir),
        work_dir=Path(args.work_dir),
        device=str(args.device),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        train_size=int(args.train_size),
        val_ratio=float(args.val_ratio),
        refresh_export=bool(args.refresh_export),
        max_folds=int(args.max_folds),
        include_stems=list(args.include_stems or []),
        infer_only=bool(args.infer_only),
        pretrain_images_dir=pretrain_images_dir,
        pretrain_epochs=int(args.pretrain_epochs),
        pretrain_batch_size=int(args.pretrain_batch_size),
        pretrain_lr=float(args.pretrain_lr),
        pretrain_weight_decay=float(args.pretrain_weight_decay),
        pretrain_max_images=int(args.pretrain_max_images),
    )


if __name__ == "__main__":
    main()
