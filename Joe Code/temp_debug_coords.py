import cv2
import numpy as np
from pathlib import Path
from eval_and_reconstruct import load_predictions_for_image, cells_to_grid_lines, reconstruct_from_cells, mask_to_line_positions

image_path = 'images/clean1_page0.png'
annotations_path = 'training_data/annotations.json'
h_mask_path = 'masks/clean1_h.png'
v_mask_path = 'masks/clean1_v.png'

img_bgr = cv2.imread(image_path)
image_filename = Path(image_path).name
cells = load_predictions_for_image(annotations_path, image_filename)
pred_h, pred_v = cells_to_grid_lines(cells)
recon = reconstruct_from_cells(img_bgr, pred_h, pred_v)
print('recon shape', recon.shape)
grid_h, grid_w = recon.shape[:2]
gt_h = mask_to_line_positions(h_mask_path, 'h', target_height=grid_h, target_width=grid_w)
gt_v = mask_to_line_positions(v_mask_path, 'v', target_height=grid_h, target_width=grid_w)
pred_h_local = [y - pred_h[0] for y in pred_h]
pred_v_local = [x - pred_v[0] for x in pred_v]
print('pred_h_local len', len(pred_h_local), 'range', min(pred_h_local), max(pred_h_local))
print('pred_v_local len', len(pred_v_local), 'range', min(pred_v_local), max(pred_v_local))
print('gt_h len', len(gt_h), 'range', min(gt_h), max(gt_h))
print('gt_v len', len(gt_v), 'range', min(gt_v), max(gt_v))
print('grid_h,w', grid_h, grid_w)
