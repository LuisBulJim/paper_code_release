"""
metrics_coco.py -- verbatim copy of the COCO-style metric used in the released code
(LuisBulJim/paper_code_release, src/utils_research.py: compute_ap_coco,
compute_metrics_coco, compute_iou_matrix), so that the new protocol is scored with the
same evaluator as the original tables.

Behaviour to keep in mind (unchanged on purpose, for comparability):
  * Precision/Recall/F1 are computed from EVERY prediction passed in, so the confidence
    threshold is applied by the caller before calling.
  * AP is computed from the same predictions. If they were pre-filtered at conf=0.25,
    the PR curve is truncated and AP cannot exceed the recall reached at that threshold.
  * Matching is greedy per image using only the best-IoU ground truth of each prediction.
"""
from typing import Any, Dict, List, Tuple

import numpy as np

def compute_ap_coco(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """
    Compute Average Precision using COCO's 101-point interpolation method.
    
    This implementation matches pycocotools' AP calculation, which is what
    Ultralytics uses internally. Using this ensures metric alignment between
    training logs (Ultralytics' mAP50) and our custom evaluation.
    
    The COCO method samples precision at 101 recall points [0.0, 0.01, ..., 1.0]
    and computes the mean of these interpolated precision values.
    
    Args:
        recalls: Array of recall values (sorted ascending)
        precisions: Array of precision values corresponding to each recall
    
    Returns:
        ap: Average Precision value in [0, 1]
    
    Reference:
        https://cocodataset.org/#detection-eval
        pycocotools/cocoeval.py
    """
    # Handle edge cases
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    
    # Prepend sentinel values
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))
    
    # Make precision monotonically decreasing (right to left)
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    
    # COCO 101-point interpolation
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    
    # Interpolate precision at each recall threshold
    precision_at_recall = np.zeros(101)
    for i, r_thr in enumerate(recall_thresholds):
        # Find precision at this recall threshold (max precision where recall >= r_thr)
        mask = recalls >= r_thr
        if mask.any():
            precision_at_recall[i] = precisions[mask].max()
        else:
            precision_at_recall[i] = 0.0
    
    # AP is the mean of interpolated precisions
    ap = precision_at_recall.mean()
    
    return float(ap)


def compute_metrics_coco(
    predictions: List[Dict[str, Any]],
    ground_truths: List[Dict[str, Any]],
    iou_threshold: float = 0.5
) -> Tuple[float, float, float, float]:
    """
    Compute COCO-compatible mAP@50 and F1 score for a set of predictions.
    
    This function matches boxes between predictions and ground truth using
    the specified IoU threshold, then computes precision-recall curve and
    uses compute_ap_coco for the final AP calculation.
    
    Args:
        predictions: List of prediction dicts with 'boxes', 'scores', 'labels'
                    Each box is [x1, y1, x2, y2] in pixel coordinates
        ground_truths: List of ground truth dicts with 'boxes', 'labels'
        iou_threshold: IoU threshold for matching (default: 0.5)
    
    Returns:
        ap: Average Precision at the specified IoU threshold
        precision: Overall precision
        recall: Overall recall
        f1: F1 score
    """
    all_scores = []
    all_tp = []
    total_gt = 0
    
    for pred, gt in zip(predictions, ground_truths):
        pred_boxes = pred.get('boxes', np.array([]))
        pred_scores = pred.get('scores', np.array([]))
        gt_boxes = gt.get('boxes', np.array([]))
        
        if len(gt_boxes) > 0:
            total_gt += len(gt_boxes)
        
        if len(pred_boxes) == 0:
            continue
            
        if len(gt_boxes) == 0:
            # All predictions are false positives
            all_scores.extend(pred_scores.tolist())
            all_tp.extend([False] * len(pred_boxes))
            continue
        
        # Compute IoU matrix
        iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)
        
        # Match predictions to ground truth (greedy matching)
        gt_matched = set()
        
        # Sort predictions by score (descending)
        sorted_indices = np.argsort(-pred_scores)
        
        for idx in sorted_indices:
            score = pred_scores[idx]
            all_scores.append(score)
            
            # Find best matching GT
            ious = iou_matrix[idx]
            best_gt_idx = np.argmax(ious)
            best_iou = ious[best_gt_idx]
            
            if best_iou >= iou_threshold and best_gt_idx not in gt_matched:
                all_tp.append(True)
                gt_matched.add(best_gt_idx)
            else:
                all_tp.append(False)
    
    if len(all_scores) == 0 or total_gt == 0:
        return 0.0, 0.0, 0.0, 0.0
    
    # Sort by score (descending)
    sorted_indices = np.argsort(-np.array(all_scores))
    all_tp = np.array(all_tp)[sorted_indices]
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(~all_tp)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / total_gt
    
    # Compute AP using COCO method
    ap = compute_ap_coco(recalls, precisions)
    
    # Compute overall precision, recall, F1
    total_tp = int(all_tp.sum())
    total_fp = len(all_tp) - total_tp
    
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / total_gt if total_gt > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return ap, precision, recall, f1


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: Array of shape (N, 4) with [x1, y1, x2, y2] format
        boxes2: Array of shape (M, 4) with [x1, y1, x2, y2] format
    
    Returns:
        iou_matrix: Array of shape (N, M) with IoU values
    """
    boxes1 = np.asarray(boxes1)
    boxes2 = np.asarray(boxes2)
    
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    
    # Ensure 2D
    if boxes1.ndim == 1:
        boxes1 = boxes1.reshape(1, -1)
    if boxes2.ndim == 1:
        boxes2 = boxes2.reshape(1, -1)
    
    # Compute intersection
    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3].T)
    
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Compute union
    union = area1[:, np.newaxis] + area2 - intersection
    
    # Compute IoU
    iou = intersection / (union + 1e-6)
    
    return iou
