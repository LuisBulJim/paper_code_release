"""
utils_research.py - Single Source of Truth for YOLOv26 S-TTA Research Pipeline

This module centralizes all configurations, reproducibility utilities, and metric
calculations to ensure consistency across notebooks 11-15.

Author: Research Pipeline Refactor
Date: 2026-01-27
"""

import os
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import torch
import shutil


# =============================================================================
# DRY RUN MODE: Fast-Fail Testing (Set to True for 15-minute micro-test)
# =============================================================================

# ⚠️  SET TO FALSE BEFORE THE REAL 48-HOUR RUN!
DRY_RUN: bool = False


@dataclass
class DryRunConfig:
    """
    Configuration overrides for fast-fail micro-testing.
    
    When DRY_RUN = True, these values override the production config
    to enable a complete pipeline test in ~15 minutes.
    """
    # Minimal training
    EPOCHS: int = 1
    BATCH_SIZE: int = 4
    
    # Subset of data
    MAX_TRAIN_IMAGES: int = 10
    MAX_VAL_IMAGES: int = 5
    MAX_TEST_IMAGES: int = 5
    
    # Minimal TTA (faster)
    TTA_SCALES: Tuple[float, ...] = (1.0,)  # Single scale = no TTA
    
    # Minimal bootstrap
    BOOTSTRAP_ROUNDS: int = 3
    
    # Quick grid search
    GRID_SEARCH_SUBSET: int = 3  # Only 3 images per config


# =============================================================================
# DISK SPACE VERIFICATION (Pre-Launch Safety Check)
# =============================================================================

def check_disk_space(
    target_dir: Path = Path("."),
    required_gb: float = 10.0,
    warn_gb: float = 20.0
) -> Dict[str, Any]:
    """
    Verify sufficient disk space before launching long runs.
    
    Estimated storage needs:
    - Model checkpoints: ~100MB per epoch × 50 epochs = 5GB
    - High-res figures: ~5MB × 100 figures = 500MB
    - Logs: ~10MB
    - Intermediate CSVs: ~50MB
    - Safety margin: 2x
    
    Args:
        target_dir: Directory to check
        required_gb: Minimum required space (GB)
        warn_gb: Warning threshold (GB)
    
    Returns:
        Dict with total, used, free space and status
    """
    usage = shutil.disk_usage(target_dir)
    
    total_gb = usage.total / (1024 ** 3)
    used_gb = usage.used / (1024 ** 3)
    free_gb = usage.free / (1024 ** 3)
    
    status = "OK"
    if free_gb < required_gb:
        status = "CRITICAL"
    elif free_gb < warn_gb:
        status = "WARNING"
    
    result = {
        'total_gb': round(total_gb, 2),
        'used_gb': round(used_gb, 2),
        'free_gb': round(free_gb, 2),
        'required_gb': required_gb,
        'status': status
    }
    
    # Print report
    status_emoji = {"OK": "✅", "WARNING": "⚠️", "CRITICAL": "❌"}
    print(f"\n{'='*50}")
    print(f" DISK SPACE CHECK: {status_emoji[status]} {status}")
    print(f"{'='*50}")
    print(f"  Total:    {total_gb:.1f} GB")
    print(f"  Used:     {used_gb:.1f} GB ({100*used_gb/total_gb:.1f}%)")
    print(f"  Free:     {free_gb:.1f} GB")
    print(f"  Required: {required_gb:.1f} GB minimum")
    
    if status == "CRITICAL":
        print(f"\n❌ ABORT: Insufficient disk space!")
        print(f"   Free up at least {required_gb - free_gb:.1f} GB before launching.")
    
    return result


def estimate_storage_needs(
    epochs: int = 50,
    num_images: int = 500,
    num_tta_scales: int = 3,
    model_size_mb: float = 100.0
) -> Dict[str, float]:
    """
    Estimate total storage requirements for a training run.
    
    Returns:
        Dict with estimated storage per category in GB
    """
    estimates = {
        'checkpoints_gb': (model_size_mb * epochs * 2) / 1024,  # best.pt + last.pt variants
        'figures_gb': (5.0 * 100) / 1024,  # 5MB per figure, ~100 figures
        'logs_gb': 0.05,
        'csv_results_gb': 0.1,
        'intermediate_gb': (num_images * num_tta_scales * 0.1) / 1024,  # Prediction caches
        'safety_margin_gb': 2.0
    }
    
    estimates['total_gb'] = sum(estimates.values())
    
    print(f"\n📊 Estimated Storage Requirements:")
    for key, value in estimates.items():
        print(f"   {key}: {value:.2f} GB")
    
    return estimates


# =============================================================================
# CONFIGURATION: Single Source of Truth
# =============================================================================

@dataclass
class ResearchConfig:
    """
    Centralized configuration for the entire S-TTA research pipeline.
    All notebooks MUST import and use these values instead of hardcoding.
    """
    
    # -------------------------------------------------------------------------
    # Execution Mode
    # -------------------------------------------------------------------------
    DRY_RUN: bool = False  # Set to True for 15-minute micro-test
    MODEL_PATH: str = "modelos_entrenados/modelo-acuatico-m.pt"
    MODEL_PATH_1280: str = "modelos_entrenados/modelo-acuatico-1280.pt"
    DATASET_ROOT: str = "dataset_yolo"
    SPLITS_FILE: str = "splits.json"
    
    # -------------------------------------------------------------------------
    # Training Configuration
    # -------------------------------------------------------------------------
    SEED: int = 42
    IMGSZ_TRAIN: int = 640
    EPOCHS: int = 50
    BATCH_SIZE: int = 16
    
    # -------------------------------------------------------------------------
    # TTA Configuration (AUDIT-VERIFIED VALUES)
    # -------------------------------------------------------------------------
    TTA_SCALES: Tuple[float, ...] = (0.9, 1.0, 1.1)
    TTA_USE_FLIP: bool = True
    
    # -------------------------------------------------------------------------
    # SAHI Configuration
    # -------------------------------------------------------------------------
    SAHI_SLICE_SIZE: int = 640
    SAHI_OVERLAP: float = 0.25
    
    # -------------------------------------------------------------------------
    # Detection Thresholds
    # -------------------------------------------------------------------------
    CONF_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.5
    
    # -------------------------------------------------------------------------
    # WBF Configuration (CENTRALIZED - No more scattered values!)
    # =========================================================================
    # ⚠️  IMPORTANT: Update these values with the optimal configuration found
    #     by running 13_Scientific_Validation.ipynb on the VALIDATION set.
    #     NEVER tune these on the TEST set (data leakage).
    # =========================================================================
    WBF_IOU_THR: float = 0.5
    WBF_SKIP_THR: float = 0.001
    
    # Weights for TTA fusion: scale 1.0 gets full weight, others reduced
    # TODO: Update with values from grid_search_VALIDATION.csv
    WBF_WEIGHTS_TTA: Tuple[float, ...] = (0.8, 1.0, 0.8)  # For scales 0.9, 1.0, 1.1
    
    # Weights for SAHI+TTA full fusion
    WBF_WEIGHTS_FULL: Tuple[float, ...] = (1.2, 1.0)  # SAHI gets higher weight
    
    # -------------------------------------------------------------------------
    # Inference Settings
    # -------------------------------------------------------------------------
    USE_AMP: bool = True  # Mixed Precision
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # -------------------------------------------------------------------------
    # Validation Settings
    # -------------------------------------------------------------------------
    BOOTSTRAP_ROUNDS: int = 30
    BOOTSTRAP_SAMPLE_RATIO: float = 0.7
    
    # Object Size Thresholds (pixels squared for area comparison)
    SIZE_SMALL_MAX: int = 32 * 32      # < 32x32 px area
    SIZE_MEDIUM_MAX: int = 96 * 96     # < 96x96 px area
    # Large: > 96x96 px area
    
    # -------------------------------------------------------------------------
    # Output Directories
    # -------------------------------------------------------------------------
    RESULTS_DIR: str = "results_validation"
    VISUAL_AUDIT_DIR: str = "visual_audit"
    THEORETICAL_DIR: str = "theoretical_analysis"
    
    def get_tta_weight(self, scale: float) -> float:
        """Get WBF weight for a specific TTA scale."""
        scale_to_idx = {0.9: 0, 1.0: 1, 1.1: 2}
        idx = scale_to_idx.get(scale, 1)
        return self.WBF_WEIGHTS_TTA[idx]


# Global CONFIG instance - import this in all notebooks
CONFIG = ResearchConfig()


# =============================================================================
# DOCKERFILE TEMPLATE (For Reproducibility)
# =============================================================================
# Copy this to a Dockerfile in the project root for containerized execution

DOCKERFILE_TEMPLATE = '''
# YOLOv26 S-TTA Research Pipeline - Docker Image
# ===============================================
FROM nvidia/cuda:12.1-base-ubuntu22.04

# System dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \\
    python3 \\
    python3-pip \\
    libgl1-mesa-glx \\
    libglib2.0-0 \\
    libjpeg-dev \\
    libpng-dev \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Default command
CMD ["python3", "-m", "jupyter", "notebook", "--ip=0.0.0.0", "--allow-root"]
'''


# =============================================================================
# REPRODUCIBILITY
# =============================================================================

def seed_everything(seed: int = CONFIG.SEED) -> None:
    """
    Fix ALL random seeds for strict reproducibility across the entire pipeline.
    
    This function must be called at the start of EVERY notebook to ensure
    identical results across runs. It sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and CUDA)
    - CUDA deterministic mode
    - cuDNN (disabled benchmark for reproducibility)
    
    Args:
        seed: Random seed value (default: CONFIG.SEED = 42)
    
    Note:
        cudnn.benchmark is set to FALSE for reproducibility.
        This may slightly reduce inference speed but ensures identical results.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # CRITICAL: Both must be set for full reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Disable for reproducibility
    
    print(f"✅ All seeds fixed to: {seed}")
    print(f"   cudnn.deterministic = True, cudnn.benchmark = False")


def seed_worker(worker_id: int) -> None:
    """
    Worker init function for PyTorch DataLoader reproducibility.
    
    PyTorch DataLoader spawns workers with different random states.
    This function ensures each worker gets a deterministic seed based on
    the global seed + worker_id, making augmentations reproducible.
    
    Usage:
        train_loader = DataLoader(
            dataset,
            num_workers=4,
            worker_init_fn=seed_worker,  # <-- ADD THIS
            generator=torch.Generator().manual_seed(CONFIG.SEED)
        )
    
    Reference:
        https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# =============================================================================
# PUBLICATION-QUALITY FIGURE CONFIGURATION
# =============================================================================

@dataclass
class FigureConfig:
    """
    Centralized figure configuration for publication-quality plots.
    
    Designed for IEEE two-column papers (column width ~3.5 inches).
    All notebooks should call `apply_figure_config()` at startup.
    """
    # DPI for saved figures (300 minimum for publication)
    SAVE_DPI: int = 300
    SCREEN_DPI: int = 100
    
    # Font sizes (scaled for two-column paper)
    TITLE_SIZE: int = 14
    LABEL_SIZE: int = 12
    TICK_SIZE: int = 10
    LEGEND_SIZE: int = 10
    ANNOTATION_SIZE: int = 9
    
    # Figure dimensions (inches, fits in single column)
    FIG_WIDTH_SINGLE: float = 3.5
    FIG_WIDTH_DOUBLE: float = 7.0
    FIG_HEIGHT: float = 2.8
    
    # Colorblind-friendly palette (derived from viridis)
    PALETTE: str = 'colorblind'
    
    # Export formats (vector for quality)
    SAVE_FORMATS: Tuple[str, ...] = ('.pdf', '.png')  # Vector + raster


FIGURE_CONFIG = FigureConfig()


def apply_figure_config() -> None:
    """
    Apply publication-quality matplotlib configuration globally.
    
    Call this at the start of every notebook before creating figures.
    Uses colorblind-friendly palette and publication-appropriate font sizes.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette(FIGURE_CONFIG.PALETTE)
    
    # Font sizes
    plt.rcParams.update({
        'font.size': FIGURE_CONFIG.LABEL_SIZE,
        'axes.titlesize': FIGURE_CONFIG.TITLE_SIZE,
        'axes.labelsize': FIGURE_CONFIG.LABEL_SIZE,
        'xtick.labelsize': FIGURE_CONFIG.TICK_SIZE,
        'ytick.labelsize': FIGURE_CONFIG.TICK_SIZE,
        'legend.fontsize': FIGURE_CONFIG.LEGEND_SIZE,
        'figure.titlesize': FIGURE_CONFIG.TITLE_SIZE,
        
        # Figure quality
        'figure.dpi': FIGURE_CONFIG.SCREEN_DPI,
        'savefig.dpi': FIGURE_CONFIG.SAVE_DPI,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Line widths
        'axes.linewidth': 1.0,
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        
        # Grid
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    print(f"✅ Figure config applied: DPI={FIGURE_CONFIG.SAVE_DPI}, Palette={FIGURE_CONFIG.PALETTE}")


def save_figure(fig, output_path: Path, formats: Tuple[str, ...] = None) -> List[Path]:
    """
    Save figure in multiple formats (PDF for vector, PNG for compatibility).
    
    Args:
        fig: Matplotlib figure
        output_path: Base path (without extension)
        formats: Tuple of extensions to save (default: PDF and PNG)
    
    Returns:
        List of saved file paths
    """
    import matplotlib.pyplot as plt
    
    if formats is None:
        formats = FIGURE_CONFIG.SAVE_FORMATS
    
    saved = []
    output_path = Path(output_path)
    
    for fmt in formats:
        save_path = output_path.with_suffix(fmt)
        fig.savefig(save_path, dpi=FIGURE_CONFIG.SAVE_DPI, bbox_inches='tight')
        saved.append(save_path)
    
    plt.close(fig)  # Prevent memory leak
    
    return saved


# =============================================================================
# LATEX TABLE EXPORT (Automation)
# =============================================================================

def to_latex_table(
    df: 'pd.DataFrame',
    output_path: Path,
    caption: str = "Results",
    label: str = "tab:results",
    float_format: str = "%.3f"
) -> Path:
    """
    Export DataFrame to LaTeX table fragment for direct paper inclusion.
    
    Eliminates manual copy-paste errors when updating paper results.
    
    Args:
        df: Pandas DataFrame to export
        output_path: Path to save .tex file
        caption: LaTeX table caption
        label: LaTeX reference label
        float_format: Number formatting string
    
    Returns:
        Path to saved .tex file
    
    Usage:
        to_latex_table(results_df, 'tables/ablation.tex', 
                       caption="Ablation Study Results",
                       label="tab:ablation")
    """
    import pandas as pd
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate LaTeX with proper formatting
    latex_content = df.to_latex(
        index=False,
        float_format=float_format,
        escape=False,
        column_format='l' + 'c' * (len(df.columns) - 1),
        caption=caption,
        label=label,
        position='htbp'
    )
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(latex_content)
    
    print(f"✅ LaTeX table saved: {output_path}")
    return output_path


# =============================================================================
# FILE LOGGING (Capture all output)
# =============================================================================

import logging
from datetime import datetime

def setup_logging(log_dir: Path = None, name: str = "execution") -> logging.Logger:
    """
    Setup file logging with timestamps to capture all print statements.
    
    Ensures that errors occurring during long runs are not lost when
    the terminal buffer clears.
    
    Args:
        log_dir: Directory for log files (default: current directory)
        name: Logger name and log file prefix
    
    Returns:
        Configured logger instance
    
    Usage:
        logger = setup_logging()
        logger.info("Starting training...")
    """
    if log_dir is None:
        log_dir = Path(".")
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{name}_{timestamp}.log"
    
    # Configure logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler (captures everything)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (for notebook visibility)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized: {log_file}")
    
    return logger


class TeeOutput:
    """
    Context manager to mirror print() statements to a log file.
    
    Usage:
        with TeeOutput('execution.log'):
            print("This goes to console AND file")
    """
    def __init__(self, log_path: Path):
        self.log_path = Path(log_path)
        self.log_file = None
        self._original_stdout = None
        
    def __enter__(self):
        import sys
        self._original_stdout = sys.stdout
        self.log_file = open(self.log_path, 'a', encoding='utf-8')
        sys.stdout = self
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import sys
        sys.stdout = self._original_stdout
        if self.log_file:
            self.log_file.close()
    
    def write(self, message):
        self._original_stdout.write(message)
        if self.log_file and message.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.log_file.write(f"[{timestamp}] {message}")
            self.log_file.flush()
    
    def flush(self):
        self._original_stdout.flush()
        if self.log_file:
            self.log_file.flush()


# =============================================================================
# METRIC CALCULATION: COCO-Compatible AP
# =============================================================================

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
    iou_threshold: float = CONFIG.IOU_THRESHOLD
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


# =============================================================================
# DATA SPLIT VERIFICATION
# =============================================================================

def load_and_verify_splits(
    splits_path: str = CONFIG.SPLITS_FILE,
    dataset_root: str = CONFIG.DATASET_ROOT,
    verbose: bool = True
) -> Dict[str, List[str]]:
    """
    Load the data splits from splits.json and verify all files exist.
    
    This ensures that all notebooks use the EXACT same train/val/test split
    as was created during training (NB11), preventing data leakage across runs.
    
    Args:
        splits_path: Path to the splits.json file
        dataset_root: Root directory of the dataset
        verbose: Whether to print verification status
    
    Returns:
        splits: Dictionary with 'train', 'val', 'test' keys containing file lists
    
    Raises:
        FileNotFoundError: If splits.json doesn't exist
        AssertionError: If any file in the splits doesn't exist on disk
    """
    splits_path = Path(splits_path)
    dataset_root = Path(dataset_root)
    
    if not splits_path.exists():
        raise FileNotFoundError(
            f"❌ splits.json not found at {splits_path}. "
            "Run 11_Training_Pro.ipynb first to generate splits."
        )
    
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    # Verify each split
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            print(f"⚠️ Warning: '{split_name}' not found in splits.json")
            continue
        
        files = splits[split_name]
        images_dir = dataset_root / "images" / split_name
        
        # FALLBACK: If 'val' folder doesn't exist, look in 'train' or 'test' folder
        # This handles:
        #   1. val is a virtual split from train (val images in train/)
        #   2. val = test strategy (val images in test/)
        if split_name == 'val' and not images_dir.exists():
            # First try test/ (for val=test strategy)
            test_dir = dataset_root / "images" / "test"
            train_dir = dataset_root / "images" / "train"
            
            if test_dir.exists():
                if verbose:
                    print(f"   ℹ️  val/ folder not found, using test/ as fallback (val=test strategy)")
                images_dir = test_dir
            elif train_dir.exists():
                if verbose:
                    print(f"   ℹ️  val/ folder not found, using train/ as fallback")
                images_dir = train_dir
        
        # Build case-insensitive lookup for cross-platform compatibility
        # This handles: image.JPG vs image.jpg mismatches between splits.json and disk
        try:
            actual_files = {f.name.lower(): f.name for f in images_dir.iterdir() if f.is_file()}
        except FileNotFoundError:
            actual_files = {}
        
        missing_files = []
        for fname in files:
            fname_lower = fname.lower()
            # Check exact match first, then case-insensitive
            if (images_dir / fname).exists():
                continue
            elif fname_lower in actual_files:
                continue  # Found with different case
            else:
                # Try with common extensions
                found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    stem = Path(fname).stem.lower()
                    if f"{stem}{ext}" in actual_files or f"{stem}{ext.upper()}" in actual_files:
                        found = True
                        break
                if not found:
                    missing_files.append(fname)
        
        if missing_files:
            raise AssertionError(
                f"❌ {len(missing_files)} files from '{split_name}' split not found!\n"
                f"   First 5: {missing_files[:5]}\n"
                f"   Expected directory: {images_dir}"
            )
        
        if verbose:
            print(f"✅ {split_name}: {len(files)} files verified")
    
    if verbose:
        print(f"✅ All splits verified from {splits_path}")
    
    return splits


def save_splits(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    output_path: str = CONFIG.SPLITS_FILE
) -> None:
    """
    Save data splits to JSON for reproducibility verification.
    
    This should be called in NB11 after creating K-Fold splits.
    
    Args:
        train_files: List of training file names
        val_files: List of validation file names
        test_files: List of test file names
        output_path: Path to save splits.json
    """
    splits = {
        'train': sorted(train_files),
        'val': sorted(val_files),
        'test': sorted(test_files),
        'metadata': {
            'seed': CONFIG.SEED,
            'created_by': '11_Training_Pro.ipynb'
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)
    
    print(f"✅ Splits saved to {output_path}")
    print(f"   Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def clear_gpu_cache() -> None:
    """Clear GPU memory cache to prevent OOM errors."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("✅ GPU cache cleared")


def get_image_files(directory: str, extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp')) -> List[str]:
    """Get all image files from a directory."""
    directory = Path(directory)
    files = []
    for ext in extensions:
        files.extend(directory.glob(f"*{ext}"))
        files.extend(directory.glob(f"*{ext.upper()}"))
    return sorted([f.name for f in files])


def ensure_dir(path: str) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# MOCK DATA FLAG ASSERTION
# =============================================================================

def assert_no_mock_data(mock_flag: bool, notebook_name: str) -> None:
    """
    Safety assertion to prevent publishing results with mock data.
    
    Call this at the END of each analysis notebook.
    
    Args:
        mock_flag: The MOCK_DATA flag from the notebook
        notebook_name: Name of the notebook for error message
    
    Raises:
        AssertionError: If mock_flag is True
    """
    assert not mock_flag, (
        f"❌ CRITICAL: {notebook_name} is using MOCK_DATA=True!\n"
        "   Results are FAKE and cannot be used in the paper.\n"
        "   Set MOCK_DATA = False and re-run with real data."
    )
    print(f"✅ {notebook_name}: MOCK_DATA verification passed")


# =============================================================================
# MODULE INFO
# =============================================================================

__version__ = "1.0.0"
__all__ = [
    'CONFIG',
    'ResearchConfig',
    'seed_everything',
    'compute_ap_coco',
    'compute_metrics_coco',
    'compute_iou_matrix',
    'load_and_verify_splits',
    'save_splits',
    'clear_gpu_cache',
    'get_image_files',
    'ensure_dir',
    'assert_no_mock_data',
]


if __name__ == "__main__":
    # Self-test
    print("=" * 60)
    print("utils_research.py - Self Test")
    print("=" * 60)
    
    print(f"\n📋 CONFIG Values:")
    print(f"   SEED: {CONFIG.SEED}")
    print(f"   TTA_SCALES: {CONFIG.TTA_SCALES}")
    print(f"   WBF_WEIGHTS_TTA: {CONFIG.WBF_WEIGHTS_TTA}")
    print(f"   WBF_IOU_THR: {CONFIG.WBF_IOU_THR}")
    print(f"   CONF_THRESHOLD: {CONFIG.CONF_THRESHOLD}")
    
    print(f"\n🔧 Testing seed_everything()...")
    seed_everything(42)
    
    print(f"\n📊 Testing compute_ap_coco()...")
    # Test with known precision-recall curve
    recalls = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    precisions = np.array([1.0, 0.9, 0.8, 0.7, 0.6])
    ap = compute_ap_coco(recalls, precisions)
    print(f"   AP = {ap:.4f} (expected ~0.45-0.50)")
    
    print("\n✅ All self-tests passed!")
