from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from src.config import CV_FOLDS, OUTPUT_DIR, RANDOM_SEED
from src.utils import get_logger, save_figure, set_plot_style, PALETTE
logger = get_logger(__name__)

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int=10) -> float:
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for (lo, hi) in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob > lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_weight = mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_acc - bin_conf)
    return ece

def calibrate_model(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray, method: str='sigmoid', cv_folds: int=CV_FOLDS) -> Pipeline:
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    calibrated = CalibratedClassifierCV(estimator=pipeline, method=method, cv=cv)
    logger.info("Calibrating with method='%s', cv=%s …", method, cv_folds)
    calibrated.fit(X_train, y_train)
    logger.info('Calibration complete.')
    return calibrated

def compare_calibration_methods(pipeline: Pipeline, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[str, Any]:
    results: Dict[str, Dict[str, Any]] = {}
    for method in ('sigmoid', 'isotonic'):
        cal = calibrate_model(pipeline, X_train, y_train, method=method)
        y_prob = cal.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, y_prob)
        ece = expected_calibration_error(y_test.values if hasattr(y_test, 'values') else y_test, y_prob)
        results[method] = {'model': cal, 'brier': brier, 'ece': ece}
        logger.info('  %s — Brier=%.4f  ECE=%.4f', method, brier, ece)
    best_method = min(results, key=lambda m: results[m]['brier'])
    logger.info('Selected calibration method: %s', best_method)
    return (best_method, results[best_method]['model'])

def plot_calibration_curve(y_true: np.ndarray, y_prob_uncal: np.ndarray, y_prob_cal: np.ndarray, output_dir: Path=OUTPUT_DIR, n_bins: int=10) -> Path:
    set_plot_style()
    (frac_pos_uncal, mean_pred_uncal) = calibration_curve(y_true, y_prob_uncal, n_bins=n_bins, strategy='uniform')
    (frac_pos_cal, mean_pred_cal) = calibration_curve(y_true, y_prob_cal, n_bins=n_bins, strategy='uniform')
    brier_uncal = brier_score_loss(y_true, y_prob_uncal)
    brier_cal = brier_score_loss(y_true, y_prob_cal)
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6))
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
    ax.plot(mean_pred_uncal, frac_pos_uncal, 's-', color=PALETTE[3], label=f'Uncalibrated (Brier={brier_uncal:.4f})')
    ax.plot(mean_pred_cal, frac_pos_cal, 'o-', color=PALETTE[0], label=f'Calibrated (Brier={brier_cal:.4f})')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Observed Frequency')
    ax.set_title('Calibration Curve (Reliability Diagram)')
    ax.legend(loc='lower right')
    ax2 = axes[1]
    ax2.hist(y_prob_uncal, bins=30, alpha=0.5, color=PALETTE[3], label='Uncalibrated', density=True)
    ax2.hist(y_prob_cal, bins=30, alpha=0.5, color=PALETTE[0], label='Calibrated', density=True)
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Predicted Probability Distribution')
    ax2.legend()
    fig.tight_layout()
    return save_figure(fig, 'calibration_curve.png', output_dir)