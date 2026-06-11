from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score
from src.config import OUTPUT_DIR
from src.utils import get_logger, save_figure, set_plot_style, PALETTE
logger = get_logger(__name__)

def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_precision: float=0.15, strategy: str='max_recall') -> Dict[str, float]:
    (precisions, recalls, thresholds) = precision_recall_curve(y_true, y_prob)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    if strategy == 'max_recall':
        mask = precisions >= min_precision
        if not mask.any():
            logger.warning('No threshold achieves precision ≥ %.2f — falling back to max-F1 strategy.', min_precision)
            return find_optimal_threshold(y_true, y_prob, strategy='max_f1')
        valid_idx = np.where(mask)[0]
        best_idx = valid_idx[np.argmax(recalls[valid_idx])]
    elif strategy == 'max_f1':
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx = np.argmax(f1_scores)
    else:
        raise ValueError(f'Unknown strategy: {strategy}')
    threshold = float(thresholds[best_idx])
    y_pred = (y_prob >= threshold).astype(int)
    result = {'threshold': threshold, 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0)}
    logger.info('═' * 50)
    logger.info('Optimal threshold (strategy=%s): %.3f', strategy, result['threshold'])
    logger.info('  Precision : %.4f', result['precision'])
    logger.info('  Recall    : %.4f', result['recall'])
    logger.info('  F1        : %.4f', result['f1'])
    return result

def plot_threshold_analysis(y_true: np.ndarray, y_prob: np.ndarray, chosen_threshold: float, output_dir: Path=OUTPUT_DIR) -> Path:
    set_plot_style()
    thresholds = np.linspace(0.01, 0.99, 200)
    (precs, recs, f1s) = ([], [], [])
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    (fig, ax) = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, precs, color=PALETTE[0], lw=2, label='Precision')
    ax.plot(thresholds, recs, color=PALETTE[1], lw=2, label='Recall')
    ax.plot(thresholds, f1s, color=PALETTE[2], lw=2, label='F1 Score')
    ax.axvline(x=chosen_threshold, color=PALETTE[3], ls='--', lw=2, label=f'Chosen threshold = {chosen_threshold:.3f}')
    ax.axvline(x=0.5, color='gray', ls=':', lw=1, label='Default 0.5')
    ax.set_xlabel('Decision Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Threshold Optimisation: Precision / Recall / F1 Trade-off')
    ax.legend(loc='center left', fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    return save_figure(fig, 'threshold_analysis.png', output_dir)

def document_threshold_rationale(result: Dict[str, float], output_dir: Path=OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'threshold_rationale.md'
    lines = ['# Threshold Optimisation Rationale', '', '## Clinical Context', '', 'In a hospital readmission reduction programme, there are two types of', 'errors:', '', '1. **False Negative (missed readmission)** — A patient who *will* be', '   readmitted is not flagged.  This is costly: the hospital misses an', '   opportunity for transitional care (discharge planning, medication', '   reconciliation, home-health referral) and may incur CMS penalties.', '', '2. **False Positive (unnecessary flag)** — A patient who would *not*', '   be readmitted receives additional follow-up.  This is far less', '   harmful: the extra phone call or clinic visit has minimal downside', '   and may even improve patient satisfaction.', '', 'Because the asymmetry clearly favours **catching more true positives**', '(high recall) over avoiding false alarms, we optimise the threshold to', 'maximise recall while keeping precision above a clinically acceptable', 'floor.', '', '## Chosen Operating Point', '', f'| Metric    | Value  |', f'|-----------|--------|', f"| Threshold | {result['threshold']:.3f}  |", f"| Precision | {result['precision']:.4f} |", f"| Recall    | {result['recall']:.4f} |", f"| F1 Score  | {result['f1']:.4f} |", '', '## Business Impact', '', f"At a threshold of **{result['threshold']:.3f}**, the model catches", f"**{result['recall'] * 100:.1f}%** of patients who will be readmitted", f"within 30 days.  The trade-off is that **{(1 - result['precision']) * 100:.1f}%**", 'of flagged patients are false positives — they would not actually have', 'been readmitted.  For a care-management programme, this is an', 'acceptable rate: the cost of an unnecessary follow-up call is trivial', 'compared to the cost of a preventable readmission (~$15,000 average).', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Threshold rationale saved → %s', path)
    return path