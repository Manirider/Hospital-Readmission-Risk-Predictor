from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from src.config import OUTPUT_DIR
from src.utils import get_logger, save_figure, set_plot_style, PALETTE
logger = get_logger(__name__)

def compute_subgroup_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    unique_groups = np.unique(groups)
    rows: List[Dict[str, Any]] = []
    for g in unique_groups:
        mask = groups == g
        yt = y_true[mask]
        yp = y_pred[mask]
        if len(yt) < 10:
            logger.warning("Subgroup '%s' has <10 samples — metrics unreliable.", g)
        cm = confusion_matrix(yt, yp, labels=[0, 1])
        (tn, fp, fn, tp) = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        rows.append({'group': str(g), 'precision': precision_score(yt, yp, zero_division=0), 'recall': recall_score(yt, yp, zero_division=0), 'f1': f1_score(yt, yp, zero_division=0), 'fpr': fp / max(fp + tn, 1), 'fnr': fn / max(fn + tp, 1), 'support': int(mask.sum()), 'positive_rate': float(yt.mean())})
    return pd.DataFrame(rows).sort_values('support', ascending=False).reset_index(drop=True)

def run_fairness_audit(X_test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path=OUTPUT_DIR) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    audit_frames: List[pd.DataFrame] = []
    for col in ('race', 'gender', 'age'):
        if col not in X_test.columns:
            logger.warning("Column '%s' not in X_test — skipping.", col)
            continue
        groups = X_test[col].fillna('Unknown').values
        metrics = compute_subgroup_metrics(y_true_arr, y_pred_arr, groups)
        metrics.insert(0, 'attribute', col)
        audit_frames.append(metrics)
        logger.info('Fairness — %s subgroups:', col)
        for (_, row) in metrics.iterrows():
            logger.info('  %-20s  P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f  FNR=%.3f  n=%d', row['group'], row['precision'], row['recall'], row['f1'], row['fpr'], row['fnr'], row['support'])
    if not audit_frames:
        logger.error('No demographic columns found for fairness audit.')
        return pd.DataFrame()
    combined = pd.concat(audit_frames, ignore_index=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'bias_audit.csv'
    combined.to_csv(csv_path, index=False)
    logger.info('Bias audit saved → %s', csv_path)
    return combined

def detect_significant_gaps(audit_df: pd.DataFrame, gap_threshold: float=0.1) -> pd.DataFrame:
    flags: List[Dict[str, Any]] = []
    for attr in audit_df['attribute'].unique():
        sub = audit_df[audit_df['attribute'] == attr]
        for metric in ('recall', 'precision'):
            max_val = sub[metric].max()
            min_val = sub[metric].min()
            gap = max_val - min_val
            if gap > gap_threshold:
                max_group = sub.loc[sub[metric].idxmax(), 'group']
                min_group = sub.loc[sub[metric].idxmin(), 'group']
                flags.append({'attribute': attr, 'metric': metric, 'group_high': max_group, 'group_low': min_group, 'gap': gap})
                logger.warning('⚠ %s gap in %s: %s (%.3f) vs %s (%.3f) — Δ=%.3f', metric.capitalize(), attr, max_group, max_val, min_group, min_val, gap)
    return pd.DataFrame(flags) if flags else pd.DataFrame(columns=['attribute', 'metric', 'group_high', 'group_low', 'gap'])

def generate_fairness_report(audit_df: pd.DataFrame, gaps_df: pd.DataFrame, output_dir: Path=OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'fairness_report.md'
    lines = ['# Fairness & Bias Audit Report', '', '## Overview', '', 'This report evaluates whether the readmission risk model performs', 'equitably across demographic subgroups defined by **race**, **gender**,', 'and **age**.  Disparities can arise from data imbalance, systemic', 'healthcare inequities reflected in the training data, or feature', 'engineering choices.', '', '## Subgroup Performance', '', audit_df.to_markdown(index=False), '']
    if len(gaps_df) > 0:
        lines += ['## ⚠ Significant Gaps Detected', '', gaps_df.to_markdown(index=False), '', '## Discussion', '', '### Potential Causes', '', '1. **Data Imbalance** — Minority subgroups have fewer samples, leading', '   to noisier metric estimates and potentially worse model fit.', '', '2. **Healthcare Inequities** — The training data reflects real-world', '   disparities in access, treatment, and documentation quality.', '   Patients from underserved populations may have sparser clinical', '   records, making their readmission risk harder to predict.', '', '3. **Sampling Effects** — The dataset covers 130 US hospitals over', '   1999–2008; geographic and temporal sampling may not represent the', '   current patient population.', '', '4. **Feature Bias** — Some features (e.g. number of prior visits) may', '   proxy for insurance status or socioeconomic factors that correlate', '   with race.', '', '### Limitations', '', '- Fairness metrics are computed on a single hold-out split; bootstrap', '  confidence intervals would provide more robust estimates.', '- We assess *group* fairness only; individual fairness and causal', '  fairness require additional analysis.', '- The `Unknown` race category may conflate multiple underrepresented', '  groups.', '', '### Recommendations', '', '1. Compute bootstrap confidence intervals around subgroup metrics.', '2. Investigate whether re-calibrating per subgroup improves equity.', '3. Consider threshold-adjustment strategies for underserved groups.', '4. Engage clinical and health-equity stakeholders before deployment.', '5. Monitor subgroup performance in production with automated alerts.', '']
    else:
        lines += ['## ✓ No significant gaps detected', '', 'All recall and precision gaps are within ±10% across subgroups.', 'This does not guarantee fairness — ongoing monitoring is essential.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Fairness report saved → %s', path)
    return path

def plot_fairness_comparison(audit_df: pd.DataFrame, attribute: str='race', output_dir: Path=OUTPUT_DIR) -> Path:
    set_plot_style()
    sub = audit_df[audit_df['attribute'] == attribute].copy()
    (fig, axes) = plt.subplots(1, 2, figsize=(14, 6))
    for (ax, metric, color) in zip(axes, ['recall', 'precision'], [PALETTE[0], PALETTE[1]]):
        sub_sorted = sub.sort_values(metric, ascending=True)
        ax.barh(sub_sorted['group'], sub_sorted[metric], color=color, edgecolor='white')
        ax.set_xlabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by {attribute.capitalize()}')
        ax.set_xlim([0, 1])
        for (i, v) in enumerate(sub_sorted[metric]):
            ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    fig.tight_layout()
    return save_figure(fig, f'fairness_{attribute}.png', output_dir)