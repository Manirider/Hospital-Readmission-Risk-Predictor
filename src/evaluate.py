from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, auc, average_precision_score, classification_report, confusion_matrix, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from src.config import OUTPUT_DIR, RANDOM_SEED
from src.utils import get_logger, save_figure, set_plot_style, PALETTE
logger = get_logger(__name__)

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'f1': f1_score(y_true, y_pred, zero_division=0), 'roc_auc': roc_auc_score(y_true, y_prob), 'pr_auc': average_precision_score(y_true, y_prob)}
    return metrics

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, title: str='Classification Report') -> str:
    report = classification_report(y_true, y_pred, target_names=['Not Readmitted', 'Readmitted <30d'])
    logger.info('\n%s\n%s', title, report)
    return report

def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path=OUTPUT_DIR) -> Path:
    set_plot_style()
    (fpr, tpr, _) = roc_curve(y_true, y_prob)
    roc_auc_val = auc(fpr, tpr)
    (fig, ax) = plt.subplots(figsize=(8, 7))
    ax.plot(fpr, tpr, color=PALETTE[0], lw=2.5, label=f'ROC curve (AUC = {roc_auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random baseline')
    ax.fill_between(fpr, tpr, alpha=0.12, color=PALETTE[0])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    return save_figure(fig, 'roc_curve.png', output_dir)

def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, output_dir: Path=OUTPUT_DIR) -> Path:
    set_plot_style()
    (precision, recall, _) = precision_recall_curve(y_true, y_prob)
    pr_auc_val = average_precision_score(y_true, y_prob)
    (fig, ax) = plt.subplots(figsize=(8, 7))
    ax.plot(recall, precision, color=PALETTE[1], lw=2.5, label=f'PR curve (AP = {pr_auc_val:.3f})')
    ax.fill_between(recall, precision, alpha=0.12, color=PALETTE[1])
    baseline = y_true.mean()
    ax.axhline(y=baseline, color='gray', ls='--', lw=1, label=f'Baseline prevalence = {baseline:.3f}')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision–Recall Curve')
    ax.legend(loc='upper right', fontsize=11)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([0, 1.05])
    return save_figure(fig, 'pr_curve.png', output_dir)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path=OUTPUT_DIR) -> Path:
    set_plot_style()
    cm = confusion_matrix(y_true, y_pred)
    (fig, ax) = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', xticklabels=['Not Readmitted', 'Readmitted <30d'], yticklabels=['Not Readmitted', 'Readmitted <30d'], ax=ax, linewidths=0.5)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    return save_figure(fig, 'confusion_matrix.png', output_dir)

def plot_feature_importance(pipeline: Pipeline, top_n: int=20, output_dir: Path=OUTPUT_DIR) -> Path:
    set_plot_style()
    estimator = pipeline
    if hasattr(estimator, 'calibrated_classifiers_'):
        estimator = estimator.calibrated_classifiers_[0].estimator
    classifier = estimator.named_steps['classifier']
    preprocessor = estimator.named_steps['preprocessor']
    if not hasattr(classifier, 'feature_importances_'):
        logger.warning('Classifier lacks feature_importances_ — skipping plot.')
        return output_dir / 'feature_importance.png'
    importances = classifier.feature_importances_
    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f'feature_{i}' for i in range(len(importances))]
    clean_names = [n.replace('num__', '').replace('cat__', '') for n in feature_names]
    indices = np.argsort(importances)[::-1][:top_n]
    (fig, ax) = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices][::-1], color=PALETTE[0], edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([clean_names[i] for i in indices][::-1])
    ax.set_xlabel('Feature Importance')
    ax.set_title(f'Top {top_n} Features by Importance')
    ax.invert_yaxis()
    return save_figure(fig, 'feature_importance.png', output_dir)

def evaluate_model(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, threshold: float=0.5, output_dir: Path=OUTPUT_DIR) -> Dict[str, Any]:
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_test.values, y_pred, y_prob)
    report = print_classification_report(y_test.values, y_pred)
    logger.info('Evaluation metrics at threshold=%.2f:', threshold)
    for (k, v) in metrics.items():
        logger.info('  %-12s  %.4f', k, v)
    roc_path = plot_roc_curve(y_test.values, y_prob, output_dir)
    pr_path = plot_pr_curve(y_test.values, y_prob, output_dir)
    cm_path = plot_confusion_matrix(y_test.values, y_pred, output_dir)
    fi_path = plot_feature_importance(pipeline, output_dir=output_dir)
    return {'metrics': metrics, 'classification_report': report, 'y_prob': y_prob, 'y_pred': y_pred, 'plots': {'roc_curve': roc_path, 'pr_curve': pr_path, 'confusion_matrix': cm_path, 'feature_importance': fi_path}}