from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from src.config import OUTPUT_DIR
from src.utils import get_logger, save_figure, set_plot_style
logger = get_logger(__name__)

def compute_shap_values(pipeline, X_test: pd.DataFrame, max_samples: int=1000) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    estimator = pipeline
    if hasattr(estimator, 'estimator'):
        estimator = estimator.calibrated_classifiers_[0].estimator
    preprocessor = estimator.named_steps['preprocessor']
    classifier = estimator.named_steps['classifier']
    if len(X_test) > max_samples:
        X_sample = X_test.sample(n=max_samples, random_state=42)
    else:
        X_sample = X_test
    X_transformed = preprocessor.transform(X_sample)
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f'feature_{i}' for i in range(X_transformed.shape[1])]
    feature_names = [n.replace('num__', '').replace('cat__', '') for n in feature_names]
    if hasattr(classifier, 'feature_importances_'):
        logger.info('Using TreeExplainer for SHAP values …')
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]
    else:
        logger.info('Using KernelExplainer (slower) for SHAP values …')
        background = shap.sample(X_transformed, min(100, len(X_transformed)))
        explainer = shap.KernelExplainer(classifier.predict_proba, background)
        shap_values = explainer.shap_values(X_transformed)[1]
    logger.info('SHAP values computed — shape: %s  features: %s', shap_values.shape, len(feature_names))
    return (shap_values, X_transformed, feature_names)

def plot_shap_summary(shap_values: np.ndarray, X_transformed: np.ndarray, feature_names: List[str], top_n: int=20, output_dir: Path=OUTPUT_DIR) -> Path:
    set_plot_style()
    (fig, ax) = plt.subplots(figsize=(12, 10))
    plt.sca(ax)
    shap.summary_plot(shap_values, X_transformed, feature_names=feature_names, max_display=top_n, show=False, plot_size=None)
    ax.set_title('SHAP Feature Importance — Global Summary', fontsize=14, pad=15)
    return save_figure(fig, 'shap_summary.png', output_dir)

def analyse_top_features(shap_values: np.ndarray, feature_names: List[str], top_n: int=20) -> pd.DataFrame:
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({'feature': feature_names, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False).head(top_n)
    df['rank'] = range(1, len(df) + 1)
    df = df.reset_index(drop=True)
    return df

def find_example_indices(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[int]]:
    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)

    def _first(mask):
        idxs = np.where(mask)[0]
        return int(idxs[0]) if len(idxs) > 0 else None
    return {'tp': _first(tp_mask), 'fp': _first(fp_mask), 'fn': _first(fn_mask)}

def plot_force_explanation(shap_values: np.ndarray, X_transformed: np.ndarray, feature_names: List[str], sample_idx: int, label: str, expected_value: float, output_dir: Path=OUTPUT_DIR) -> Path:
    set_plot_style()
    explanation = shap.Explanation(values=shap_values[sample_idx], base_values=expected_value, data=X_transformed[sample_idx], feature_names=feature_names)
    (fig, ax) = plt.subplots(figsize=(12, 8))
    plt.sca(ax)
    shap.plots.waterfall(explanation, max_display=15, show=False)
    title_map = {'tp': 'True Positive — Correctly Flagged for Readmission', 'fp': 'False Positive — Incorrectly Flagged', 'fn': 'False Negative — Missed Readmission'}
    ax.set_title(title_map.get(label, label), fontsize=13, pad=15)
    filename = f'shap_force_{label}.png'
    return save_figure(fig, filename, output_dir)

def generate_local_explanations(pipeline, X_test: pd.DataFrame, y_test: np.ndarray, y_pred: np.ndarray, shap_values: np.ndarray, X_transformed: np.ndarray, feature_names: List[str], output_dir: Path=OUTPUT_DIR) -> Dict[str, Path]:
    if len(y_test) > len(X_transformed):
        sample_idx_map = X_test.head(len(X_transformed)).index
        y_test_sub = y_test.loc[sample_idx_map].values if hasattr(y_test, 'loc') else y_test[:len(X_transformed)]
        y_pred_sub = y_pred[:len(X_transformed)]
    else:
        y_test_sub = y_test if isinstance(y_test, np.ndarray) else y_test.values
        y_pred_sub = y_pred
    examples = find_example_indices(y_test_sub, y_pred_sub)
    estimator = pipeline
    if hasattr(estimator, 'calibrated_classifiers_'):
        estimator = estimator.calibrated_classifiers_[0].estimator
    classifier = estimator.named_steps['classifier']
    if hasattr(classifier, 'feature_importances_'):
        explainer = shap.TreeExplainer(classifier)
        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    else:
        expected_value = shap_values.mean()
    paths: Dict[str, Path] = {}
    for (label, idx) in examples.items():
        if idx is None:
            logger.warning('No %s example found — skipping force plot.', label.upper())
            continue
        path = plot_force_explanation(shap_values, X_transformed, feature_names, idx, label, float(expected_value), output_dir)
        paths[label] = path
    return paths