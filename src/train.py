from __future__ import annotations
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from src.config import BINARY_TARGET, CV_FOLDS, HP_GRID, N_ITER_SEARCH, RANDOM_SEED, TEST_SIZE
from src.preprocessing import build_pipeline, resolve_feature_lists, split_X_y
from src.utils import get_logger, save_pipeline, set_seed
logger = get_logger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

def _get_optional_classifiers() -> Dict[str, Any]:
    classifiers: Dict[str, Any] = {}
    try:
        from xgboost import XGBClassifier
        classifiers['XGBoost'] = XGBClassifier
    except ImportError:
        logger.warning('XGBoost not installed — skipping.')
    try:
        from lightgbm import LGBMClassifier
        classifiers['LightGBM'] = LGBMClassifier
    except ImportError:
        logger.warning('LightGBM not installed — skipping.')
    return classifiers

def train_baseline(X_train: pd.DataFrame, y_train: pd.Series, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    set_seed()
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED, solver='lbfgs')
    pipe = build_pipeline(lr, numeric_cols, categorical_cols)
    logger.info('Training baseline Logistic Regression …')
    pipe.fit(X_train, y_train)
    logger.info('Baseline training complete.')
    return pipe

def _build_candidate_pipelines(numeric_cols: List[str], categorical_cols: List[str]) -> Dict[str, Tuple[Pipeline, Dict[str, Any]]]:
    candidates: Dict[str, Tuple[Pipeline, Dict[str, Any]]] = {}
    gb = GradientBoostingClassifier(random_state=RANDOM_SEED)
    candidates['GradientBoosting'] = (build_pipeline(gb, numeric_cols, categorical_cols), HP_GRID.gradient_boosting)
    rf = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)
    candidates['RandomForest'] = (build_pipeline(rf, numeric_cols, categorical_cols), HP_GRID.random_forest)
    opt = _get_optional_classifiers()
    if 'XGBoost' in opt:
        xgb = opt['XGBoost'](random_state=RANDOM_SEED, eval_metric='logloss', use_label_encoder=False, verbosity=0)
        candidates['XGBoost'] = (build_pipeline(xgb, numeric_cols, categorical_cols), HP_GRID.xgboost)
    if 'LightGBM' in opt:
        lgbm = opt['LightGBM'](random_state=RANDOM_SEED, verbose=-1)
        candidates['LightGBM'] = (build_pipeline(lgbm, numeric_cols, categorical_cols), HP_GRID.lightgbm)
    return candidates

def train_all_models(X_train: pd.DataFrame, y_train: pd.Series, numeric_cols: List[str], categorical_cols: List[str], n_iter: int=N_ITER_SEARCH, cv_folds: int=CV_FOLDS) -> Dict[str, Dict[str, Any]]:
    set_seed()
    candidates = _build_candidate_pipelines(numeric_cols, categorical_cols)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
    results: Dict[str, Dict[str, Any]] = {}
    for (name, (pipe, param_grid)) in candidates.items():
        logger.info('═' * 60)
        logger.info('Tuning %s  (%s iterations, %s-fold CV) …', name, n_iter, cv_folds)
        t0 = time.time()
        tune_size = min(15000, len(X_train))
        X_tune = X_train.sample(n=tune_size, random_state=RANDOM_SEED)
        y_tune = y_train.loc[X_tune.index]
        search = RandomizedSearchCV(estimator=pipe, param_distributions=param_grid, n_iter=n_iter, scoring='roc_auc', cv=cv, random_state=RANDOM_SEED, n_jobs=1, verbose=0, error_score='raise')
        search.fit(X_tune, y_tune)
        logger.info('Refitting best %s model on full training set (%s rows) …', name, f'{len(X_train):,}')
        best_pipe = search.best_estimator_
        best_pipe.fit(X_train, y_train)
        elapsed = time.time() - t0
        results[name] = {'pipeline': best_pipe, 'cv_score': search.best_score_, 'best_params': search.best_params_, 'train_time_s': elapsed}
        logger.info('%s — best CV ROC-AUC: %.4f  (%.1fs)', name, search.best_score_, elapsed)
    return results

def select_best_model(results: Dict[str, Dict[str, Any]]) -> Tuple[str, Pipeline]:
    ranked = sorted(results.items(), key=lambda kv: kv[1]['cv_score'], reverse=True)
    (best_name, best_info) = ranked[0]
    logger.info('═' * 60)
    logger.info('Model comparison summary:')
    for (name, info) in ranked:
        marker = ' ◀ SELECTED' if name == best_name else ''
        logger.info('  %-20s  ROC-AUC=%.4f  time=%.1fs%s', name, info['cv_score'], info['train_time_s'], marker)
    return (best_name, best_info['pipeline'])

def run_training(df: pd.DataFrame, test_size: float=TEST_SIZE) -> Dict[str, Any]:
    set_seed()
    (X, y) = split_X_y(df)
    (numeric_cols, categorical_cols) = resolve_feature_lists(X)
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=test_size, random_state=RANDOM_SEED, stratify=y)
    logger.info('Train/test split — train: %s  test: %s  positive rate: %.2f%%', f'{len(X_train):,}', f'{len(X_test):,}', 100 * y_train.mean())
    baseline = train_baseline(X_train, y_train, numeric_cols, categorical_cols)
    all_results = train_all_models(X_train, y_train, numeric_cols, categorical_cols)
    (best_name, best_pipeline) = select_best_model(all_results)
    save_pipeline(best_pipeline)
    return {'best_name': best_name, 'best_pipeline': best_pipeline, 'baseline_pipeline': baseline, 'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test, 'all_results': all_results, 'numeric_cols': numeric_cols, 'categorical_cols': categorical_cols}