from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, BINARY_TARGET, RANDOM_SEED
from src.utils import get_logger
logger = get_logger(__name__)

def resolve_feature_lists(df: pd.DataFrame, numeric: Optional[List[str]]=None, categorical: Optional[List[str]]=None) -> Tuple[List[str], List[str]]:
    num_candidates = list(numeric or NUMERIC_FEATURES)
    cat_candidates = list(categorical or CATEGORICAL_FEATURES)
    engineered_numeric = ['n_med_changed', 'any_med_changed', 'prior_visits_total', 'age_ordinal']
    for col in engineered_numeric:
        if col in df.columns and col not in num_candidates:
            num_candidates.append(col)
    num_cols = [c for c in num_candidates if c in df.columns]
    cat_cols = [c for c in cat_candidates if c in df.columns]
    for lst in (num_cols, cat_cols):
        for target in (BINARY_TARGET, 'readmitted'):
            if target in lst:
                lst.remove(target)
    logger.info('Numeric features  (%s): %s', len(num_cols), num_cols)
    logger.info('Categorical features (%s): %s', len(cat_cols), cat_cols)
    return (num_cols, cat_cols)

def build_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_pipeline = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False, min_frequency=0.01))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_pipeline, numeric_cols), ('cat', categorical_pipeline, categorical_cols)], remainder='drop', verbose_feature_names_out=True)
    return preprocessor

def build_pipeline(classifier, numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    logger.info('Built pipeline: %s → %s', type(preprocessor).__name__, type(classifier).__name__)
    return pipe

def split_X_y(df: pd.DataFrame, target: str=BINARY_TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    y = df[target].copy()
    X = df.drop(columns=[target], errors='ignore')
    return (X, y)