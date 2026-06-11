from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from src.config import ICD9_RANGES, MEDICATION_FEATURES, DROP_COLS, BINARY_TARGET
from src.utils import get_logger
logger = get_logger(__name__)

def _icd9_to_numeric(code: str) -> Optional[float]:
    if pd.isna(code):
        return None
    code = str(code).strip()
    if code == '':
        return None
    if code[0] in ('E', 'V', 'e', 'v'):
        return None
    try:
        return float(code)
    except ValueError:
        return None

def map_icd9_to_category(code: str) -> str:
    num = _icd9_to_numeric(code)
    if num is None:
        return 'Other'
    prefix = int(num)
    for (category, ranges) in ICD9_RANGES.items():
        for (lo, hi) in ranges:
            if lo <= prefix <= hi:
                return category
    return 'Other'

def add_diagnosis_categories(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ('diag_1', 'diag_2', 'diag_3'):
        new_col = f'{col}_cat'
        df[new_col] = df[col].apply(map_icd9_to_category)
        logger.debug('Mapped %s → %s  (%s unique categories)', col, new_col, df[new_col].nunique())
    return df

def add_medication_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    med_cols = [c for c in MEDICATION_FEATURES if c in df.columns]
    n_changed = pd.DataFrame()
    for col in med_cols:
        n_changed[col] = df[col].isin(['Up', 'Down']).astype(int)
    df['n_med_changed'] = n_changed.sum(axis=1)
    df['any_med_changed'] = (df['n_med_changed'] > 0).astype(int)
    logger.info('Medication flags — mean changes per encounter: %.2f', df['n_med_changed'].mean())
    return df

def add_utilisation_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    visit_cols = ['number_inpatient', 'number_outpatient', 'number_emergency']
    available = [c for c in visit_cols if c in df.columns]
    df['prior_visits_total'] = df[available].sum(axis=1)
    return df
AGE_BRACKET_MAP: Dict[str, int] = {'[0-10)': 0, '[10-20)': 1, '[20-30)': 2, '[30-40)': 3, '[40-50)': 4, '[50-60)': 5, '[60-70)': 6, '[70-80)': 7, '[80-90)': 8, '[90-100)': 9}

def encode_age_bracket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['age_ordinal'] = df['age'].map(AGE_BRACKET_MAP).fillna(-1).astype(int)
    return df

def fill_race_unknown(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n_missing = df['race'].isna().sum()
    df['race'] = df['race'].fillna('Unknown')
    logger.info("Filled %s missing race values → 'Unknown'", f'{n_missing:,}')
    return df

def drop_unwanted_columns(df: pd.DataFrame, extra: Optional[List[str]]=None) -> pd.DataFrame:
    df = df.copy()
    to_drop = list(DROP_COLS) + (extra or [])
    for col in ('diag_1', 'diag_2', 'diag_3'):
        if col in df.columns and f'{col}_cat' in df.columns:
            to_drop.append(col)
    med_cols = [c for c in MEDICATION_FEATURES if c in df.columns]
    to_drop.extend(med_cols)
    if 'readmitted' in df.columns and BINARY_TARGET in df.columns:
        to_drop.append('readmitted')
    present = [c for c in to_drop if c in df.columns]
    df.drop(columns=present, inplace=True)
    logger.info('Dropped %s columns: %s', len(present), present)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    logger.info('Starting feature engineering …')
    df = fill_race_unknown(df)
    df = add_diagnosis_categories(df)
    df = add_medication_flags(df)
    df = add_utilisation_score(df)
    df = encode_age_bracket(df)
    df = drop_unwanted_columns(df)
    logger.info('Feature engineering complete — %s rows × %s columns', f'{len(df):,}', df.shape[1])
    return df