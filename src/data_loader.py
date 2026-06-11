from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
from src.config import DATASET_URL, DATASET_FILENAME, RAW_DATA_DIR, TARGET_COL, POSITIVE_LABEL, BINARY_TARGET
from src.utils import get_logger
logger = get_logger(__name__)

def load_raw_data(data_dir: Optional[Path]=None, url: str=DATASET_URL) -> pd.DataFrame:
    data_dir = data_dir or RAW_DATA_DIR
    csv_path = data_dir / DATASET_FILENAME
    if csv_path.exists():
        logger.info('Loading cached dataset from %s', csv_path)
    else:
        logger.info('Dataset not found locally — downloading from UCI …')
        _download_and_extract(url, data_dir)
    df = pd.read_csv(csv_path, na_values=['?'], low_memory=False)
    logger.info('Loaded %s rows × %s columns  (%.1f MB in memory)', f'{len(df):,}', df.shape[1], df.memory_usage(deep=True).sum() / 1000000.0)
    return df

def create_binary_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[BINARY_TARGET] = (df[TARGET_COL] == POSITIVE_LABEL).astype(int)
    n_pos = df[BINARY_TARGET].sum()
    n_neg = len(df) - n_pos
    ratio = n_neg / max(n_pos, 1)
    logger.info('Target distribution — positive (readmit <30d): %s  |  negative: %s  |  imbalance ratio: %.1f:1', f'{n_pos:,}', f'{n_neg:,}', ratio)
    return df

def validate_data(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError('Dataset is empty after loading.')
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COL}'.")
    dup_encounters = df['encounter_id'].duplicated().sum()
    if dup_encounters > 0:
        logger.warning('%s duplicate encounter IDs detected — these will be handled during preprocessing.', f'{dup_encounters:,}')
    logger.info('Data validation passed ✓')

def load_and_prepare(data_dir: Optional[Path]=None) -> pd.DataFrame:
    df = load_raw_data(data_dir)
    validate_data(df)
    df = create_binary_target(df)
    return df

def _download_and_extract(url: str, dest_dir: Path) -> None:
    import urllib.request
    dest_dir.mkdir(parents=True, exist_ok=True)
    logger.info('Downloading %s …', url)
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            raw_bytes = resp.read()
    except Exception as exc:
        raise RuntimeError(f'Failed to download dataset from {url}.  Please download manually and place the CSV in {dest_dir}.') from exc
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_files = [n for n in zf.namelist() if n.endswith('.csv')]
        if not csv_files:
            raise RuntimeError('No CSV files found inside the downloaded archive.')
        for name in csv_files:
            zf.extract(name, dest_dir)
            logger.info('Extracted %s → %s', name, dest_dir / name)