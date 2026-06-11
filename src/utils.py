from __future__ import annotations
import logging
import random
import sys
from pathlib import Path
from typing import Optional
import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from src.config import LOG_LEVEL, RANDOM_SEED, OUTPUT_DIR, MODEL_DIR

def get_logger(name: str, level: Optional[str]=None) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter('%(asctime)s | %(name)-24s | %(levelname)-7s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, (level or LOG_LEVEL).upper(), logging.INFO))
    return logger

def set_seed(seed: int=RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
PALETTE = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B', '#44BBA4', '#E94F37', '#393E41']

def set_plot_style() -> None:
    sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1, rc={'figure.figsize': (10, 6), 'axes.titlesize': 14, 'axes.labelsize': 12, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 10, 'figure.dpi': 120, 'savefig.dpi': 150, 'savefig.bbox': 'tight', 'font.family': 'sans-serif'})
    sns.set_palette(PALETTE)

def save_figure(fig: plt.Figure, filename: str, output_dir: Path=OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger = get_logger(__name__)
    logger.info('Saved figure → %s', path)
    return path

def save_pipeline(pipeline, filename: str='pipeline.pkl', model_dir: Path=MODEL_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / filename
    joblib.dump(pipeline, path)
    logger = get_logger(__name__)
    logger.info('Saved pipeline → %s  (%.1f MB)', path, path.stat().st_size / 1000000.0)
    return path

def load_pipeline(filename: str='pipeline.pkl', model_dir: Path=MODEL_DIR):
    path = model_dir / filename
    if not path.exists():
        raise FileNotFoundError(f'Pipeline not found at {path}')
    pipeline = joblib.load(path)
    logger = get_logger(__name__)
    logger.info('Loaded pipeline ← %s', path)
    return pipeline