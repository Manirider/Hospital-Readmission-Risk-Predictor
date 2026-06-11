from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
from src.config import LEAKAGE_DISCHARGE_IDS, OUTPUT_DIR, BINARY_TARGET
from src.utils import get_logger
logger = get_logger(__name__)
LEAKAGE_SOURCES: List[Dict[str, str]] = [{'feature': 'discharge_disposition_id (values 11, 19, 20, 21)', 'reason': 'These codes indicate that the patient **expired** (died) during the encounter.  A deceased patient cannot be readmitted, so keeping these rows would give the model a trivially-predictable negative class and inflate specificity.', 'action': 'Remove rows.'}, {'feature': 'discharge_disposition_id (values 13, 14)', 'reason': 'Codes 13 and 14 indicate discharge to **hospice** (home or medical facility).  Hospice patients have elected comfort-only care and are virtually never readmitted.  Including them introduces a near-deterministic signal unrelated to the clinical features we want the model to learn.', 'action': 'Remove rows.'}, {'feature': 'discharge_disposition_id (general)', 'reason': "Even after removing expired / hospice rows, the remaining discharge disposition values (e.g. 'Discharged to home' vs. 'Transferred to another short-term hospital') carry some post-encounter information.  However, this disposition is typically known *at* discharge — the same moment a care team would use the model — so retaining it is defensible for a discharge-time predictor.  We flag this decision explicitly.", 'action': 'Retain (with documentation).'}]

def audit_leakage(df: pd.DataFrame) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    expired_ids = [11, 19, 20, 21]
    mask_expired = df['discharge_disposition_id'].isin(expired_ids)
    counts['expired_patients'] = int(mask_expired.sum())
    hospice_ids = [13, 14]
    mask_hospice = df['discharge_disposition_id'].isin(hospice_ids)
    counts['hospice_patients'] = int(mask_hospice.sum())
    counts['total_leakage_rows'] = counts['expired_patients'] + counts['hospice_patients']
    counts['dataset_size_before'] = len(df)
    counts['dataset_size_after'] = len(df) - counts['total_leakage_rows']
    logger.info('Leakage audit — expired: %s | hospice: %s | total to remove: %s (%.2f%%)', f"{counts['expired_patients']:,}", f"{counts['hospice_patients']:,}", f"{counts['total_leakage_rows']:,}", 100 * counts['total_leakage_rows'] / max(len(df), 1))
    return counts

def remove_leakage_rows(df: pd.DataFrame) -> pd.DataFrame:
    n_before = len(df)
    mask = df['discharge_disposition_id'].isin(LEAKAGE_DISCHARGE_IDS)
    df_clean = df[~mask].copy()
    n_removed = n_before - len(df_clean)
    logger.info('Removed %s leakage rows (expired / hospice) — %s → %s rows', f'{n_removed:,}', f'{n_before:,}', f'{len(df_clean):,}')
    if BINARY_TARGET in df_clean.columns:
        pos_count = df_clean[BINARY_TARGET].sum()
        if pos_count == 0:
            raise ValueError('After leakage removal the positive class is empty — something went wrong.')
    return df_clean

def generate_leakage_report(audit_counts: Dict[str, int], output_dir: Path=OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / 'leakage_report.md'
    lines = ['# Data Leakage Audit Report', '', '## Summary', '', f'| Metric | Count |', f'|---|---|', f"| Dataset rows (before filtering) | {audit_counts['dataset_size_before']:,} |", f"| Expired patients removed | {audit_counts['expired_patients']:,} |", f"| Hospice patients removed | {audit_counts['hospice_patients']:,} |", f"| **Total rows removed** | **{audit_counts['total_leakage_rows']:,}** |", f"| Dataset rows (after filtering) | {audit_counts['dataset_size_after']:,} |", '', '## Leakage Sources', '']
    for src in LEAKAGE_SOURCES:
        lines += [f"### {src['feature']}", '', f"**Reason:** {src['reason']}", '', f"**Action:** {src['action']}", '']
    lines += ['## Rationale', '', 'In hospital readmission prediction the goal is to estimate, *at the', 'moment of discharge*, how likely a patient is to return within 30 days.', 'Any feature that encodes whether the patient *actually* returned — or', 'whether they were even *able* to return — constitutes information', 'leakage.  Deceased and hospice patients fall squarely into the latter', 'category: they cannot be readmitted, so including them gives the model', 'a cheap shortcut that would not generalise to the intended use case.', '', 'We retain the remaining `discharge_disposition_id` values because', 'discharge destination (e.g. home vs. skilled nursing facility) is known', 'at the point of discharge and is clinically relevant to readmission', 'risk.  This is a deliberate, documented design choice.', '']
    path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info('Leakage report saved → %s', path)
    return path