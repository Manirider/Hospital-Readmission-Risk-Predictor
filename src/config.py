from __future__ import annotations
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Any
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent / '.env')

def _env_int(key: str, default: int) -> int:
    return int(os.getenv(key, str(default)))

def _env_float(key: str, default: float) -> float:
    return float(os.getenv(key, str(default)))

def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / _env_str('DATA_DIR', 'data')
RAW_DATA_DIR = PROJECT_ROOT / _env_str('RAW_DATA_DIR', 'data/raw')
PROCESSED_DATA_DIR = PROJECT_ROOT / _env_str('PROCESSED_DATA_DIR', 'data/processed')
OUTPUT_DIR = PROJECT_ROOT / _env_str('OUTPUT_DIR', 'outputs')
MODEL_DIR = PROJECT_ROOT / _env_str('MODEL_DIR', '.')
for _d in (DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = _env_int('RANDOM_SEED', 42)
DATASET_URL = 'https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip'
DATASET_FILENAME = 'diabetic_data.csv'
TEST_SIZE = _env_float('TEST_SIZE', 0.2)
CV_FOLDS = _env_int('CV_FOLDS', 5)
N_ITER_SEARCH = _env_int('N_ITER_SEARCH', 50)
TARGET_COL = 'readmitted'
POSITIVE_LABEL = '<30'
BINARY_TARGET = 'readmit_30d'
LEAKAGE_DISCHARGE_IDS: List[int] = [11, 13, 14, 19, 20, 21]
DROP_COLS: List[str] = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'citoglipton', 'examide']
NUMERIC_FEATURES: List[str] = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']
CATEGORICAL_FEATURES: List[str] = ['race', 'gender', 'age', 'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed', 'diag_1_cat', 'diag_2_cat', 'diag_3_cat']
MEDICATION_FEATURES: List[str] = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide', 'insulin', 'glyburide-metformin', 'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone']
ICD9_RANGES: Dict[str, List[tuple]] = {'Circulatory': [(390, 459), (785, 785)], 'Respiratory': [(460, 519), (786, 786)], 'Digestive': [(520, 579), (787, 787)], 'Diabetes': [(250, 250)], 'Injury': [(800, 999)], 'Musculoskeletal': [(710, 739)], 'Genitourinary': [(580, 629), (788, 788)], 'Neoplasms': [(140, 239)], 'Mental Disorders': [(290, 319)]}

@dataclass
class HPGrid:
    gradient_boosting: Dict[str, Any] = field(default_factory=lambda : {'classifier__n_estimators': [100, 200, 300, 500], 'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2], 'classifier__max_depth': [3, 4, 5, 6], 'classifier__min_samples_split': [2, 5, 10, 20], 'classifier__min_samples_leaf': [1, 2, 4, 8], 'classifier__subsample': [0.7, 0.8, 0.9, 1.0]})
    random_forest: Dict[str, Any] = field(default_factory=lambda : {'classifier__n_estimators': [100, 200, 300, 500], 'classifier__max_depth': [5, 10, 15, 20, None], 'classifier__min_samples_split': [2, 5, 10, 20], 'classifier__min_samples_leaf': [1, 2, 4, 8], 'classifier__max_features': ['sqrt', 'log2', 0.3, 0.5], 'classifier__class_weight': ['balanced', 'balanced_subsample']})
    xgboost: Dict[str, Any] = field(default_factory=lambda : {'classifier__n_estimators': [100, 200, 300, 500], 'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2], 'classifier__max_depth': [3, 4, 5, 6, 8], 'classifier__min_child_weight': [1, 3, 5, 7], 'classifier__subsample': [0.7, 0.8, 0.9, 1.0], 'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9], 'classifier__scale_pos_weight': [1, 3, 5, 10]})
    lightgbm: Dict[str, Any] = field(default_factory=lambda : {'classifier__n_estimators': [100, 200, 300, 500], 'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2], 'classifier__max_depth': [3, 5, 7, -1], 'classifier__num_leaves': [15, 31, 63, 127], 'classifier__min_child_samples': [5, 10, 20, 50], 'classifier__subsample': [0.7, 0.8, 0.9, 1.0], 'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9], 'classifier__scale_pos_weight': [1, 3, 5, 10]})
HP_GRID = HPGrid()
LOG_LEVEL = _env_str('LOG_LEVEL', 'INFO')