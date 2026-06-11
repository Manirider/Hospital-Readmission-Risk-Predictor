from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from src.config import BINARY_TARGET, LEAKAGE_DISCHARGE_IDS
from src.data_loader import create_binary_target
from src.feature_engineering import fill_race_unknown, map_icd9_to_category, add_diagnosis_categories, add_medication_flags, add_utilisation_score, encode_age_bracket, drop_unwanted_columns
from src.leakage_detection import remove_leakage_rows, audit_leakage
from src.preprocessing import build_preprocessor, build_pipeline, resolve_feature_lists

@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({'encounter_id': [1, 2, 3, 4, 5], 'patient_nbr': [100, 101, 102, 103, 104], 'race': ['Caucasian', np.nan, 'AfricanAmerican', 'Caucasian', np.nan], 'gender': ['Female', 'Male', 'Female', 'Male', 'Female'], 'age': ['[50-60)', '[70-80)', '[60-70)', '[80-90)', '[40-50)'], 'weight': [np.nan, np.nan, np.nan, np.nan, np.nan], 'admission_type_id': [1, 2, 1, 3, 1], 'discharge_disposition_id': [1, 1, 11, 1, 13], 'admission_source_id': [7, 7, 4, 7, 7], 'time_in_hospital': [3, 5, 1, 7, 2], 'payer_code': [np.nan, 'MC', np.nan, 'SP', np.nan], 'medical_specialty': [np.nan, 'Cardiology', np.nan, np.nan, np.nan], 'num_lab_procedures': [40, 60, 20, 80, 30], 'num_procedures': [1, 3, 0, 2, 1], 'num_medications': [10, 15, 5, 20, 8], 'number_outpatient': [0, 1, 0, 3, 0], 'number_emergency': [0, 0, 1, 2, 0], 'number_inpatient': [0, 2, 0, 5, 0], 'number_diagnoses': [5, 9, 3, 12, 4], 'diag_1': ['250', '428', '486', '250.01', 'V58'], 'diag_2': ['401', '250', '599', '428', '250'], 'diag_3': ['599', '276', np.nan, '518', '401'], 'max_glu_serum': ['None', '>300', 'None', 'Norm', 'None'], 'A1Cresult': ['None', '>8', 'None', '>7', 'None'], 'metformin': ['No', 'Steady', 'No', 'Up', 'Steady'], 'insulin': ['No', 'Up', 'No', 'Steady', 'Down'], 'change': ['No', 'Ch', 'No', 'Ch', 'No'], 'diabetesMed': ['No', 'Yes', 'No', 'Yes', 'Yes'], 'readmitted': ['NO', '<30', '>30', '<30', 'NO'], 'citoglipton': ['No', 'No', 'No', 'No', 'No'], 'examide': ['No', 'No', 'No', 'No', 'No']})

class TestTargetEncoding:

    def test_binary_target_created(self, sample_df):
        result = create_binary_target(sample_df)
        assert BINARY_TARGET in result.columns

    def test_positive_label(self, sample_df):
        result = create_binary_target(sample_df)
        assert result[BINARY_TARGET].sum() == 2

    def test_negative_labels(self, sample_df):
        result = create_binary_target(sample_df)
        neg_count = (result[BINARY_TARGET] == 0).sum()
        assert neg_count == 3

class TestMissingValues:

    def test_race_filled_with_unknown(self, sample_df):
        result = fill_race_unknown(sample_df)
        assert result['race'].isna().sum() == 0
        assert (result['race'] == 'Unknown').sum() == 2

    def test_race_preserves_known(self, sample_df):
        result = fill_race_unknown(sample_df)
        assert result.loc[0, 'race'] == 'Caucasian'
        assert result.loc[2, 'race'] == 'AfricanAmerican'

class TestLeakageFiltering:

    def test_leakage_rows_removed(self, sample_df):
        df = create_binary_target(sample_df)
        result = remove_leakage_rows(df)
        assert 11 not in result['discharge_disposition_id'].values
        assert 13 not in result['discharge_disposition_id'].values

    def test_non_leakage_rows_preserved(self, sample_df):
        df = create_binary_target(sample_df)
        result = remove_leakage_rows(df)
        assert len(result) == 3

    def test_audit_counts(self, sample_df):
        df = create_binary_target(sample_df)
        counts = audit_leakage(df)
        assert counts['expired_patients'] == 1
        assert counts['hospice_patients'] == 1
        assert counts['total_leakage_rows'] == 2

class TestColumnSelection:

    def test_drop_unwanted(self, sample_df):
        df = create_binary_target(sample_df)
        df = add_diagnosis_categories(df)
        result = drop_unwanted_columns(df)
        assert 'encounter_id' not in result.columns
        assert 'patient_nbr' not in result.columns
        assert 'weight' not in result.columns

    def test_target_preserved(self, sample_df):
        df = create_binary_target(sample_df)
        df = add_diagnosis_categories(df)
        result = drop_unwanted_columns(df)
        assert BINARY_TARGET in result.columns

class TestPipelineConstruction:

    def test_build_pipeline_returns_pipeline(self):
        from sklearn.linear_model import LogisticRegression
        pipe = build_pipeline(LogisticRegression(), numeric_cols=['time_in_hospital'], categorical_cols=['race'])
        assert isinstance(pipe, Pipeline)

    def test_pipeline_has_preprocessor_and_classifier(self):
        from sklearn.linear_model import LogisticRegression
        pipe = build_pipeline(LogisticRegression(), numeric_cols=['time_in_hospital'], categorical_cols=['race'])
        assert 'preprocessor' in pipe.named_steps
        assert 'classifier' in pipe.named_steps