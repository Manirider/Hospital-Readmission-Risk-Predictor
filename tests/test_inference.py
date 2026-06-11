from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.inference import ReadmissionPredictor, categorise_risk, validate_input

class TestRiskCategorisation:

    def test_low_risk(self):
        result = categorise_risk(0.05)
        assert result['tier'] == 'Low'

    def test_moderate_risk(self):
        result = categorise_risk(0.2)
        assert result['tier'] == 'Moderate'

    def test_high_risk(self):
        result = categorise_risk(0.35)
        assert result['tier'] == 'High'

    def test_very_high_risk(self):
        result = categorise_risk(0.6)
        assert result['tier'] == 'Very High'

    def test_boundary_low_moderate(self):
        result = categorise_risk(0.15)
        assert result['tier'] == 'Moderate'

    def test_boundary_moderate_high(self):
        result = categorise_risk(0.3)
        assert result['tier'] == 'High'

    def test_boundary_high_very_high(self):
        result = categorise_risk(0.5)
        assert result['tier'] == 'Very High'

    def test_zero_probability(self):
        result = categorise_risk(0.0)
        assert result['tier'] == 'Low'

    def test_recommendation_not_empty(self):
        for prob in [0.05, 0.2, 0.35, 0.7]:
            result = categorise_risk(prob)
            assert len(result['recommendation']) > 0

class TestInputValidation:

    def test_empty_dataframe_raises(self):
        with pytest.raises(ValueError, match='empty'):
            validate_input(pd.DataFrame())

    def test_valid_input_passes(self):
        df = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
        validate_input(df)

    def test_all_nan_warning(self, caplog):
        import logging
        df = pd.DataFrame({'col1': [np.nan], 'col2': [np.nan]})
        with caplog.at_level(logging.WARNING):
            validate_input(df)

class TestReadmissionPredictor:

    @pytest.fixture
    def predictor(self):
        try:
            return ReadmissionPredictor()
        except FileNotFoundError:
            pytest.skip('pipeline.pkl not found — skipping integration tests')

    def test_single_prediction_structure(self, predictor):
        patient = {'race': 'Caucasian', 'gender': 'Female', 'age': '[70-80)', 'admission_type_id': 1, 'discharge_disposition_id': 1, 'admission_source_id': 7, 'time_in_hospital': 5, 'num_lab_procedures': 50, 'num_procedures': 2, 'num_medications': 15, 'number_outpatient': 0, 'number_emergency': 0, 'number_inpatient': 1, 'number_diagnoses': 7, 'max_glu_serum': 'None', 'A1Cresult': 'None', 'change': 'Ch', 'diabetesMed': 'Yes', 'diag_1_cat': 'Circulatory', 'diag_2_cat': 'Diabetes', 'diag_3_cat': 'Other', 'n_med_changed': 1, 'any_med_changed': 1, 'prior_visits_total': 1, 'age_ordinal': 7}
        result = predictor.predict_single(patient)
        assert 'probability' in result
        assert 'prediction' in result
        assert 'risk_tier' in result
        assert 'recommendation' in result
        assert 0 <= result['probability'] <= 1
        assert result['prediction'] in (0, 1)

    def test_batch_prediction_adds_columns(self, predictor):
        patients = pd.DataFrame([{'race': 'Caucasian', 'gender': 'Female', 'age': '[70-80)', 'admission_type_id': 1, 'discharge_disposition_id': 1, 'admission_source_id': 7, 'time_in_hospital': 5, 'num_lab_procedures': 50, 'num_procedures': 2, 'num_medications': 15, 'number_outpatient': 0, 'number_emergency': 0, 'number_inpatient': 1, 'number_diagnoses': 7, 'max_glu_serum': 'None', 'A1Cresult': 'None', 'change': 'Ch', 'diabetesMed': 'Yes', 'diag_1_cat': 'Circulatory', 'diag_2_cat': 'Diabetes', 'diag_3_cat': 'Other', 'n_med_changed': 1, 'any_med_changed': 1, 'prior_visits_total': 1, 'age_ordinal': 7}])
        result = predictor.predict_batch(patients)
        assert 'readmit_probability' in result.columns
        assert 'readmit_prediction' in result.columns
        assert 'risk_tier' in result.columns