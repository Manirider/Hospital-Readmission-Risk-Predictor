from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from src.feature_engineering import map_icd9_to_category, add_diagnosis_categories, add_medication_flags, add_utilisation_score, encode_age_bracket, AGE_BRACKET_MAP

class TestICD9Mapping:

    @pytest.mark.parametrize('code,expected', [('250', 'Diabetes'), ('250.01', 'Diabetes'), ('428', 'Circulatory'), ('410', 'Circulatory'), ('486', 'Respiratory'), ('540', 'Digestive'), ('820', 'Injury'), ('715', 'Musculoskeletal'), ('590', 'Genitourinary'), ('174', 'Neoplasms'), ('296', 'Mental Disorders'), ('780', 'Other'), ('V58', 'Other'), ('E879', 'Other'), (None, 'Other'), ('', 'Other')])
    def test_category_mapping(self, code, expected):
        assert map_icd9_to_category(code) == expected

    def test_boundary_circulatory_low(self):
        assert map_icd9_to_category('390') == 'Circulatory'

    def test_boundary_circulatory_high(self):
        assert map_icd9_to_category('459') == 'Circulatory'

    def test_boundary_diabetes(self):
        assert map_icd9_to_category('250.99') == 'Diabetes'

    def test_boundary_injury_low(self):
        assert map_icd9_to_category('800') == 'Injury'

    def test_boundary_injury_high(self):
        assert map_icd9_to_category('999') == 'Injury'

class TestDiagnosisCategories:

    def test_creates_category_columns(self):
        df = pd.DataFrame({'diag_1': ['250', '428'], 'diag_2': ['401', '250'], 'diag_3': ['599', np.nan]})
        result = add_diagnosis_categories(df)
        assert 'diag_1_cat' in result.columns
        assert 'diag_2_cat' in result.columns
        assert 'diag_3_cat' in result.columns

    def test_correct_mapping_in_dataframe(self):
        df = pd.DataFrame({'diag_1': ['250'], 'diag_2': ['428'], 'diag_3': ['486']})
        result = add_diagnosis_categories(df)
        assert result.loc[0, 'diag_1_cat'] == 'Diabetes'
        assert result.loc[0, 'diag_2_cat'] == 'Circulatory'
        assert result.loc[0, 'diag_3_cat'] == 'Respiratory'

    def test_handles_missing_gracefully(self):
        df = pd.DataFrame({'diag_1': [np.nan], 'diag_2': [None], 'diag_3': ['']})
        result = add_diagnosis_categories(df)
        assert result.loc[0, 'diag_1_cat'] == 'Other'
        assert result.loc[0, 'diag_2_cat'] == 'Other'

class TestMedicationFlags:

    def test_counts_changes(self):
        df = pd.DataFrame({'metformin': ['No', 'Up', 'Steady'], 'insulin': ['Down', 'No', 'Steady'], 'glipizide': ['No', 'No', 'Down']})
        result = add_medication_flags(df)
        assert result.loc[0, 'n_med_changed'] == 1
        assert result.loc[1, 'n_med_changed'] == 1
        assert result.loc[2, 'n_med_changed'] == 1

    def test_any_flag(self):
        df = pd.DataFrame({'metformin': ['No', 'Up'], 'insulin': ['Steady', 'No']})
        result = add_medication_flags(df)
        assert result.loc[0, 'any_med_changed'] == 0
        assert result.loc[1, 'any_med_changed'] == 1

class TestUtilisationScore:

    def test_sums_visits(self):
        df = pd.DataFrame({'number_inpatient': [1, 0, 3], 'number_outpatient': [2, 0, 1], 'number_emergency': [0, 1, 0]})
        result = add_utilisation_score(df)
        assert result.loc[0, 'prior_visits_total'] == 3
        assert result.loc[1, 'prior_visits_total'] == 1
        assert result.loc[2, 'prior_visits_total'] == 4

class TestAgeEncoding:

    def test_all_brackets(self):
        df = pd.DataFrame({'age': list(AGE_BRACKET_MAP.keys())})
        result = encode_age_bracket(df)
        for (bracket, ordinal) in AGE_BRACKET_MAP.items():
            row = result[result['age'] == bracket]
            assert row['age_ordinal'].values[0] == ordinal

    def test_unknown_bracket(self):
        df = pd.DataFrame({'age': ['[100-110)']})
        result = encode_age_bracket(df)
        assert result.loc[0, 'age_ordinal'] == -1