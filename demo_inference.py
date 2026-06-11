from __future__ import annotations
import pandas as pd
from src.inference import ReadmissionPredictor

def run_demo():
    print('=' * 80)
    print('Hospital Readmission Risk Predictor — Interactive Inference Demo')
    print('=' * 80)
    try:
        predictor = ReadmissionPredictor(pipeline_path='pipeline.pkl', threshold=0.086)
    except FileNotFoundError:
        print('Error: pipeline.pkl not found in the root directory. Please run run_pipeline.py first.')
        return
    print('\n1. Defining mock patient profiles...')
    patient_high_risk = {'race': 'AfricanAmerican', 'gender': 'Female', 'age': '[80-90)', 'admission_type_id': 1, 'discharge_disposition_id': 1, 'admission_source_id': 7, 'time_in_hospital': 8, 'num_lab_procedures': 65, 'num_procedures': 3, 'num_medications': 24, 'number_outpatient': 1, 'number_emergency': 2, 'number_inpatient': 3, 'number_diagnoses': 9, 'max_glu_serum': 'None', 'A1Cresult': '>8', 'change': 'Ch', 'diabetesMed': 'Yes', 'diag_1_cat': 'Circulatory', 'diag_2_cat': 'Diabetes', 'diag_3_cat': 'Respiratory', 'n_med_changed': 2, 'any_med_changed': 1, 'prior_visits_total': 6, 'age_ordinal': 8}
    patient_low_risk = {'race': 'Caucasian', 'gender': 'Male', 'age': '[40-50)', 'admission_type_id': 3, 'discharge_disposition_id': 1, 'admission_source_id': 1, 'time_in_hospital': 2, 'num_lab_procedures': 25, 'num_procedures': 0, 'num_medications': 8, 'number_outpatient': 0, 'number_emergency': 0, 'number_inpatient': 0, 'number_diagnoses': 3, 'max_glu_serum': 'None', 'A1Cresult': 'None', 'change': 'No', 'diabetesMed': 'No', 'diag_1_cat': 'Other', 'diag_2_cat': 'Other', 'diag_3_cat': 'Other', 'n_med_changed': 0, 'any_med_changed': 0, 'prior_visits_total': 0, 'age_ordinal': 4}
    print('\n2. Executing single-patient predictions...')
    res_high = predictor.predict_single(patient_high_risk)
    res_low = predictor.predict_single(patient_low_risk)
    print('\n--- PATIENT A (High Risk Profile) ---')
    print(f"Age/Gender:     {patient_high_risk['age']} {patient_high_risk['gender']}")
    print(f"Prior Visits:   {patient_high_risk['prior_visits_total']}")
    print(f"Primary Diag:   {patient_high_risk['diag_1_cat']}")
    print(f"Readmit Prob:   {res_high['probability'] * 100:.1f}%")
    print(f"Prediction:     {('READMISSION RISK DETECTED (1)' if res_high['prediction'] == 1 else 'NO RISK (0)')}")
    print(f"Risk Tier:      {res_high['risk_tier']}")
    print(f"Recommendation: {res_high['recommendation']}")
    print('\n--- PATIENT B (Low Risk Profile) ---')
    print(f"Age/Gender:     {patient_low_risk['age']} {patient_low_risk['gender']}")
    print(f"Prior Visits:   {patient_low_risk['prior_visits_total']}")
    print(f"Primary Diag:   {patient_low_risk['diag_1_cat']}")
    print(f"Readmit Prob:   {res_low['probability'] * 100:.1f}%")
    print(f"Prediction:     {('READMISSION RISK DETECTED (1)' if res_low['prediction'] == 1 else 'NO RISK (0)')}")
    print(f"Risk Tier:      {res_low['risk_tier']}")
    print(f"Recommendation: {res_low['recommendation']}")
    print('\n3. Executing batch prediction...')
    batch_df = pd.DataFrame([patient_high_risk, patient_low_risk])
    batch_results = predictor.predict_batch(batch_df)
    print('\nBatch Results DataFrame:')
    print(batch_results[['age', 'prior_visits_total', 'readmit_probability', 'readmit_prediction', 'risk_tier']])
    print('\n' + '=' * 80)
if __name__ == '__main__':
    run_demo()