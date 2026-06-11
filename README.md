# 🏥 Clinical Readmission Risk Predictor

This repository implements a production-grade machine learning system designed to predict **30-day hospital readmission risk** using the Diabetes 130-US Hospitals dataset. Developed with a rigorous focus on clinical utility, algorithmic fairness, and model interpretability (SHAP), this project mirrors the architecture of AI systems deployed in modern healthcare systems to drive transitional care interventions.

## 📋 Clinical & Operational Context

In modern healthcare operations, reducing 30-day readmissions is a primary clinical and financial priority. Programs like the CMS **Hospital Readmissions Reduction Program (HRRP)** financially penalize institutions with higher-than-expected readmission rates. 

A predictive model serves as a clinical decision support tool at the point of discharge. By identifying high-risk patients before they leave the facility, care teams can deploy targeted interventions:
1. **Transitional Care Management (TCM)**: Pharmacist-led medication reconciliation, follow-up scheduling within 48 hours, and home-health referrals.
2. **Resource Allocation**: Directing intensive case-management resources to patients who benefit most.
3. **Asymmetric Error Rationale**: In readmission prevention, a **False Negative** (failing to identify a patient who will be readmitted) is far costlier than a **False Positive** (providing extra follow-up care to a stable patient). Thus, our system optimizes the operating threshold to maximize sensitivity (recall) while preserving a clinically viable precision floor.

## 🏗️ Technical Architecture & Design Decisions

The system is designed around a strictly serializable, leakage-free scikit-learn pipeline, ensuring identical data preprocessing in both batch training and real-time inference environments.

```
                                  [ Raw Clinical Data ]
                                            │
                                            ▼
                                  [ Data Leakage Filter ]  ◄── Excludes deceased & hospice patients
                                            │
                                            ▼
                                [ Clinical Feature Eng. ]  ◄── ICD-9 Chapters, Med Changes, Prior Visits
                                            │
                                            ▼
                               [ Pipeline Preprocessing ]  ◄── Scaling, Imputation, One-Hot Encoding
                                            │
                                            ▼
                                [ Calibrated Classifier ]  ◄── Isotonic Probability Calibration
                                            │
                                            ▼
                              [ Decision Threshold Tuning ] ◄── Optimized for Recall (Sensitivity)
                                            │
                                            ▼
                             [ Production Inference & SHAP ]
```

### Key Engineering Decisions
* **Strict Leakage Prevention**: We audit and remove encounters where patients expired or were discharged to hospice (discharge disposition IDs 11, 13, 14, 19, 20, 21). These patients cannot be readmitted; keeping them introduces a predictive shortcut that invalidates real-world performance.
* **Clinical Feature Representation**: Raw ICD-9 codes are mapped to their corresponding clinical chapters (e.g., Circulatory, Respiratory, Diabetes, Injury). We also generate composite markers for total prior visit intensity and active medication adjustments.
* **Probability Calibration**: Tree-based models tend to produce uncalibrated probabilities near the decision boundaries. We apply **Isotonic Regression** so the predicted risk scores align with actual readmission frequencies—a prerequisite for clinical decision support.
* **Fairness-Aware Preprocessing**: Missing demographic data (specifically `race`) is explicitly categorized as `"Unknown"` rather than imputed using the statistical mode. This prevents masking underlying disparities and lets the model learn missingness as a potential predictor.


## 📂 Repository Structure

The codebase is modular, stateless, and fully unit-tested:

```
hospital-readmission-predictor/
├── README.md                 # Project documentation
├── MODEL_CARD.md             # Standard model reporting card
├── requirements.txt          # Pinned package versions
├── Dockerfile                # Multi-stage production container
├── docker-compose.yml        # Docker composition orchestration
├── .env.example              # Template for environment settings
├── pipeline.pkl              # Persisted calibrated model pipeline
│
├── src/
│   ├── config.py             # Centralised hyperparameters and mappings
│   ├── data_loader.py        # UCI dataset downloader and parser
│   ├── preprocessing.py      # sklearn preprocessing pipelines
│   ├── feature_engineering.py # ICD-9 mappings and aggregate metrics
│   ├── leakage_detection.py  # Deceased/hospice patient auditing
│   ├── train.py              # Candidate model tuning and selection
│   ├── evaluate.py           # Standard metrics and plot generation
│   ├── calibrate.py          # Probability calibration wrapper
│   ├── threshold_optimizer.py # Custom recall-focused threshold finder
│   ├── explainability.py     # SHAP tree explainer integration
│   ├── fairness.py           # Subgroup demographic audits
│   ├── inference.py          # High-level API for production scoring
│   └── utils.py              # Logging, visualization style, and I/O
│
├── notebooks/                # Jupyter exploration templates
│   ├── 01_eda.ipynb          
│   ├── 02_feature_engineering.ipynb
│   ├── 03_modeling.ipynb     
│   ├── 04_explainability.ipynb
│   └── 05_bias_audit.ipynb   
│
├── outputs/                  # Exported charts, CSV audits, and reports
│   ├── roc_curve.png         # Model ROC performance
│   ├── pr_curve.png          # Precision-Recall trade-offs
│   ├── confusion_matrix.png  # Classification boundaries
│   ├── calibration_curve.png # Probability calibration before/after
│   ├── feature_importance.png# Scikit-learn impurity importances
│   ├── shap_summary.png      # Global SHAP beeswarm plot
│   ├── shap_force_*.png      # Local explanations (TP/FP/FN cases)
│   ├── bias_audit.csv        # Raw metrics by subgroup
│   ├── leakage_report.md     # Rationale for removed encounters
│   ├── fairness_report.md    # Subgroup audit analysis
│   └── threshold_rationale.md# Chosen operational decision point
│
└── tests/                    # Unit testing suite
    ├── test_preprocessing.py
    ├── test_features.py
    ├── test_pipeline.py
    └── test_inference.py
```

## 🚀 Getting Started

### Local Installation

1. Clone the repository and navigate to the directory:
   ```bash
   git clone https://github.com/your-org/hospital-readmission-predictor.git
   cd hospital-readmission-predictor
   ```

2. Create and activate a clean Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   # source .venv/bin/activate
   ```

3. Install the pinned dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure your environment variables:
   ```bash
   copy .env.example .env
   ```


## 🏋️ Pipeline Workflows

### 1. Programmatic Execution (End-to-End)
To download the dataset, clean the data, run hyperparameter tuning, calibrate probabilities, select features, and audit fairness, run the orchestrator script:
```bash
python run_pipeline.py
```
This script saves the final trained pipeline as `pipeline.pkl` in the root directory and writes all metrics, reports, and diagnostic charts to the `outputs/` directory.

### 2. Running Containerized Workflows (Docker)
Ensure the Docker daemon is active on your system, then use Docker Compose to run training, validation, or exploration:
```bash
# Run the end-to-end model training container
docker compose run train

# Execute the test suite inside the container
docker compose run test

# Launch the containerized JupyterLab server
docker compose up jupyter
```
*Note: JupyterLab will be exposed at `http://localhost:8888`.*

### 3. Running the Test Suite
The codebase is validated by 69 unit tests covering preprocessing transforms, feature engineering boundaries, model calibration correctness, and API edge-cases. To run the suite:
```bash
python -m pytest tests/ -v
```


## 🔍 Scoring Patients (Inference API)

The `ReadmissionPredictor` class provides a clean, deployment-ready interface for scoring new patient records. It handles single-record scoring and batch DataFrame scoring, automatically converting probabilities to clinical risk tiers and care recommendations.

```python
import pandas as pd
from src.inference import ReadmissionPredictor

# Initialise with the calibrated model and optimized threshold
predictor = ReadmissionPredictor(pipeline_path="pipeline.pkl", threshold=0.086)

# Predict risk for a single patient record
patient_data = {
    "race": "AfricanAmerican",
    "gender": "Female",
    "age": "[80-90)",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "time_in_hospital": 8,
    "num_lab_procedures": 65,
    "num_procedures": 3,
    "num_medications": 24,
    "number_outpatient": 1,
    "number_emergency": 2,
    "number_inpatient": 3,
    "number_diagnoses": 9,
    "max_glu_serum": "None",
    "A1Cresult": ">8",
    "change": "Ch",
    "diabetesMed": "Yes",
    "diag_1_cat": "Circulatory",
    "diag_2_cat": "Diabetes",
    "diag_3_cat": "Respiratory",
    "n_med_changed": 2,
    "any_med_changed": 1,
    "prior_visits_total": 6,
    "age_ordinal": 8,
}

result = predictor.predict_single(patient_data)
print(result)
# Output:
# {
#   'probability': 0.2764,
#   'prediction': 1,
#   'risk_tier': 'Moderate',
#   'recommendation': 'Schedule 48-hour post-discharge call.'
# }
```


## 📊 Core Performance & Audit Metrics

The training script evaluates candidate models and optimizes parameters. The selected champion model is a **Random Forest** calibrated via **Isotonic Regression**.

### Evaluation Summary
* **ROC-AUC**: `0.6532`
* **PR-AUC**: `0.2031`
* **Brier Score (Calibration)**: `0.0911` (indicating highly reliable risk probabilities)
* **Optimal Decision Threshold**: `0.086` (tuned to prioritize patient safety)
* **Performance at Operational Threshold**:
  * **Recall (Sensitivity)**: **`78.0%`** (detects 78% of readmitting patients)
  * **Precision**: **`15.2%`**
  * **F1-Score**: **`25.4%`**

### Summary of Subgroup Fairness Audit
A demographic audit was conducted across patient age brackets, gender, and racial identifiers.
* **Racial Gaps**: The model achieves high recall for Asian (`80.0%`) and Hispanic (`81.3%`) cohorts. However, a significant gap was detected for patients with `"Unknown"` race records, where recall dropped to `54.2%`. This highlights the importance of collecting complete clinical and demographic data during intake to prevent bias.
* **Age Gaps**: Recall was highest in the `[80-90)` age bracket (`89.1%`) and lowest in pediatric/younger adult categories, reflecting the dataset's focus on geriatric diabetic populations.

*Detailed diagnostic reports, data leakage findings, and fairness audit tables are located in the [outputs/](file:///c:/Users/lenovo/Downloads/hospital-readmission-predictor/outputs/) directory.*

## ⚠️ Limitations & Future Work

1. **Administrative Data Constraints**: The dataset contains administrative, billing, and demographic elements. The maximum ROC-AUC of ~0.65 is consistent with literature for models relying purely on administrative data without clinical notes or social determinants of health (SDOH).
2. **Temporal Shifts**: The training data spans 1999–2008. Clinical guidelines, Electronic Health Record (EHR) practices, and readmission reduction strategies have evolved significantly.
3. **Future EHR Integrations**: Transitioning this model to production would benefit from:
   * Integrating **FHIR (Fast Healthcare Interoperability Resources)** APIs to ingest real-time EHR data.
   * Applying NLP to extract clinical nuances (e.g., social support, cognitive status) from free-text discharge summaries.

## 📚 References

1. Strack, B., et al. (2014). "Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database records." *BioMed Research International*.
2. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
3. CMS Hospital Readmissions Reduction Program (HRRP) Guidelines.

# Author

*MANIKANTA SURYASAI*

*AIML ENGINEER | DEVELOPER*