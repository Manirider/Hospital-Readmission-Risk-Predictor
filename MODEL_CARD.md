# Model Card: Hospital 30-Day Readmission Risk Predictor

## Model Overview

| Field | Value |
|-------|-------|
| **Model Name** | Hospital 30-Day Readmission Risk Predictor |
| **Version** | 1.0.0 |
| **Type** | Binary classification |
| **Architecture** | Gradient boosted trees (calibrated) within a scikit-learn Pipeline |
| **Framework** | scikit-learn 1.4.2, XGBoost 2.0.3, LightGBM 4.3.0 |
| **Task** | Predict whether a diabetic patient will be readmitted within 30 days of hospital discharge |
| **Last Updated** | June 2025 |

## Intended Use

### Primary Use Case

This model is designed for use at the **point of discharge** in US hospitals to identify diabetic patients at elevated risk of 30-day readmission.  It is intended to support — not replace — clinical decision-making by flagging high-risk patients for transitional care interventions such as:

- Enhanced discharge planning
- Post-discharge follow-up calls within 48 hours
- Medication reconciliation referrals
- Home health service coordination

### Intended Users

- **Hospital care managers** — for case prioritisation
- **Discharge planning teams** — for intervention targeting
- **Quality improvement departments** — for readmission reduction programmes

### Out-of-Scope Use

| Use Case | Reason |
|----------|--------|
| Automated denial of services | Model should inform, not deny |
| Patients without diabetes | Trained exclusively on diabetic cohort |
| Paediatric patients | Dataset covers ages 0–100 but is heavily skewed towards adults 50+ |
| Non-US hospital settings | Training data is from 130 US hospitals only |
| Real-time ICU monitoring | Not designed for continuous risk scoring |
| Individual clinical diagnosis | Predictions are probabilistic, not diagnostic |

## Dataset Description

### Source

**Diabetes 130-US Hospitals for Years 1999–2008** from the UCI Machine Learning Repository. This dataset contains ~100,000 inpatient encounters across 130 US hospitals over a 10-year period.

### Key Characteristics

| Attribute | Value |
|-----------|-------|
| Total encounters | ~101,766 |
| After leakage removal | ~98,000 |
| Features (raw) | 50 |
| Features (engineered) | ~25 |
| Positive class (readmit <30d) | ~11% |
| Time span | 1999–2008 |
| Geographic scope | 130 US hospitals |

### Known Limitations

- **Temporal gap**: Data is from 1999–2008; clinical practices have evolved significantly.
- **Selection bias**: Only includes encounters with diabetic diagnoses and hospital stays of 1–14 days.
- **Missing data**: Weight (97% missing), payer code (40%), and medical specialty (49%) have substantial missingness.
- **Race representation**: ~75% Caucasian; performance may be less reliable for underrepresented groups.

## Training Procedure

### Preprocessing

1. **Leakage removal**: Expired and hospice patients removed (structurally unable to be readmitted)
2. **Target encoding**: `readmitted == "<30"` → 1, else → 0
3. **Missing values**: Race → "Unknown" (fairness-preserving); numeric → median imputation; categorical → "missing" category
4. **Feature engineering**: ICD-9 → clinical categories, medication change aggregation, prior-visit composite score
5. **Encoding**: OneHotEncoder for categorical features, StandardScaler for numeric features

### Model Selection

Four candidate models were compared via 5-fold stratified cross-validation with RandomizedSearchCV (50 iterations), optimising ROC-AUC:

1. GradientBoostingClassifier
2. RandomForestClassifier
3. XGBClassifier
4. LGBMClassifier

### Calibration

The selected model was calibrated using `CalibratedClassifierCV` comparing sigmoid (Platt scaling) and isotonic regression.  The method with the lower Brier score on the hold-out set was selected.

### Threshold Optimisation

The default 0.5 threshold was replaced with an optimised threshold that maximises recall while maintaining precision ≥ 15%.  Clinical rationale: missing a readmission (FN) is far more costly than an unnecessary follow-up (FP).

## Metrics

### Primary Metrics (on hold-out test set)

Metrics are reported at the optimised threshold.  Exact values depend on the training run; representative values below.

| Metric | Value |
|--------|-------|
| ROC-AUC | ~0.64–0.68 |
| PR-AUC | ~0.18–0.22 |
| Brier Score (calibrated) | ~0.09 |
| Recall | ~0.50–0.65 (at optimised threshold) |
| Precision | ~0.15–0.20 (at optimised threshold) |
| F1 | ~0.22–0.28 |

### Interpretation

The model's ROC-AUC of ~0.65 is consistent with published benchmarks for this dataset and readmission prediction in general.  Readmission is influenced by many factors not captured in administrative data (social support, medication adherence, post-discharge events), which limits ceiling performance.

## Fairness Findings

### Audited Dimensions

- **Race**: Caucasian, AfricanAmerican, Hispanic, Asian, Other, Unknown
- **Gender**: Male, Female
- **Age**: 10 age brackets from [0-10) to [90-100)

### Key Observations

1. Recall and precision vary modestly across racial subgroups.  Performance is most stable for the majority group (Caucasian) due to sample size.
2. Gender-based disparities are minimal.
3. Age-based differences reflect genuine clinical variation: older patients (70–90) tend to have higher readmission rates and different prediction accuracy patterns.
4. An explicit "Unknown" race category was used instead of mode imputation to avoid masking missingness patterns that may correlate with socioeconomic factors.

### Mitigation Steps

- Class-balanced training to prevent majority-class bias
- Fairness-aware missing value handling for race
- Comprehensive bias audit with flagging of gaps >10%
- Recommendations for per-subgroup threshold calibration in production

## Ethical Considerations

### Potential Risks

1. **Over-reliance**: Clinicians may defer to model predictions instead of clinical judgement.
2. **Equity**: Performance disparities across racial groups could lead to unequal care quality.
3. **False negatives**: Missed high-risk patients may not receive needed follow-up.
4. **Labelling effects**: Being flagged as "high risk" could negatively affect patient experience or insurance interactions.

### Mitigations

- The model outputs a probability and risk tier, not a binary decision
- Designed as a decision-support tool, not a replacement for clinical assessment
- Threshold tuned to prioritise recall (catching more at-risk patients)
- Fairness audit conducted and documented

## Limitations

1. **Data age**: Training data spans 1999–2008; clinical workflows have changed substantially.
2. **Generalisability**: Performance on non-diabetic populations is unknown and likely poor.
3. **Missing context**: Social determinants of health, caregiver availability, and post-discharge medication adherence are not captured.
4. **Modest discrimination**: ROC-AUC ~0.65 reflects the inherent difficulty of readmission prediction from administrative data alone.
5. **Static model**: Does not update with new data; performance may degrade over time (concept drift).

## Monitoring Recommendations

| Aspect | Frequency | Method |
|--------|-----------|--------|
| Overall performance (ROC-AUC, recall) | Monthly | Compare against hold-out test set baseline |
| Subgroup performance (by race, age) | Quarterly | Automated fairness dashboard |
| Calibration drift | Monthly | Reliability diagram on recent predictions |
| Feature drift | Monthly | Distribution comparison of input features |
| Outcome labels | Quarterly | Verify that actual readmission rates align with predictions |

## Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | June 2025 | Initial release: GBT with calibration, SHAP explainability, fairness audit |

## Authors

Healthcare ML Team

## References

1. Strack, B., et al. (2014). "Impact of HbA1c measurement on hospital readmission rates: Analysis of 70,000 clinical database patient records." *BioMed Research International*.
2. UCI Machine Learning Repository: [Diabetes 130-US Hospitals](https://archive.ics.uci.edu/dataset/296)
3. Lundberg, S.M. & Lee, S.I. (2017). "A Unified Approach to Interpreting Model Predictions." *NeurIPS*.
4. CMS Hospital Readmissions Reduction Program (HRRP).
5. Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting good probabilities with supervised learning." *ICML*.
