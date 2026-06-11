# Fairness & Bias Audit Report

## Overview

This report evaluates whether the readmission risk model performs
equitably across demographic subgroups defined by **race**, **gender**,
and **age**.  Disparities can arise from data imbalance, systemic
healthcare inequities reflected in the training data, or feature
engineering choices.

## Subgroup Performance

| attribute   | group           |   precision |   recall |       f1 |      fpr |      fnr |   support |   positive_rate |
|:------------|:----------------|------------:|---------:|---------:|---------:|---------:|----------:|----------------:|
| race        | Caucasian       |    0.153186 | 0.791132 | 0.256672 | 0.575332 | 0.208868 |     14743 |       0.116259  |
| race        | AfricanAmerican |    0.148837 | 0.763723 | 0.249124 | 0.539823 | 0.236277 |      3809 |       0.110003  |
| race        | Unknown         |    0.122642 | 0.541667 | 0.2      | 0.435597 | 0.458333 |       475 |       0.101053  |
| race        | Hispanic        |    0.132653 | 0.8125   | 0.22807  | 0.449735 | 0.1875   |       410 |       0.0780488 |
| race        | Other           |    0.13125  | 0.7      | 0.221053 | 0.507299 | 0.3      |       304 |       0.0986842 |
| race        | Asian           |    0.271186 | 0.8      | 0.405063 | 0.398148 | 0.2      |       128 |       0.15625   |
| gender      | Female          |    0.145888 | 0.783694 | 0.245985 | 0.58366  | 0.216306 |     10651 |       0.112853  |
| gender      | Male            |    0.159126 | 0.775683 | 0.264078 | 0.533162 | 0.224317 |      9218 |       0.115101  |
| age         | [70-80)         |    0.142857 | 0.80895  | 0.242831 | 0.634278 | 0.19105  |      5027 |       0.115576  |
| age         | [60-70)         |    0.149002 | 0.759295 | 0.249117 | 0.556085 | 0.240705 |      4496 |       0.113657  |
| age         | [50-60)         |    0.152104 | 0.693215 | 0.249469 | 0.431063 | 0.306785 |      3378 |       0.100355  |
| age         | [80-90)         |    0.145285 | 0.890819 | 0.249826 | 0.73872  | 0.109181 |      3262 |       0.123544  |
| age         | [40-50)         |    0.198091 | 0.706383 | 0.309413 | 0.393443 | 0.293617 |      1943 |       0.120947  |
| age         | [30-40)         |    0.187919 | 0.777778 | 0.302703 | 0.372881 | 0.222222 |       721 |       0.0998613 |
| age         | [90-100)        |    0.141058 | 0.788732 | 0.239316 | 0.754425 | 0.211268 |       523 |       0.135755  |
| age         | [20-30)         |    0.196078 | 0.731707 | 0.309278 | 0.414141 | 0.268293 |       338 |       0.121302  |
| age         | [10-20)         |    0.151515 | 0.5      | 0.232558 | 0.205882 | 0.5      |       146 |       0.0684932 |
| age         | [0-10)          |    0        | 0        | 0        | 0        | 0        |        35 |       0         |

## ⚠ Significant Gaps Detected

| attribute   | metric    | group_high   | group_low   |      gap |
|:------------|:----------|:-------------|:------------|---------:|
| race        | recall    | Hispanic     | Unknown     | 0.270833 |
| race        | precision | Asian        | Unknown     | 0.148545 |
| age         | recall    | [80-90)      | [0-10)      | 0.890819 |
| age         | precision | [40-50)      | [0-10)      | 0.198091 |

## Discussion

### Potential Causes

1. **Data Imbalance** — Minority subgroups have fewer samples, leading
   to noisier metric estimates and potentially worse model fit.

2. **Healthcare Inequities** — The training data reflects real-world
   disparities in access, treatment, and documentation quality.
   Patients from underserved populations may have sparser clinical
   records, making their readmission risk harder to predict.

3. **Sampling Effects** — The dataset covers 130 US hospitals over
   1999–2008; geographic and temporal sampling may not represent the
   current patient population.

4. **Feature Bias** — Some features (e.g. number of prior visits) may
   proxy for insurance status or socioeconomic factors that correlate
   with race.

### Limitations

- Fairness metrics are computed on a single hold-out split; bootstrap
  confidence intervals would provide more robust estimates.
- We assess *group* fairness only; individual fairness and causal
  fairness require additional analysis.
- The `Unknown` race category may conflate multiple underrepresented
  groups.

### Recommendations

1. Compute bootstrap confidence intervals around subgroup metrics.
2. Investigate whether re-calibrating per subgroup improves equity.
3. Consider threshold-adjustment strategies for underserved groups.
4. Engage clinical and health-equity stakeholders before deployment.
5. Monitor subgroup performance in production with automated alerts.
