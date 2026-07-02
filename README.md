# Hospital Readmission Risk Predictor

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat-square&logo=python&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white) ![SHAP](https://img.shields.io/badge/SHAP-000000?style=flat-square&logoColor=white) ![License](https://img.shields.io/github/license/Manirider/Hospital-Readmission-Risk-Predictor?style=flat-square)

An end-to-end predictive healthcare AI system designed to assess the 30-day hospital readmission risk of patients, incorporating SHAP explainability, model calibration, and fairness audits.

`healthcare-ai` `machine-learning` `shap` `fairness` `scikit-learn` `python` `clinical-ml` `model-interpretability`

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Motivation & Objectives](#motivation--objectives)
- [Core Features](#core-features)
- [System Architecture](#system-architecture)
- [Folder Structure](#folder-structure)
- [Installation & Local Setup](#installation--local-setup)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Model Interpretability (SHAP)](#model-interpretability-shap)
- [Fairness Auditing](#fairness-auditing)
- [Testing & Validation](#testing--validation)
- [Contributing](#contributing)
- [License](#license)

## Overview

Hospital Readmission Risk Predictor provides clinicians with validated, transparent predictions of 30-day patient readmission risks. Designed around the complexities of Electronic Health Records (EHR), the project implements robust data preprocessing, probability calibration, and localized model explainability to support clinical decision-making.

## Problem Statement

Predictive models in healthcare face unique validation and trust challenges:
- **Trust Deficits:** Black-box models are difficult to integrate into clinical workflows without clear explanation metrics.
- **Uncalibrated Predictions:** Raw model scores often overestimate or underestimate risk, compromising clinical decisions.
- **Biased Outcomes:** Algorithmic bias can prioritize certain patient cohorts over others, exacerbating disparities.

## Motivation & Objectives

This repository targets three major goals:
- **Interpretability:** Generate local and global explainability metrics for every classification prediction using SHAP.
- **Calibration Accuracy:** Calibrate output probabilities to ensure scores reflect actual patient readmission rates.
- **Fairness Compliance:** Provide demographic audits to verify model performance consistency across age and gender subgroups.

## Core Features

- **Clinical Feature Ingestion:** Specialized preprocessors handling missing data and high-cardinality codes.
- **Calibrated Classifier Pipeline:** Gradient Boosting classifiers calibrated via Isotonic Regression.
- **SHAP Explanation Suite:** Utilities plotting patient-level force plots and global feature importances.
- **Fairness Audits:** Metric modules checking equalized odds and demographic parity parameters.
- **Interactive Dashboards:** Streamlit interface display risk scores alongside SHAP explanations.

## System Architecture

The clinical prediction pipeline operates as follows:
- Data Prep cleans raw patient records, encoding medical variables.
- The Estimator runs classification pipelines, applying probability calibration layers.
- The SHAP explainer processes features to calculate local contribution weights.
- The Evaluation module audits demographic subgroups for fairness consistency.
- The UI exposes predictions and feature importances to clinicians.

## Folder Structure

```
Hospital-Readmission-Risk-Predictor/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── SECURITY.md
├── ARCHITECTURE.md
├── API.md
├── ROADMAP.md
├── DEPLOYMENT.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── model.py
│   ├── explain.py
│   ├── fairness.py
│   └── app.py
├── tests/
│   ├── __init__.py
│   └── test_model.py
└── .github/
    └── workflows/
        └── ci.yml
```

## Installation & Local Setup

```bash
# Clone the repository
git clone https://github.com/Manirider/Hospital-Readmission-Risk-Predictor.git
cd Hospital-Readmission-Risk-Predictor

# Set up virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage Guide

### 1. Data Prep and Training

Pre-process datasets and train the calibrated model:

```bash
python src/model.py --data data/patients.csv --output models/calibrated_model.joblib
```

### 2. Launch Explainability Dashboard

Start the Streamlit dashboard to explore predictions and SHAP visualizations:

```bash
streamlit run src/app.py
```

## API Documentation

The inference wrapper provides:

#### `predict_patient_risk(features: dict) -> dict`
- **Input:** Patient clinical metrics (diagnoses, lab scores, visits).
- **Output:** Predicted readmission probability and local SHAP vectors.

## Model Interpretability (SHAP)

Every prediction generates a local SHAP force plot. This visualization shows which variables pushed the model's prediction higher (e.g., number of recent inpatient admissions) and which pulled it lower (e.g., stable lab results).

## Fairness Auditing

The system evaluates:
- **Demographic Parity:** Verifying similar selection rates across age groups.
- **Equalized Odds:** Confirming consistent false positive and true positive rates across gender cohorts.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)