# Architecture & Model Design

## Model Calibration
We implement Isotonic Regression calibration on top of Gradient Boosting ensembles. Uncalibrated models tend to output overconfident scores, whereas calibrated probabilities map directly to actual clinical readmission frequencies.

## Explanation Mechanics
Using SHAP tree explainers, we calculate local shapley values for every patient prediction. This attributes risk to specific inputs, ensuring clinical transparency.

Developed by [S. Manikanta Suryasai](https://github.com/Manirider)