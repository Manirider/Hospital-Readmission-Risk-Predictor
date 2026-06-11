from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
from src.data_loader import load_and_prepare
from src.leakage_detection import remove_leakage_rows, audit_leakage, generate_leakage_report
from src.feature_engineering import engineer_features
from src.train import run_training
from src.calibrate import compare_calibration_methods, plot_calibration_curve
from src.threshold_optimizer import find_optimal_threshold, plot_threshold_analysis, document_threshold_rationale
from src.evaluate import evaluate_model
from src.explainability import compute_shap_values, plot_shap_summary, generate_local_explanations
from src.fairness import run_fairness_audit, detect_significant_gaps, generate_fairness_report, plot_fairness_comparison
from src.utils import save_pipeline, get_logger
logger = get_logger('run_pipeline')

def main():
    logger.info('Starting end-to-end ML pipeline execution...')
    logger.info('1. Loading raw clinical dataset...')
    df = load_and_prepare()
    logger.info('2. Performing data leakage audit...')
    leakage_counts = audit_leakage(df)
    generate_leakage_report(leakage_counts)
    df = remove_leakage_rows(df)
    logger.info('3. Engineering clinical features...')
    df = engineer_features(df)
    logger.info('4. Training and tuning candidate models...')
    train_results = run_training(df)
    best_name = train_results['best_name']
    best_pipeline = train_results['best_pipeline']
    X_train = train_results['X_train']
    X_test = train_results['X_test']
    y_train = train_results['y_train']
    y_test = train_results['y_test']
    logger.info('5. Calibrating probabilities of the selected best model (%s)...', best_name)
    (best_cal_method, calibrated_pipeline) = compare_calibration_methods(best_pipeline, X_train, y_train, X_test, y_test)
    y_prob_uncal = best_pipeline.predict_proba(X_test)[:, 1]
    y_prob_cal = calibrated_pipeline.predict_proba(X_test)[:, 1]
    plot_calibration_curve(y_test, y_prob_uncal, y_prob_cal)
    logger.info('6. Optimizing decision threshold based on clinical costs...')
    thresh_results = find_optimal_threshold(y_test.values, y_prob_cal, min_precision=0.15, strategy='max_recall')
    chosen_threshold = thresh_results['threshold']
    plot_threshold_analysis(y_test.values, y_prob_cal, chosen_threshold)
    document_threshold_rationale(thresh_results)
    logger.info('7. Evaluating calibrated model at optimized threshold (%.3f)...', chosen_threshold)
    eval_results = evaluate_model(calibrated_pipeline, X_test, y_test, threshold=chosen_threshold)
    logger.info('8. Generating SHAP explanations...')
    (shap_values, X_transformed, feature_names) = compute_shap_values(calibrated_pipeline, X_test, max_samples=500)
    plot_shap_summary(shap_values, X_transformed, feature_names)
    y_pred = (y_prob_cal >= chosen_threshold).astype(int)
    generate_local_explanations(calibrated_pipeline, X_test, y_test, y_pred, shap_values, X_transformed, feature_names)
    logger.info('9. Auditing fairness across demographic groups...')
    audit_df = run_fairness_audit(X_test, y_test.values, y_pred)
    gaps_df = detect_significant_gaps(audit_df)
    generate_fairness_report(audit_df, gaps_df)
    plot_fairness_comparison(audit_df, attribute='race')
    plot_fairness_comparison(audit_df, attribute='gender')
    plot_fairness_comparison(audit_df, attribute='age')
    logger.info('10. Persisting the final calibrated pipeline...')
    save_pipeline(calibrated_pipeline, filename='pipeline.pkl')
    logger.info('End-to-end execution completed successfully! Calibrated pipeline saved as pipeline.pkl.')
if __name__ == '__main__':
    main()