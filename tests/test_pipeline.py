from __future__ import annotations
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from src.calibrate import expected_calibration_error
from src.preprocessing import build_pipeline
from src.threshold_optimizer import find_optimal_threshold
from src.utils import save_pipeline, load_pipeline

@pytest.fixture
def fitted_pipeline():
    np.random.seed(42)
    X = pd.DataFrame({'num_feat': np.random.randn(100), 'cat_feat': np.random.choice(['A', 'B', 'C'], 100)})
    y = np.random.randint(0, 2, 100)
    pipe = build_pipeline(LogisticRegression(random_state=42), numeric_cols=['num_feat'], categorical_cols=['cat_feat'])
    pipe.fit(X, y)
    return (pipe, X, y)

class TestPipelineSerialization:

    def test_save_and_load(self, fitted_pipeline, tmp_path):
        (pipe, X, _) = fitted_pipeline
        save_pipeline(pipe, 'test_pipe.pkl', tmp_path)
        loaded = load_pipeline('test_pipe.pkl', tmp_path)
        assert isinstance(loaded, Pipeline)

    def test_predictions_match_after_load(self, fitted_pipeline, tmp_path):
        (pipe, X, _) = fitted_pipeline
        preds_before = pipe.predict(X)
        save_pipeline(pipe, 'test_pipe.pkl', tmp_path)
        loaded = load_pipeline('test_pipe.pkl', tmp_path)
        preds_after = loaded.predict(X)
        np.testing.assert_array_equal(preds_before, preds_after)

    def test_probabilities_match_after_load(self, fitted_pipeline, tmp_path):
        (pipe, X, _) = fitted_pipeline
        probs_before = pipe.predict_proba(X)
        save_pipeline(pipe, 'test_pipe.pkl', tmp_path)
        loaded = load_pipeline('test_pipe.pkl', tmp_path)
        probs_after = loaded.predict_proba(X)
        np.testing.assert_array_almost_equal(probs_before, probs_after)

    def test_load_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline('nonexistent.pkl', tmp_path)

class TestPredictionShapes:

    def test_predict_shape(self, fitted_pipeline):
        (pipe, X, _) = fitted_pipeline
        preds = pipe.predict(X)
        assert preds.shape == (len(X),)

    def test_predict_proba_shape(self, fitted_pipeline):
        (pipe, X, _) = fitted_pipeline
        probs = pipe.predict_proba(X)
        assert probs.shape == (len(X), 2)

    def test_probabilities_sum_to_one(self, fitted_pipeline):
        (pipe, X, _) = fitted_pipeline
        probs = pipe.predict_proba(X)
        np.testing.assert_array_almost_equal(probs.sum(axis=1), 1.0)

    def test_probabilities_in_range(self, fitted_pipeline):
        (pipe, X, _) = fitted_pipeline
        probs = pipe.predict_proba(X)
        assert (probs >= 0).all()
        assert (probs <= 1).all()

class TestECE:

    def test_perfect_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0])
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert 0 <= ece <= 1

    def test_worst_calibration(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_prob = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
        ece = expected_calibration_error(y_true, y_prob, n_bins=5)
        assert ece > 0.5

    def test_ece_range(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.random.rand(500)
        ece = expected_calibration_error(y_true, y_prob)
        assert 0 <= ece <= 1

class TestThresholdOptimization:

    def test_returns_valid_threshold(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.random.rand(500)
        result = find_optimal_threshold(y_true, y_prob, strategy='max_f1')
        assert 0 < result['threshold'] < 1
        assert 0 <= result['precision'] <= 1
        assert 0 <= result['recall'] <= 1
        assert 0 <= result['f1'] <= 1

    def test_max_recall_strategy(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 500)
        y_prob = np.random.rand(500)
        result = find_optimal_threshold(y_true, y_prob, strategy='max_recall')
        assert result['recall'] > 0

    def test_invalid_strategy_raises(self):
        y_true = np.array([0, 1])
        y_prob = np.array([0.3, 0.7])
        with pytest.raises(ValueError, match='Unknown strategy'):
            find_optimal_threshold(y_true, y_prob, strategy='invalid')