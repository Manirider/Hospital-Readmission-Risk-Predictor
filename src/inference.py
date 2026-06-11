from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from src.config import MODEL_DIR
from src.utils import get_logger, load_pipeline
logger = get_logger(__name__)
RISK_TIERS = [(0.0, 0.15, 'Low', 'Routine discharge plan; standard follow-up.'), (0.15, 0.3, 'Moderate', 'Schedule 48-hour post-discharge call.'), (0.3, 0.5, 'High', 'Refer to transitional care team.'), (0.5, 1.01, 'Very High', 'Intensive case management and follow-up within 24h.')]

def categorise_risk(probability: float) -> Dict[str, str]:
    for (lo, hi, tier, rec) in RISK_TIERS:
        if lo <= probability < hi:
            return {'tier': tier, 'recommendation': rec}
    return {'tier': 'Unknown', 'recommendation': 'Review manually.'}

def validate_input(data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError('Input dataframe is empty.')
    all_nan_rows = data.isna().all(axis=1).sum()
    if all_nan_rows > 0:
        logger.warning('%s rows are entirely NaN — predictions will be unreliable.', all_nan_rows)

class ReadmissionPredictor:

    def __init__(self, pipeline_path: Optional[Union[str, Path]]=None, threshold: float=0.3) -> None:
        if pipeline_path is None:
            self.pipeline = load_pipeline()
        else:
            import joblib
            self.pipeline = joblib.load(pipeline_path)
        self.threshold = threshold
        logger.info('ReadmissionPredictor initialised — threshold=%.2f', self.threshold)

    def predict_single(self, patient: Dict[str, Any]) -> Dict[str, Any]:
        df = pd.DataFrame([patient])
        validate_input(df)
        probability = float(self.pipeline.predict_proba(df)[:, 1][0])
        prediction = int(probability >= self.threshold)
        risk = categorise_risk(probability)
        result = {'probability': round(probability, 4), 'prediction': prediction, 'risk_tier': risk['tier'], 'recommendation': risk['recommendation']}
        logger.info('Single prediction — P(readmit)=%.3f  tier=%s', probability, risk['tier'])
        return result

    def predict_batch(self, patients: pd.DataFrame) -> pd.DataFrame:
        validate_input(patients)
        probabilities = self.pipeline.predict_proba(patients)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)
        result = patients.copy()
        result['readmit_probability'] = np.round(probabilities, 4)
        result['readmit_prediction'] = predictions
        result['risk_tier'] = [categorise_risk(p)['tier'] for p in probabilities]
        result['recommendation'] = [categorise_risk(p)['recommendation'] for p in probabilities]
        logger.info('Batch prediction — %s patients | flagged: %s (%.1f%%)', len(patients), predictions.sum(), 100 * predictions.mean())
        return result