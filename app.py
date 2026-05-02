# FastAPI scoring app for the Credit Risk Scoring ML project.
import warnings
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

warnings.filterwarnings('ignore')

MODEL_PATH = Path('credit_risk_model.pkl')
SCALER_PATH = Path('credit_scaler.pkl')
FEATURE_NAMES_PATH = Path('feature_names.txt')
DECISION_THRESHOLD = 0.50

app = FastAPI(title='Credit Risk Scoring API', version='1.0.0')

class ApplicantFeatures(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0)
    age: float = Field(..., ge=18)
    NumberOfTime30_59DaysPastDueNotWorse: float = Field(..., alias='NumberOfTime30-59DaysPastDueNotWorse')
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: float
    NumberOfTimes90DaysLate: float
    NumberRealEstateLoansOrLines: float
    NumberOfTime60_89DaysPastDueNotWorse: float = Field(..., alias='NumberOfTime60-89DaysPastDueNotWorse')
    NumberOfDependents: float

# Load artifacts at startup so predictions are fast.
def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists() or not FEATURE_NAMES_PATH.exists():
        raise RuntimeError('Required artifacts are missing. Run the notebook save step first.')
    model = joblib.load(MODEL_PATH)
    sc = joblib.load(SCALER_PATH)
    fnames = FEATURE_NAMES_PATH.read_text(encoding='utf-8').splitlines()
    exp = shap.TreeExplainer(model)
    return model, sc, fnames, exp

try:
    model, scaler, feature_names, explainer = load_artifacts()
    startup_error = None
except Exception as exc:
    model, scaler, feature_names, explainer = None, None, None, None
    startup_error = str(exc)

def build_model_frame(payload: Dict[str, Any]) -> pd.DataFrame:
    row = dict(payload)
    row['DebtToIncomeRatio'] = row['DebtRatio'] * row['MonthlyIncome']
    row['TotalLatePayments'] = (
        row['NumberOfTime30-59DaysPastDueNotWorse']
        + row['NumberOfTime60-89DaysPastDueNotWorse']
        + row['NumberOfTimes90DaysLate']
    )
    row['CreditUtilizationFlag'] = int(row['RevolvingUtilizationOfUnsecuredLines'] > 1)
    row['IncomePerDependent'] = row['MonthlyIncome'] / (row['NumberOfDependents'] + 1)
    missing = [f for f in feature_names if f not in row]
    if missing:
        raise HTTPException(status_code=400, detail=f'Missing features: {missing}')
    return pd.DataFrame([{f: row[f] for f in feature_names}])

@app.get('/health')
def health():
    return {'status': 'ok' if startup_error is None else 'artifact_error', 'detail': startup_error}

@app.post('/predict')
def predict(applicant: ApplicantFeatures):
    if startup_error is not None:
        raise HTTPException(status_code=500, detail=startup_error)
    payload = applicant.model_dump(by_alias=True)
    applicant_df = build_model_frame(payload)
    applicant_scaled = pd.DataFrame(scaler.transform(applicant_df), columns=feature_names)
    default_probability = float(model.predict_proba(applicant_scaled)[:, 1][0])
    decision = 'deny' if default_probability >= DECISION_THRESHOLD else 'approve'
    shap_vals = explainer.shap_values(applicant_scaled)
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    top_idx = int(np.argmax(np.abs(sv[0])))
    top_reason = feature_names[top_idx]
    return {
        'default_probability': round(default_probability, 4),
        'decision': decision,
        'top_reason': top_reason,
    }
