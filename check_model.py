import joblib
import pandas as pd
import os

model_path = r"e:\paco_ml\models\win_model.joblib"
print("Checking model path:", model_path, os.path.exists(model_path))
if not os.path.exists(model_path):
    raise SystemExit("Model not found: " + model_path)

m = joblib.load(model_path)
print("Loaded model type:", type(m))
print("Named steps (if pipeline):", getattr(m, 'named_steps', None))

NUMERIC_COLS = [
    "opportunity_id",
    "opportunity_value",
    "proposal_score",
    "teaming_strength",
    "partner_past_win_rate",
    "agency_past_win_rate",
    "submission_speed_days",
    "team_size",
    "technical_score",
    "bd_lead_experience_years",
    "project_complexity_score",
    "deadline_urgency_days",
]

CATEGORICAL_COLS = [
    "agency",
    "partner",
    "prime_sub",
    "risk_level",
    "competition_level",
    "status",
]

# Build a single-row DataFrame with default values (numeric=0, categorical empty string)
row = {}
for c in NUMERIC_COLS:
    row[c] = 0
for c in CATEGORICAL_COLS:
    row[c] = ""

df = pd.DataFrame([row])
print("Input df columns:", df.columns.tolist())

# Try prediction
try:
    pred = m.predict(df)
    print("Prediction:", pred)
except Exception as e:
    print("Prediction error:", repr(e))

try:
    if hasattr(m, 'predict_proba'):
        proba = m.predict_proba(df)
        print("Predict_proba shape:", getattr(proba, 'shape', None))
        print("Predict_proba sample:", proba.tolist())
    else:
        print("Model has no predict_proba")
except Exception as e:
    print("Predict_proba error:", repr(e))
