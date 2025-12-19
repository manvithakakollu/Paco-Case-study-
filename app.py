"""Unified Streamlit UI for all trained models.

Features:
- Allows selecting one of the four models: Win Probability, Project Cost Overrun,
  Time-to-Fill, Employee Exit Risk.
- Auto-detects expected input features from each saved pipeline (when possible)
  and builds a form dynamically.
- Displays predictions appropriate to each model type (probabilities for
  classifiers, numeric value for regressors). For Prophet-based time-series
  models it shows the saved forecast object.

Run with:
    streamlit run e:\\paco_ml\\app.py
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import json

# Page configuration and basic branding
st.set_page_config(page_title="Paco Reputation Earned", page_icon=":trophy:", layout="wide")

# Small inline SVG logo used in the banner (keeps repo free of binary assets)
_PACO_SVG = '''
<svg width="120" height="40" viewBox="0 0 120 40" xmlns="http://www.w3.org/2000/svg">
    <rect rx="8" ry="8" width="120" height="40" fill="#0b5fff" />
    <rect rx="8" ry="8" width="120" height="40" fill="#ffd600" />
    <text x="60" y="25" font-family="Arial, Helvetica, sans-serif" font-size="13" font-weight="700" fill="#222222" text-anchor="middle">PACO</text>
    <circle cx="100" cy="10" r="6" fill="#ffffff" />
</svg>
'''

# Custom CSS to adjust header and sidebar styling
_CUSTOM_CSS = """
<style>
/* Page background */
body {
    background-color: #f3f6ff;
}
.stApp header {visibility: hidden}
.banner {
    display: flex;
    align-items: center;
    gap: 16px;
    background: linear-gradient(90deg,#ffd600,#ffbf00);
    color: #222222;
    padding: 18px;
    border-radius: 8px;
}
.banner h1 { margin: 0; font-size: 28px; }
.banner p { margin: 0; opacity: 0.9 }
.stSidebar .css-1d391kg { background-color: #ffffffee; }
.card {
    background: white;
    border-radius: 8px;
    padding: 12px 16px;
    box-shadow: 0 1px 6px rgba(20,40,80,0.06);
}
</style>
"""


MODEL_FILES = {
    "Project Success Probability": "models/project_success_model.joblib",
    "Win Probability": "models/win_model.joblib",
    "Project Cost Overrun": "models/project_cost_model.joblib",
    "Time-to-Fill": "models/time_to_fill_model.joblib",
    "Employee Exit Risk": "models/exit_risk_model.joblib",
}


@st.cache_resource
def load_model(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def infer_columns_from_pipeline(pipeline) -> Tuple[List[str], List[str]]:
    """Return (numeric_cols, categorical_cols) inferred from a sklearn Pipeline's preprocessor."""
    numeric_cols: List[str] = []
    categorical_cols: List[str] = []
    pre = None
    if hasattr(pipeline, "named_steps"):
        pre = pipeline.named_steps.get("pre")
    if pre is None:
        return numeric_cols, categorical_cols

    # ColumnTransformer stores transformers as list of (name, transformer, columns)
    for item in getattr(pre, "transformers", []):
        if len(item) >= 3:
            name = item[0]
            columns = item[2]
            # normalize columns
            try:
                if isinstance(columns, (list, tuple)):
                    col_list = list(columns)
                else:
                    col_list = list(columns)
            except Exception:
                col_list = [columns]

            # Use transformer name to decide type when possible (common names: 'num','cat')
            is_numeric = False
            try:
                if isinstance(name, str) and "num" in name.lower():
                    is_numeric = True
            except Exception:
                pass

            for c in col_list:
                if is_numeric:
                    numeric_cols.append(str(c))
                else:
                    categorical_cols.append(str(c))

    # dedupe
    numeric_cols = list(dict.fromkeys(numeric_cols))
    categorical_cols = list(dict.fromkeys(categorical_cols))
    return numeric_cols, categorical_cols


def clean_label(name: str) -> str:
    """Return a user-friendly label for column names."""
    if name is None:
        return ""
    s = str(name)
    # replace underscores and multiple spaces, trim
    s = s.replace('_', ' ')
    s = ' '.join(s.split())
    return s.title()


def load_sample_for_model(model_name: str) -> Optional[pd.DataFrame]:
    mapping = {
        "Project Success Probability": "data/project_cost_overrun_data.csv",
        "Win Probability": "data/win_probability_data.csv",
        "Project Cost Overrun": "data/project_cost_overrun_data.csv",
        "Time-to-Fill": "data/time_to_fill_data.csv",
        "Employee Exit Risk": "data/employee_exit_risk_data.csv",
    }
    path = mapping.get(model_name)
    if path and os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            return None
    return None


def load_manifest_for_model(model_path: str) -> Optional[dict]:
    """Load an inputs manifest next to a saved model if present.

    The manifest file is expected to be named like the model with
    `_inputs.json` suffix, e.g. `project_success_model_inputs.json`.
    """
    base, _ = os.path.splitext(model_path)
    manifest_path = base + '_inputs.json'
    if os.path.exists(manifest_path):
        try:
            return json.load(open(manifest_path, 'r'))
        except Exception:
            return None
    return None


def build_inputs(numeric_cols: List[str], categorical_cols: List[str], sample: Optional[pd.DataFrame]):
    st.sidebar.header("Model Inputs")
    inputs: Dict[str, object] = {}

    medians: Dict[str, float] = {}
    uniques: Dict[str, List[str]] = {}
    if sample is not None:
        for c in numeric_cols:
            if c in sample.columns:
                # compute median only for numeric-like columns
                try:
                    if pd.api.types.is_numeric_dtype(sample[c]):
                        med = sample[c].median(skipna=True)
                        medians[c] = float(med if pd.notna(med) else 0)
                    else:
                        # try coercing to numeric and compute median
                        coerced = pd.to_numeric(sample[c], errors="coerce").dropna()
                        if len(coerced) > 0:
                            med = coerced.median()
                            medians[c] = float(med)
                        else:
                            medians[c] = 0.0
                except Exception:
                    medians[c] = 0.0
        for c in categorical_cols:
            if c in sample.columns:
                uniques[c] = list(sample[c].dropna().astype(str).unique()[:200])

    numeric_is_int: Dict[str, bool] = {}
    if sample is not None:
        for c in numeric_cols:
            if c in sample.columns:
                col = sample[c].dropna()
                if pd.api.types.is_integer_dtype(col):
                    numeric_is_int[c] = True
                else:
                    numeric_is_int[c] = col.apply(lambda x: float(x).is_integer()).all() if len(col) > 0 else False

    st.sidebar.subheader("Numeric")
    for c in numeric_cols:
        default = medians.get(c, 0)
        if numeric_is_int.get(c, False):
            val = st.sidebar.number_input(c, value=int(default), step=1, format="%d")
            inputs[c] = int(val)
        else:
            val = st.sidebar.number_input(c, value=float(default))
            inputs[c] = float(val)

    st.sidebar.subheader("Categorical")
    for c in categorical_cols:
        disp = clean_label(c)
        choices = uniques.get(c)
        if choices:
            val = st.sidebar.selectbox(disp, options=["(select)"] + choices)
            if val == "(select)":
                val = ""
        else:
            val = st.sidebar.text_input(disp, value="")
        inputs[c] = val

    return inputs


def render_inputs(container, numeric_cols: List[str], categorical_cols: List[str], sample: Optional[pd.DataFrame], prefix: str = "") -> Dict[str, object]:
    """Render input widgets inside the given Streamlit container and return a dict of values."""
    inputs: Dict[str, object] = {}

    medians: Dict[str, float] = {}
    uniques: Dict[str, List[str]] = {}
    if sample is not None:
        for c in numeric_cols:
            if c in sample.columns:
                try:
                    if pd.api.types.is_numeric_dtype(sample[c]):
                        med = sample[c].median(skipna=True)
                        medians[c] = float(med if pd.notna(med) else 0)
                    else:
                        coerced = pd.to_numeric(sample[c], errors="coerce").dropna()
                        if len(coerced) > 0:
                            med = coerced.median()
                            medians[c] = float(med)
                        else:
                            medians[c] = 0.0
                except Exception:
                    medians[c] = 0.0
        for c in categorical_cols:
            if c in sample.columns:
                uniques[c] = list(sample[c].dropna().astype(str).unique()[:200])

    numeric_is_int: Dict[str, bool] = {}
    if sample is not None:
        for c in numeric_cols:
            if c in sample.columns:
                col = sample[c].dropna()
                if pd.api.types.is_integer_dtype(col):
                    numeric_is_int[c] = True
                else:
                    numeric_is_int[c] = col.apply(lambda x: float(x).is_integer()).all() if len(col) > 0 else False

    container.subheader("Numeric")
    for c in numeric_cols:
        default = medians.get(c, 0)
        key = f"{prefix}_num_{c}"
        if numeric_is_int.get(c, False):
            val = container.number_input(clean_label(c), value=int(default), step=1, format="%d", key=key)
            inputs[c] = int(val)
        else:
            val = container.number_input(clean_label(c), value=float(default), key=key)
            inputs[c] = float(val)

    container.subheader("Categorical")
    for c in categorical_cols:
        disp = clean_label(c)
        key = f"{prefix}_cat_{c}"
        choices = uniques.get(c)
        if choices:
            val = container.selectbox(disp, options=["(select)"] + choices, key=key)
            if val == "(select)":
                val = ""
        else:
            val = container.text_input(disp, value="", key=key)
        inputs[c] = val

    return inputs


def run_project_cost_simulation(model, base_inputs: Dict[str, object], var_feature: str, low: float, high: float, n_samples: int = 100, mode: str = "grid") -> pd.DataFrame:
    """Simulate Project Cost predictions by varying a single numeric feature.

    Returns a DataFrame with columns: var_feature, prediction
    """
    if mode == "grid":
        vals = np.linspace(low, high, num=n_samples)
    else:
        vals = np.random.uniform(low, high, size=n_samples)

    rows = []
    for v in vals:
        r = base_inputs.copy()
        r[var_feature] = v
        rows.append(r)

    df = pd.DataFrame(rows)
    # Try to coerce numeric columns where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    try:
        preds = model.predict(df)
    except Exception as e:
        raise RuntimeError(f"Simulation prediction failed: {e}")

    out = pd.DataFrame({var_feature: vals, "prediction": preds})
    return out


def run_project_cost_simulation_on_sample(model, sample_df: pd.DataFrame, var_feature: str, low: float, high: float, mode: str = "grid") -> pd.DataFrame:
    """Apply a transformation to `var_feature` across the sample DataFrame and return the sample with predictions.

    For 'grid' mode, var_feature values are replaced with a linear sweep from low->high across rows.
    For 'random' mode, var_feature values are replaced with random draws between low and high.
    The returned DataFrame includes a new column 'prediction'.
    """
    if sample_df is None or var_feature not in sample_df.columns:
        raise ValueError("Sample DataFrame not provided or does not contain the feature")

    df = sample_df.copy().reset_index(drop=True)
    n = len(df)
    if mode == "grid":
        vals = np.linspace(low, high, num=n)
    else:
        vals = np.random.uniform(low, high, size=n)

    df[var_feature] = vals

    # Try to coerce numeric columns where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="ignore")

    try:
        preds = model.predict(df)
    except Exception as e:
        raise RuntimeError(f"Simulation prediction failed on sample: {e}")

    df = df.copy()
    df["prediction"] = preds
    return df





def main():
    # Banner and CSS
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    banner_html = f"""
    <div class='banner'>
      <div>{_PACO_SVG}</div>
      <div>
        <h1><strong>Paco</strong> Reputation Earned</h1>
        <p style='opacity:.9;margin-top:6px'>Interactive model explorer — predict, simulate, and compare</p>
      </div>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)

    st.write("\n")

    # (Removed the legacy 'classic' single-model toggle — use the main selector below)

    # Two-column layout: left = Project Success card, right = other models selector
    left_col, right_col = st.columns([2, 1])

    # --- Project Success card (main) ---
    with left_col:
        # small selector to pick which model is the "main" model
        default_index = 0
        try:
            default_index = list(MODEL_FILES.keys()).index("Project Success Probability")
        except Exception:
            default_index = 0
        main_model_picker = st.selectbox("Main model", options=list(MODEL_FILES.keys()), index=default_index, key='main_model_picker')

        # Friendly main card title (do not show raw model filename here)
        st.header("360° Project Success")
        ps_name = main_model_picker
        ps_path = MODEL_FILES.get(ps_name)
        ps_model = load_model(ps_path) if ps_path else None
        if ps_model is None:
            st.warning(f"Model not found: {ps_path}. Train the model first.")
        else:
            ps_manifest = load_manifest_for_model(ps_path) or {}
            if ps_manifest:
                ps_numeric = ps_manifest.get('numeric', []) or []
                ps_categorical = ps_manifest.get('categorical', []) or []
            else:
                ps_numeric, ps_categorical = infer_columns_from_pipeline(ps_model)
            ps_sample = load_sample_for_model(ps_name)

            with st.container():
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                ps_inputs = render_inputs(st, ps_numeric, ps_categorical, ps_sample, prefix='ps')

                # Note: features list removed by request — keeping inputs only

                # Usage notes editor (allow writing guidance and saving locally)
                notes_path = None
                if ps_path:
                    base, _ = os.path.splitext(ps_path)
                    notes_path = base + '_notes.md'

                existing_notes = ""
                if notes_path and os.path.exists(notes_path):
                    try:
                        with open(notes_path, 'r', encoding='utf-8') as fh:
                            existing_notes = fh.read()
                    except Exception:
                        existing_notes = ""

                notes = st.text_area("Usage notes — write how to use this model and decision guidance", value=existing_notes, height=180, key='ps_usage_notes')
                if st.button("Save notes", key='save_ps_notes'):
                    if notes_path is None:
                        st.error("Cannot save notes: model path not available.")
                    else:
                        try:
                            with open(notes_path, 'w', encoding='utf-8') as fh:
                                fh.write(notes)
                            st.success(f"Saved notes to {notes_path}")
                        except Exception as e:
                            st.error(f"Failed to save notes: {e}")

                if st.button("Predict Project Success", key='predict_ps'):
                    try:
                        df_ps = pd.DataFrame([ps_inputs])
                        for c in ps_numeric:
                            if c in df_ps.columns:
                                df_ps[c] = pd.to_numeric(df_ps[c], errors='coerce').fillna(0)
                        if hasattr(ps_model, 'predict_proba'):
                            proba = ps_model.predict_proba(df_ps)
                            if proba.shape[1] == 2:
                                st.success(f"Predicted success probability: {float(proba[0,1]):.3f}")
                            else:
                                st.write('Class probabilities:', proba.tolist())
                        else:
                            pred = ps_model.predict(df_ps)
                            st.write('Predicted:', pred[0])
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")

                # WHAT-IF / AI Simulator — generic for the selected main model
                with st.expander("What-if Simulator (AI)"):
                    st.write("Quickly vary a numeric input and observe how the model prediction responds.")
                    if len(ps_numeric) == 0:
                        st.info("No numeric inputs available for simulation.")
                    else:
                        sim_feature = st.selectbox("Feature to vary", options=ps_numeric, key='ps_sim_feature')
                        sample_low = float(ps_sample[sim_feature].min()) if (ps_sample is not None and sim_feature in ps_sample.columns) else 0.0
                        sample_high = float(ps_sample[sim_feature].max()) if (ps_sample is not None and sim_feature in ps_sample.columns) else sample_low + 1.0
                        low = st.number_input("Low", value=sample_low, key='ps_sim_low')
                        high = st.number_input("High", value=sample_high, key='ps_sim_high')
                        n = st.slider("Points", min_value=10, max_value=1000, value=100, key='ps_sim_n')
                        mode = st.selectbox("Mode", ["grid", "random"], key='ps_sim_mode')
                        apply_sample = st.checkbox("Apply change across sample (show modified sample predictions)", value=False, key='ps_sim_apply')

                        if st.button("Run what-if", key='ps_run_sim'):
                            try:
                                base_inputs = ps_inputs.copy()
                                if apply_sample and ps_sample is not None and sim_feature in ps_sample.columns:
                                    out_df = run_project_cost_simulation_on_sample(ps_model, ps_sample, sim_feature, low, high, mode)
                                    st.write(out_df.head())
                                    fig, ax = plt.subplots()
                                    ax.hist(out_df['prediction'].dropna(), bins=30)
                                    ax.set_xlabel('prediction')
                                    ax.set_ylabel('count')
                                    st.pyplot(fig)
                                else:
                                    sim_df = run_project_cost_simulation(ps_model, base_inputs, sim_feature, low, high, n, mode)
                                    st.write(sim_df.head())
                                    if mode == 'grid':
                                        st.line_chart(sim_df.set_index(sim_feature)['prediction'])
                                    else:
                                        fig, ax = plt.subplots()
                                        ax.hist(sim_df['prediction'].dropna(), bins=30)
                                        ax.set_xlabel('prediction')
                                        ax.set_ylabel('count')
                                        st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Simulation failed: {e}")
                st.markdown("</div>", unsafe_allow_html=True)

    # --- Other models box (simplified) ---
    with right_col:
        st.header("Other Models")
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("Other available models:")
        for name in [k for k in MODEL_FILES.keys() if k != "Project Success Probability"]:
            st.write(f"- {name}")
        st.markdown("</div>", unsafe_allow_html=True)




if __name__ == "__main__":
    main()
