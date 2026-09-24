"""
Fraud Detection — Streamlit Demo
Run: streamlit run app.py
"""
import os, joblib
import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

OUTPUT_DIR = os.environ.get("FRAUD_OUTPUT", "output")

st.set_page_config(
    page_title="Fraud Detection Demo",
    page_icon="🔍",
    layout="wide",
)

# ── Load artefacts ──────────────────────────────────────────────────────────
@st.cache_resource
def load_artefacts():
    model  = joblib.load(f"{OUTPUT_DIR}/best_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    return model, scaler

model, scaler = load_artefacts()

# ── Feature names (same order as training) ─────────────────────────────────
FEATURES = [f"V{i}" for i in range(1, 29)] + ["log_amount", "hour_of_day"]

# ── Sidebar — transaction inputs ────────────────────────────────────────────
st.sidebar.header("🔧 Simulate a Transaction")

inputs = {}
with st.sidebar.expander("PCA Components (V1 – V14)", expanded=False):
    for i in range(1, 15):
        inputs[f"V{i}"] = st.slider(f"V{i}", -10.0, 10.0, 0.0, 0.1)
with st.sidebar.expander("PCA Components (V15 – V28)", expanded=False):
    for i in range(15, 29):
        inputs[f"V{i}"] = st.slider(f"V{i}", -10.0, 10.0, 0.0, 0.1)

amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0,
                                  max_value=30000.0, value=50.0, step=1.0)
hour   = st.sidebar.slider("Hour of Day (0–24)", 0.0, 24.0, 12.0, 0.5)

inputs["log_amount"]  = np.log1p(amount)
inputs["hour_of_day"] = hour

# ── Predict ─────────────────────────────────────────────────────────────────
X_input = pd.DataFrame([inputs])[FEATURES]
X_scaled = scaler.transform(X_input)
proba    = model.predict_proba(X_scaled)[0, 1]
pred     = "🚨 FRAUD" if proba >= 0.5 else "✅ LEGITIMATE"

# ── Main panel ──────────────────────────────────────────────────────────────
st.title("🔍 Credit Card Fraud Detection")
st.caption("Adjust the sliders on the left to simulate any transaction.")

col1, col2, col3 = st.columns(3)
col1.metric("Prediction",       pred)
col2.metric("Fraud Probability", f"{proba*100:.1f}%")
col3.metric("Amount",           f"${amount:,.2f}")

st.divider()

# ── Gauge ────────────────────────────────────────────────────────────────────
fig_g, ax_g = plt.subplots(figsize=(5, 0.6))
ax_g.barh(["Risk"], [1], color="#e0e0e0", height=0.5)
ax_g.barh(["Risk"], [proba], color="#E53935" if proba >= 0.5 else "#4C8BF5",
           height=0.5)
ax_g.axvline(0.5, color="black", linestyle="--", linewidth=1.2, label="Threshold (0.5)")
ax_g.set_xlim(0, 1)
ax_g.set_xlabel("Fraud Probability")
ax_g.legend(loc="upper right", fontsize=8)
ax_g.spines[["top", "right", "left"]].set_visible(False)
ax_g.yaxis.set_visible(False)
ax_g.set_title("Risk Gauge", fontsize=11)
st.pyplot(fig_g, use_container_width=False)

st.divider()

# ── SHAP waterfall ────────────────────────────────────────────────────────────
st.subheader("🧠 Why did the model decide this?")

@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

try:
    explainer   = get_explainer(model)
    shap_vals   = explainer.shap_values(X_scaled)
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]
        base = explainer.expected_value[1]
    else:
        sv = shap_vals[0]
        base = explainer.expected_value

    expl = shap.Explanation(
        values       = sv,
        base_values  = base,
        data         = X_scaled[0],
        feature_names= FEATURES,
    )
    fig_w, ax_w = plt.subplots(figsize=(9, 5))
    shap.waterfall_plot(expl, max_display=15, show=False)
    st.pyplot(fig_w, use_container_width=True)
    st.caption(
        "**Red bars** push the score toward FRAUD. "
        "**Blue bars** push it toward LEGITIMATE. "
        "The longer the bar, the bigger the feature's influence."
    )
except Exception as e:
    st.warning(f"SHAP explanation unavailable for this model type: {e}")

st.divider()
st.caption(
    "Model trained on the ULB Credit Card Fraud Detection dataset (284,807 transactions). "
    "V1–V28 are PCA-anonymized features. "
    "Demo only — not for production use."
)
