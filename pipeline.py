"""
Fraud Detection & Risk Prediction System
=========================================
Full ML pipeline: split → scale → SMOTE → train → evaluate → SHAP
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb
import shap
import joblib

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = os.environ.get("FRAUD_DATA", "creditcard.csv")
OUTPUT_DIR  = os.environ.get("FRAUD_OUTPUT", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Palette ────────────────────────────────────────────────────────────────────
PALETTE = {"legit": "#4C8BF5", "fraud": "#E53935", "neutral": "#78909C"}

# =============================================================================
# 1.  LOAD & BASIC CHECKS
# =============================================================================
def load_data(path=DATA_PATH):
    print(f"\n{'='*60}")
    print("STEP 1 — Loading data")
    print('='*60)
    df = pd.read_csv(path)
    print(f"Shape          : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    dups = df.duplicated().sum()
    print(f"Duplicates     : {dups}")
    df = df.drop_duplicates()
    print(f"Shape after dedup: {df.shape}")
    fraud_pct = df["Class"].mean() * 100
    print(f"Fraud rate     : {fraud_pct:.4f}%  "
          f"({df['Class'].sum()} fraud / {len(df)} total)")
    return df

# =============================================================================
# 2.  FEATURE ENGINEERING
# =============================================================================
def engineer_features(df):
    """Log-transform Amount (right-skewed); drop raw Amount & Time."""
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])
    # Normalise Time to hours-in-day cycle (optional but clean)
    df["hour_of_day"] = (df["Time"] / 3600) % 24
    df = df.drop(columns=["Time", "Amount"])
    return df

# =============================================================================
# 3.  TRAIN / TEST SPLIT  (stratified, before ANY scaling/SMOTE)
# =============================================================================
def split(df):
    print(f"\n{'='*60}")
    print("STEP 3 — Train/Test split (stratified, 80/20)")
    print('='*60)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")
    print(f"Train fraud: {y_train.sum()}  |  Test fraud: {y_test.sum()}")
    return X_train, X_test, y_train, y_test

# =============================================================================
# 4.  SCALE  (fit on train only → transform both)
# =============================================================================
def scale(X_train, X_test):
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")
    return X_train_s, X_test_s, scaler

# =============================================================================
# 5.  SMOTE  (applied on train only AFTER scaling)
# =============================================================================
def apply_smote(X_train_s, y_train):
    print(f"\n{'='*60}")
    print("STEP 5 — Applying SMOTE (train only)")
    print('='*60)
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train_s, y_train)
    print(f"Before SMOTE — Class 0: {(y_train==0).sum():,}  "
          f"Class 1: {(y_train==1).sum():,}")
    print(f"After  SMOTE — Class 0: {(y_res==0).sum():,}  "
          f"Class 1: {(y_res==1).sum():,}")
    return X_res, y_res

# =============================================================================
# 6.  EVALUATION HELPER
# =============================================================================
def evaluate(name, model, X_test_s, y_test, feature_names, results_list):
    print(f"\n  ── {name} ──")
    y_pred  = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)[:, 1]

    pr_auc  = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1      = f1_score(y_test, y_pred)

    print(f"  PR-AUC  : {pr_auc:.4f}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    results_list.append({
        "Model": name, "PR-AUC": pr_auc,
        "ROC-AUC": roc_auc, "F1": f1,
        "model_obj": model, "y_proba": y_proba
    })

    # Confusion matrix plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=["Legit", "Fraud"],
        colorbar=False, ax=ax, cmap="Blues"
    )
    ax.set_title(f"{name} — Confusion Matrix", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/cm_{name.replace(' ', '_')}.png", dpi=150)
    plt.close()

# =============================================================================
# 7.  MODELS
# =============================================================================
def train_models(X_res, y_res, X_test_s, y_test, feature_names):
    print(f"\n{'='*60}")
    print("STEP 7 — Training & evaluating models")
    print('='*60)
    results = []

    # 7a  Logistic Regression (baseline)
    lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    lr.fit(X_res, y_res)
    evaluate("Logistic Regression", lr, X_test_s, y_test, feature_names, results)

    # 7b  Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_res, y_res)
    evaluate("Random Forest", rf, X_test_s, y_test, feature_names, results)

    # 7c  XGBoost
    scale_pos = (y_res == 0).sum() / (y_res == 1).sum()
    xgb_clf = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos, use_label_encoder=False,
        eval_metric="aucpr", random_state=42, n_jobs=-1,
        verbosity=0
    )
    xgb_clf.fit(X_res, y_res)
    evaluate("XGBoost", xgb_clf, X_test_s, y_test, feature_names, results)

    # 7d  LightGBM
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        class_weight="balanced", random_state=42, n_jobs=-1,
        verbose=-1
    )
    lgb_clf.fit(X_res, y_res)
    evaluate("LightGBM", lgb_clf, X_test_s, y_test, feature_names, results)

    return results

# =============================================================================
# 8.  COMPARISON CHART
# =============================================================================
def plot_comparison(results):
    df_r = pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ("model_obj", "y_proba")}
                          for r in results])
    df_r = df_r.set_index("Model")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    metrics = ["PR-AUC", "ROC-AUC", "F1"]
    colors  = [PALETTE["legit"], PALETTE["fraud"], PALETTE["neutral"]]

    for ax, metric, color in zip(axes, metrics, colors):
        bars = ax.barh(df_r.index, df_r[metric], color=color, alpha=0.85)
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric, fontsize=10)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Model Comparison — Fraud Detection", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nComparison chart saved → {OUTPUT_DIR}/model_comparison.png")
    return df_r

# =============================================================================
# 9.  PRECISION-RECALL CURVES
# =============================================================================
def plot_pr_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for r in results:
        prec, rec, _ = precision_recall_curve(y_test, r["y_proba"])
        ax.plot(rec, prec,
                label=f"{r['Model']} (AP={r['PR-AUC']:.3f})", linewidth=1.8)
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves", fontsize=13)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pr_curves.png", dpi=150)
    plt.close()
    print(f"PR curves saved → {OUTPUT_DIR}/pr_curves.png")

# =============================================================================
# 10. SHAP EXPLAINABILITY
# =============================================================================
def shap_analysis(best_model, best_name, X_test_s, feature_names):
    print(f"\n{'='*60}")
    print(f"STEP 10 — SHAP for {best_name}")
    print('='*60)

    # Use TreeExplainer for tree models, LinearExplainer for LR
    if "Logistic" in best_name:
        explainer = shap.LinearExplainer(best_model, X_test_s)
    else:
        explainer = shap.TreeExplainer(best_model)

    # Use a sample of 500 test rows for speed
    sample_idx = np.random.RandomState(42).choice(len(X_test_s), 500, replace=False)
    X_sample = X_test_s[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    # For binary classifiers that return list (RF), take index 1
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # ── Summary (beeswarm) ──────────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    shap.summary_plot(sv, X_sample, feature_names=feature_names,
                      show=False, plot_size=None)
    plt.title(f"SHAP Summary — {best_name}", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary saved → {OUTPUT_DIR}/shap_summary.png")

    # ── Waterfall for the single highest-risk transaction ───────────────────
    # Pick the sample row with the highest fraud probability
    if "Logistic" in best_name:
        local_sv = explainer.shap_values(X_sample)
    else:
        expl_obj = explainer(X_sample)
        # Take class-1 slice if multi-class
        if len(expl_obj.shape) == 3:
            expl_obj = expl_obj[:, :, 1]

    plt.figure(figsize=(9, 5))
    if "Logistic" in best_name:
        shap.waterfall_plot(
            shap.Explanation(
                values=local_sv[0],
                base_values=explainer.expected_value,
                data=X_sample[0],
                feature_names=feature_names
            ), show=False, max_display=15
        )
    else:
        shap.waterfall_plot(expl_obj[0], show=False, max_display=15)
    plt.title(f"SHAP Waterfall — Top Flagged Transaction ({best_name})", fontsize=11)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP waterfall saved → {OUTPUT_DIR}/shap_waterfall.png")

    return explainer, sv, sample_idx

# =============================================================================
# 11. SAVE BEST MODEL
# =============================================================================
def save_best(results, X_test_s, y_test):
    best = max(results, key=lambda r: r["PR-AUC"])
    print(f"\nBest model by PR-AUC: {best['Model']}  ({best['PR-AUC']:.4f})")
    joblib.dump(best["model_obj"], f"{OUTPUT_DIR}/best_model.pkl")
    print(f"Model saved → {OUTPUT_DIR}/best_model.pkl")
    return best

# =============================================================================
# MAIN
# =============================================================================
def run():
    df           = load_data()
    df           = engineer_features(df)
    feature_names = list(df.drop(columns=["Class"]).columns)

    X_train, X_test, y_train, y_test = split(df)
    X_train_s, X_test_s, scaler      = scale(X_train, X_test)
    X_res, y_res                     = apply_smote(X_train_s, y_train)

    results   = train_models(X_res, y_res, X_test_s, y_test, feature_names)
    df_cmp    = plot_comparison(results)
    plot_pr_curves(results, y_test)
    best      = save_best(results, X_test_s, y_test)

    shap_analysis(
        best["model_obj"], best["Model"],
        X_test_s, feature_names
    )

    print(f"\n{'='*60}")
    print("All done! Outputs in →", OUTPUT_DIR)
    print('='*60)
    return results, best, feature_names, X_test_s, y_test, scaler

if __name__ == "__main__":
    run()
