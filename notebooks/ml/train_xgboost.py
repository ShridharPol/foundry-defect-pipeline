"""
XGBoost Process Anomaly Detector — tabular features from mart_defect_features.
Tracks experiment with MLflow. Includes SHAP explainability.
"""

import os
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from google.cloud import bigquery
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# --- Config ---
PROJECT_ID  = "foundry-defect-pipeline-2"
DATASET     = "foundry_raw"
TABLE       = "ai4i_maintenance"
MODEL_DIR   = "mlflow/models"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Load from BigQuery ---
def load_data():
    client = bigquery.Client()
    query  = f"""
        SELECT
            air_temperature_k,
            process_temperature_k,
            rotational_speed_rpm,
            torque_nm,
            tool_wear_min,
            twf, hdf, pwf, osf, rnf,
            machine_failure
        FROM `{PROJECT_ID}.{DATASET}.{TABLE}`
        WHERE machine_failure IS NOT NULL
    """
    print("Loading ai4i_maintenance from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"Loaded {len(df)} rows | Failures: {df['machine_failure'].sum()} | OK: {(df['machine_failure']==0).sum()}")
    return df

# --- Train ---
def train():
    df = load_data()

    # Note: failure type flags (twf, hdf, pwf, osf, rnf) excluded —
    # they are sub-components of the target (machine_failure) and
    # would constitute data leakage in a real production setting.
    feature_cols = [
        "air_temperature_k", "process_temperature_k",
        "rotational_speed_rpm", "torque_nm", "tool_wear_min",
    ]
    X = df[feature_cols]
    y = df["machine_failure"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators":     200,
        "max_depth":        4,
        "learning_rate":    0.05,
        "subsample":        0.8,
        "colsample_bytree": 0.8,
        "use_label_encoder": False,
        "eval_metric":      "logloss",
        "random_state":     42,
    }

    mlflow.set_experiment("foundry_defect_detection")

    with mlflow.start_run(run_name="xgboost_process_anomaly_v2_no_leakage"):
        mlflow.log_params({
            "model":            "XGBoost",
            "source_table":     TABLE,
            "features":         str(feature_cols),
            "n_estimators":     params["n_estimators"],
            "max_depth":        params["max_depth"],
            "learning_rate":    params["learning_rate"],
            "train_size":       len(X_train),
            "test_size":        len(X_test),
        })

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        # --- Metrics ---
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]
        acc         = accuracy_score(y_test, y_pred)
        auc         = roc_auc_score(y_test, y_pred_prob)

        mlflow.log_metrics({
            "test_accuracy": round(acc, 4),
            "test_auc":      round(auc, 4),
        })

        print(f"\nTest Accuracy : {acc:.4f}")
        print(f"Test AUC      : {auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=["ok", "failure"]))

        # --- Confusion matrix ---
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d',
                    xticklabels=["ok", "failure"],
                    yticklabels=["ok", "failure"], ax=ax)
        ax.set_title("Confusion Matrix — XGBoost")
        plt.tight_layout()
        cm_path = os.path.join(MODEL_DIR, "confusion_matrix_xgboost.png")
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()

        # --- SHAP explainability ---
        print("\nGenerating SHAP values...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # SHAP summary plot
        fig_shap, ax_shap = plt.subplots(figsize=(8, 4))
        shap.summary_plot(shap_values, X_test, show=False)
        shap_path = os.path.join(MODEL_DIR, "shap_summary.png")
        plt.tight_layout()
        plt.savefig(shap_path, bbox_inches="tight")
        mlflow.log_artifact(shap_path)
        plt.close()

        # Feature importance from SHAP
        mean_shap = pd.DataFrame({
            "feature":    feature_cols,
            "importance": np.abs(shap_values).mean(axis=0)
        }).sort_values("importance", ascending=False)
        print("\nSHAP Feature Importances:")
        print(mean_shap.to_string(index=False))

        # --- Log model ---
        mlflow.xgboost.log_model(model, "xgboost_model")
        model.save_model(os.path.join(MODEL_DIR, "best_xgboost.json"))

        print(f"\nModel saved to {MODEL_DIR}/best_xgboost.json")
        print(f"SHAP plot    → {shap_path}")
        print(f"Confusion matrix → {cm_path}")

if __name__ == "__main__":
    train()