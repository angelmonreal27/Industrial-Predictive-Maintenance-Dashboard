import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score


# -------------------------
# LOAD MODEL
# -------------------------
def load_model(model_path):
    return joblib.load(model_path)


# -------------------------
# LOAD DATA
# -------------------------
def load_data(path):
    return pd.read_csv(path)


# -------------------------
# PREPROCESS DATA
# (one-hot encoding + align features)
# -------------------------
def preprocess_data(df, features, drop_cols):
    df = pd.get_dummies(df.drop(columns=drop_cols), drop_first=True)
    df = df.reindex(columns=features, fill_value=0)
    return df


# -------------------------
# APPLY SCALER (optional)
# -------------------------
def apply_scaler(X, scaler):
    return scaler.transform(X)


# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict(model, X, threshold=0.3):
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs, preds


# -------------------------
# METRICS CALCULATION
# -------------------------
def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0)
    }


# -------------------------
# HIGH RISK FILTER
# -------------------------
def get_high_risk_data(df, probs, threshold):
    result = df.copy()
    result["Risk Probability"] = probs
    result["Prediction"] = (probs >= threshold).astype(int)
    return result[result["Risk Probability"] > threshold]


# -------------------------
# CONTROL LIMITS (3-sigma)
# -------------------------
def control_limits(probs):
    mean = np.mean(probs)
    std = np.std(probs)

    ucl = mean + 3 * std
    lcl = max(0, mean - 3 * std)

    return mean, std, ucl, lcl