# train.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, log_loss
)
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def train_and_evaluate(df, target_col="target", random_state=42):
    datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64', 'timedelta']).columns
    X = df.drop([target_col, "customer_id"] + list(datetime_cols), axis=1, errors="ignore")
    y = df[target_col]
    X = pd.get_dummies(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        y_proba = y_pred

    metrics = {
        "roc_auc": roc_auc_score(y_test, y_proba) if hasattr(model, "predict_proba") else None,
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "log_loss": log_loss(y_test, y_proba) if hasattr(model, "predict_proba") else None,
    }
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics.update({
        "false_positive": fp,
        "false_negative": fn
    })
    return model, X, metrics