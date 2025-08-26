import os
import numpy as np
import pandas as pd
import mlflow
from collections import defaultdict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.preprocessing import (
    OneHotEncoder, 
    SplineTransformer, 
    QuantileTransformer, 
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix,
)
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import Schema, ColSpec

import psycopg
from dotenv import load_dotenv
load_dotenv()

from sklearn.linear_model import LinearRegression
from autofeat import AutoFeatClassifier
from sklearn.impute import SimpleImputer

from catboost import CatBoostClassifier


from mlxtend.feature_selection import SequentialFeatureSelector as SFS

import optuna
from optuna.samplers import CmaEsSampler
from optuna.integration.mlflow import MLflowCallback

# ENV VARIABLES

TABLE_NAME = 'users_churn'
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "bayesian_search"
RUN_NAME = "model_bayesian_search"
REGISTRY_MODEL_NAME = 'model with bayesian search'
H_ASSETS = "h_assets"

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

STUDY_DB_NAME = "sqlite:///local.study.db"
STUDY_NAME = "churn_model"

# GET DATA

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}

features = ["monthly_charges", "total_charges", "senior_citizen"]
target = "target"

connection.update(postgres_credentials)
with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME} limit 2000")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]
df = pd.DataFrame(data, columns=columns)

# SAMPLE

split_column = "monthly_charges"
stratify_column = "senior_citizen"
test_size = 0.2

df = df.sort_values(by=[split_column])

X = df[features]

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    df[features], df[target], test_size=test_size, shuffle=True, stratify=df[target], random_state=42
)

# OPTUNA OPTIMIZATION

np.random.seed(42)

# Настройка Optuna логирования:
optuna.logging.set_verbosity(optuna.logging.INFO)

storage = "sqlite:///example.db"
study_name = "bagging-optimization-study"

# MLflow Callback
mlflc = MLflowCallback(
    tracking_uri=f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}",
    metric_name="auc"
)

def objective(trial: optuna.Trial) -> float:
    param = {
    "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
    "depth": trial.suggest_int("depth", 1, 12),
    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 5),
    "random_strength": trial.suggest_float("random_strength", 0.1, 5),
    "loss_function": "Logloss",
    "task_type": "CPU",
    "random_seed": 0,
    "iterations": 300,
    "verbose": False,
}

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    auc_scores = []
    for train_index, val_index in skf.split(X_train, y_train):
        fold_model = CatBoostClassifier(**param)
        fold_model.fit(X_train.iloc[train_index], y_train.iloc[train_index])
        val_X = X_train.iloc[val_index]
        val_y = y_train.iloc[val_index]
        val_probas = fold_model.predict_proba(val_X)[:, 1]
        auc = roc_auc_score(val_y, val_probas)
        auc_scores.append(auc)

    # Optuna будет оптимизировать средний валидационный AUC cross-val
    mean_auc = np.mean(auc_scores)
    return mean_auc

# Создание/загрузка study
study = optuna.create_study(
    storage=STUDY_DB_NAME,
    study_name=STUDY_NAME,
    direction="maximize"
)
study.optimize(objective, n_trials=100, callbacks=[mlflc])

print(f"Number of finished trials: {len(study.trials)}")
print(f"Best params: {study.best_params}")