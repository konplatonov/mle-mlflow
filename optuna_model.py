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

from statistics import median

from mlflow.models.signature import infer_signature

# ENV VARIABLES

TABLE_NAME = 'users_churn'
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

REGISTRY_MODEL_NAME = 'model with bayesian search'

os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

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

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

EXPERIMENT_NAME = "bayesian_search"
RUN_NAME = "model_bayesian_search"

STUDY_DB_NAME = "sqlite:///local.study.db"
STUDY_NAME = "churn_model"

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
    model = CatBoostClassifier(**param)

    skf = StratifiedKFold(n_splits=2)

    metrics = defaultdict(list)
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        model = CatBoostClassifier(**param)

        train_x = X_train.iloc[train_index]
        train_y = y_train.iloc[train_index]
        val_x = X_train.iloc[val_index]
        val_y = y_train.iloc[val_index]

        model.fit(train_x, train_y,
                  eval_set=(val_x,val_y),
                  use_best_model=False)

        prediction = model.predict(val_x)
        probas = model.predict_proba(val_x)[:, 1]

        _, err_1, _, err_2 = confusion_matrix(val_y, prediction, normalize='all').ravel()
        auc = roc_auc_score(val_y, probas)
        precision = precision_score(val_y, prediction)
        recall = recall_score(val_y, prediction)
        f1 = f1_score(val_y, prediction)
        logloss = log_loss(val_y, prediction)
        
        metrics["err1"].append(err_1)
        metrics["err2"].append(err_2)
        metrics["auc"].append(auc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        metrics["logloss"].append(logloss)

    err_1 = median(np.array(metrics['err1']))
    err_2 = median(np.array(metrics['err2']))
    auc = median(np.array(metrics['auc']))
    precision = median(np.array(metrics['precision']))
    recall = median(np.array(metrics['recall']))
    f1 = median(np.array(metrics['f1']))
    logloss = median(np.array(metrics['logloss']))

    return auc

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if not experiment:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id
    
if mlflow.active_run() is not None:
    mlflow.end_run()

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    final_model = CatBoostClassifier(loss_function="Logloss", task_type="CPU",
                                     random_seed=0, iterations=10, verbose=False)
    final_model.fit(X_train, y_train)
    signature = infer_signature(X_train, final_model.predict(X_train))
    input_example = X_train.head(10)
    mlflow.catboost.log_model(
        final_model,
        artifact_path="model",
        registered_model_name=REGISTRY_MODEL_NAME,
        signature=signature,
        input_example=input_example
    )

mlflc = MLflowCallback(
        tracking_uri=f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}",
        metric_name='AUC',create_experiment=False,
        mlflow_kwargs={'experiment_id': experiment_id, 'tags': {'mlflow.parentRunId': run_id}}
)

study = optuna.create_study(
        storage=STUDY_DB_NAME,
        study_name=STUDY_NAME,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(),
        load_if_exists=True
)

study.optimize(objective, n_trials=10, callbacks=[mlflc])

best_params = study.best_params

print(f"Number of finished trials: {len(study.trials)}")
print(f"Best params: {best_params}")