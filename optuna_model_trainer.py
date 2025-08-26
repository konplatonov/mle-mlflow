import os
import numpy as np
import pandas as pd
import mlflow
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix,
)
from catboost import CatBoostClassifier
import optuna
from optuna.integration.mlflow import MLflowCallback
from dotenv import load_dotenv

# Загрузка переменных окружения
load_dotenv()

# MLflow & Optuna config
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

EXPERIMENT_NAME = "bayesian_search"
RUN_NAME = "model_bayesian_search"
STUDY_DB_NAME = "sqlite:///local.study.db"
STUDY_NAME = "churn_model"

# ==== Пример получения/подготовки данных ====
# Загрузите ваш df, сформируйте features и target!

# ... пропустим загрузку из SQL, оставим пользовательский пример:

# Предположим, у вас уже есть X_train, X_test, y_train, y_test с исходными данными

# === Objective функция для optuna ===
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
    metrics = defaultdict(list)

    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train)):
        # Обучаем модель CatBoost на каждом фолде отдельно
        model = CatBoostClassifier(**param)
        model.fit(X_train.iloc[train_index], y_train.iloc[train_index],
                  eval_set=(X_train.iloc[val_index], y_train.iloc[val_index]),
                  use_best_model=False)

        val_X = X_train.iloc[val_index]
        val_y = y_train.iloc[val_index]
        val_prediction = model.predict(val_X)
        val_probas = model.predict_proba(val_X)[:, 1]

        tn, fp, fn, tp = confusion_matrix(val_y, val_prediction).ravel()
        err1 = fp  # ложноположительные
        err2 = fn  # ложноотрицательные
        auc = roc_auc_score(val_y, val_probas)
        precision = precision_score(val_y, val_prediction)
        recall = recall_score(val_y, val_prediction)
        f1 = f1_score(val_y, val_prediction)
        logloss = log_loss(val_y, val_probas)

        metrics["err1"].append(err1)
        metrics["err2"].append(err2)
        metrics["auc"].append(auc)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        metrics["logloss"].append(logloss)

    # Агрегирование метрик по фолдам
    err1 = np.mean(metrics["err1"])
    err2 = np.mean(metrics["err2"])
    auc = np.mean(metrics["auc"])
    precision = np.mean(metrics["precision"])
    recall = np.mean(metrics["recall"])
    f1 = np.mean(metrics["f1"])
    logloss = np.mean(metrics["logloss"])

    # (Если хотите продвигать и логировать метрики — можно, но objective возвращает только основную метрику!)
    return auc  # Основная метрика для оптимизации

# === Работа с MLflow и Optuna ===
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    # MLflowCallback (логирует результаты Optuna-trials прямо в MLflow)
    mlflc = MLflowCallback(
        tracking_uri=f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}",
        metric_name="auc"
    )

    # Создание и запуск Optuna
    study = optuna.create_study(
        storage=STUDY_DB_NAME,
        study_name=STUDY_NAME,
        direction="maximize"
    )
    study.optimize(objective, n_trials=10, callbacks=[mlflc])

    best_params = study.best_params

    print(f"Number of finished trials: {len(study.trials)}")
    print(f"Best params: {best_params}")

    # Обучим лучшую модель на всём трейне и залогируем в MLflow
    final_model = CatBoostClassifier(**best_params,
                                    loss_function="Logloss",
                                    task_type="CPU",
                                    random_seed=0,
                                    iterations=300,
                                    verbose=False)
    final_model.fit(X_train, y_train)

    # Логирование модели
    from mlflow.models.signature import infer_signature
    signature = infer_signature(X_train, final_model.predict(X_train))
    input_example = X_train.head(10)
    mlflow.catboost.log_model(final_model,
                              artifact_path="model",
                              registered_model_name=REGISTRY_MODEL_NAME,
                              signature=signature,
                              input_example=input_example)