import os
import numpy as np
import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (
    OneHotEncoder, 
    SplineTransformer, 
    QuantileTransformer, 
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
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

TABLE_NAME = 'users_churn'
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = 'hyper_grid_search'
RUN_NAME = "model_grid_search"
REGISTRY_MODEL_NAME = 'model with grid search'
H_ASSETS = "h_assets"

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

split_column = "monthly_charges"
stratify_column = "senior_citizen"
test_size = 0.2

df = df.sort_values(by=[split_column])

X = df[features]

y = df[target]

X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=test_size, shuffle=False)

print(f"Размер выборки для обучения: {X_train.shape}")
print(f"Размер выборки для теста: {X_test.shape}")

loss_function = "Logloss"
task_type = 'CPU'
random_seed = 42
iterations = 300
verbose = False

params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

model = CatBoostClassifier(random_state=random_seed, task_type=task_type)

cv = GridSearchCV(
    estimator=model,
    param_grid=params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=verbose
)

clf = cv.fit(X_train, y_train)

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("S3_ACCESS_KEY")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("S3_SECRET_KEY")

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

cv_results = clf.cv_results_

best_params = cv_results['params'][clf.best_index_]
model_best = RandomForestClassifier(**best_params, random_state=random_seed)

model_best.fit(X_train, y_train)

prediction = model_best.predict(X_test)
probas = model_best.predict_proba(X_test)[:, 1]

# расчёт метрик качества
metrics = {}

tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
err1, err2 = fp/(fp+tn), fn/(fn+tp)  # ошибки первого и второго рода
auc = roc_auc_score(y_test, probas)  # площадь под ROC-кривой
precision = precision_score(y_test, prediction)  # точность
recall = recall_score(y_test, prediction)  # полнота
f1 = f1_score(y_test, prediction)  # F1-мера
logloss = log_loss(y_test, probas)  # LogLoss

# сохранение метрик в словарь
metrics["err1"] = err1
metrics["err2"] = err2
metrics["auc"] = auc
metrics["precision"] = precision
metrics["recall"] = recall
metrics["f1"] = f1
metrics["logloss"] = logloss

# дополнительные метрики из результатов кросс-валидации
metrics["mean_fit_time"] = cv_results['mean_fit_time'][clf.best_index_]  # среднее время обучения
metrics["std_fit_time"] = cv_results['std_fit_time'][clf.best_index_]  # стандартное отклонение времени обучения
metrics["mean_test_score"] = cv_results['mean_test_score'][clf.best_index_]  # средний результат на тесте
metrics["std_test_score"] = cv_results['std_test_score'][clf.best_index_]  # стандартное отклонение результата на тесте
metrics["best_score"] = clf.best_score_  # лучший результат кросс-валидации

# настройки для логирования в MLFlow
pip_requirements = "requirements.txt"
signature = ModelSignature(
    inputs=Schema([ColSpec("double", col) for col in features]),
    outputs=Schema([ColSpec("integer", "target")])
)
input_example = X_train.head(1)

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    mlflow.log_params({
        "loss_function": loss_function,
        "task_type": task_type,
        "random_seed": random_seed,
        "iterations": iterations
    })
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(pip_requirements)
    mlflow.log_model(model_best, "model", signature=signature, input_example=input_example)