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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
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

EXPERIMENT_NAME = 'hyper_random_search'
RUN_NAME = "model_random_search"
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

random_seed = 42
verbose = False
loss_function = 'Logloss'
task_type = 'CPU'
iterations = 1000

model = CatBoostClassifier(
random_seed=random_seed,
verbose=verbose,
loss_function=loss_function,
task_type=task_type,
iterations=iterations
)

param_distributions = {
    'learning_rate': [0.01, 0.03, 0.1],
    'depth': [4, 6]
}

cv = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,
    cv=2,
    random_state=42,
    n_jobs=-1,
    scoring='roc_auc'
)
clf = cv.fit(X_train, y_train)

best_params = clf.best_params_
model_best = CatBoostClassifier(**best_params, random_seed=random_seed, verbose=verbose,
                               loss_function=loss_function, task_type=task_type,
                               iterations=iterations)

model_best.fit(X_train, y_train)

prediction = model_best.predict(X_test)
probas = model_best.predict_proba(X_test)[:, 1]

os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

cv_results = pd.DataFrame(clf.cv_results_)

# расчёт метрик качества
metrics = {}

tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
_, err1, _, err2 = confusion_matrix(y_test, prediction, normalize='all').ravel()
auc = roc_auc_score(y_test, probas)
precision = precision_score(y_test, prediction)
recall = recall_score(y_test, prediction)
f1 = f1_score(y_test, prediction)
logloss = log_loss(y_test, prediction)

metrics["err1"] = err1
metrics["err2"] = err2
metrics["auc"] = auc
metrics["precision"] = precision
metrics["recall"] = recall
metrics["f1"] = f1
metrics["logloss"] = logloss

metrics['mean_fit_time'] = cv_results['mean_fit_time'].mean()
metrics['std_fit_time'] = cv_results['std_fit_time'].mean()
metrics['std_test_score'] = cv_results['std_test_score'].mean()
metrics['mean_test_score'] = cv_results['mean_test_score'].mean()
metrics["best_score"] = clf.best_score_

# дополнительные метрики из результатов кросс-валидации
metrics['mean_fit_time'] = cv_results['mean_fit_time'].mean()
metrics['std_fit_time'] = cv_results['std_fit_time'].mean()
metrics['std_test_score'] = cv_results['std_test_score'].mean()
metrics['mean_test_score'] = cv_results['mean_test_score'].mean()
metrics["best_score"] = clf.best_score_  # лучший результат кросс-валидации

# настройки для логирования в MLFlow
pip_requirements = 'requirements.txt'
signature = mlflow.models.infer_signature(X_test, prediction)
input_example = X_test[:10]

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_params(best_params)
    cv_info = mlflow.sklearn.log_model(cv, artifact_path='cv')
    model_info = mlflow.catboost.log_model(model_best, artifact_path='cv',
    signature=signature,
    input_example=input_example,
    registered_model_name=REGISTRY_MODEL_NAME,
    pip_requirements=pip_requirements)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(pip_requirements)