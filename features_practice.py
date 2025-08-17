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
from sklearn.model_selection import train_test_split
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

TABLE_NAME = 'users_churn'
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = 'features_experiment'
RUN_NAME = "preprocessing"
REGISTRY_MODEL_NAME = 'model_with_prepro'

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"),
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}
connection.update(postgres_credentials)
with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]
df = pd.DataFrame(data, columns=columns)

# --- Укажите ваши колонки здесь
TARGET_COL = "target"
cat_columns = ["type", "payment_method", "internet_service", "gender"]
num_columns = ["monthly_charges", "total_charges"]
datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64', 'timedelta']).columns

# --- Разделяем X и y
X = df.drop([TARGET_COL, "customer_id"] + list(datetime_cols), axis=1, errors="ignore")
y = df[TARGET_COL]

# --- Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Препроцессоры
encoder_oh = OneHotEncoder(
    categories='auto', handle_unknown='ignore', max_categories=10, sparse=False, drop='first'
)
n_knots = 3
degree_spline = 4
n_quantiles = 100
degree = 3
n_bins = 5
encode = 'ordinal'
strategy = 'uniform'

num_features = FeatureUnion([
    ('spline', SplineTransformer(n_knots=n_knots, degree=degree_spline)),
    ('quantile', QuantileTransformer(n_quantiles=n_quantiles)),
    ('robust', RobustScaler()),
    ('poly', PolynomialFeatures(degree=degree, include_bias=False)),
    ('kbd', KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', encoder_oh, cat_columns),
        ('num', Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("features", num_features)
        ]), num_columns)
    ]
)

# --- Объединяем в пайплайн с моделью
pipe = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# --- Обучаем
pipe.fit(X_train, y_train)

# --- Предсказания и метрики
y_pred = pipe.predict(X_test)
try:
    proba = pipe.predict_proba(X_test)
    pos_idx = list(pipe.named_steps['classifier'].classes_).index(1)
    y_proba = proba[:, pos_idx]
except Exception:
    y_proba = y_pred

metrics = {
    "roc_auc": roc_auc_score(y_test, y_proba) if hasattr(pipe.named_steps['classifier'], "predict_proba") and len(np.unique(y_test)) == 2 else None,
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1": f1_score(y_test, y_pred, zero_division=0),
    "log_loss": log_loss(y_test, y_proba) if hasattr(pipe.named_steps['classifier'], "predict_proba") and len(np.unique(y_test)) == 2 else None,
}
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
metrics.update({
    "false_positive": fp,
    "false_negative": fn
})

# --- mlflow логгирование модели:
os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL")
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")

mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# --- Формирование сигнатуры для логирования --
input_schema = Schema([ColSpec("double", name) for name in X_train.columns])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    mlflow.log_metrics(metrics)

    # Пример логгирования артефактов
    with open('columns.txt', 'w') as f:
        f.writelines([col + '\n' for col in df.columns])
    df.to_csv("users_churn.csv", index=False)
    mlflow.log_artifact("columns.txt", "dataframe")
    mlflow.log_artifact("users_churn.csv", "dataframe")

    # Логируем полный пайплайн
    mlflow.sklearn.log_model(
        sk_model=pipe,
        artifact_path="model_pipeline",
        signature=signature,
        input_example=X_train.head(2),
        registered_model_name=REGISTRY_MODEL_NAME
    )

# Очистка временных файлов
for filename in ['columns.txt', 'users_churn.csv']:
    if os.path.exists(filename):
        os.remove(filename)