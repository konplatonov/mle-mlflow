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

from sklearn.linear_model import LinearRegression
from autofeat import AutoFeatClassifier
from sklearn.impute import SimpleImputer

TABLE_NAME = 'users_churn'
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000
EXPERIMENT_NAME = 'features_experiment_afc'
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

X = df.drop([TARGET_COL, "customer_id"], axis=1, errors="ignore")

for col in X.select_dtypes(include=['object']):
    try:
        X[col] = pd.to_datetime(X[col])
    except Exception:
        pass

# И удалим все что теперь datetime
datetime_cols_detected = X.select_dtypes(include=['datetime', 'datetime64[ns]', 'timedelta']).columns
X = X.drop(list(datetime_cols_detected), axis=1)

y = df[TARGET_COL]

# --- Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cat_features = [
    'paperless_billing',
    'payment_method',
    'internet_service',
    'online_security',
    'online_backup',
    'device_protection',
    'tech_support',
    'streaming_tv',
    'streaming_movies',
    'gender',
    'senior_citizen',
    'partner',
    'dependents',
    'multiple_lines',
]
num_features = ["monthly_charges", "total_charges"]

features = cat_features + num_features

transformations = ('1/', 'log', 'abs', 'sqrt')

# Для числовых признаков
num_imputer = SimpleImputer(strategy='median')
X_train[num_features] = num_imputer.fit_transform(X_train[num_features])
X_test[num_features] = num_imputer.transform(X_test[num_features])

# Для категориальных признаков
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_features] = cat_imputer.fit_transform(X_train[cat_features])
X_test[cat_features] = cat_imputer.transform(X_test[cat_features])

afc = AutoFeatClassifier(categorical_cols=cat_features,
                         transformations=transformations,
                         feateng_steps=1,
                         n_jobs=-1)

X_train = X_train[cat_features + num_features]
X_test = X_test[cat_features + num_features]

X_train_features = afc.fit_transform(X_train, y_train)
X_test_features = afc.transform(X_test)

rf = RandomForestClassifier()
rf.fit(X_train_features, y_train)

# Предсказания
preds = rf.predict(X_test_features)

metrics = {
    "AUC": roc_auc_score(y_test, preds),
    "F1": f1_score(y_test, preds)
}

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
        sk_model=rf,
        artifact_path="model_pipeline",
        signature=signature,
        input_example=X_train.head(2),
        registered_model_name=REGISTRY_MODEL_NAME
    )

    afc_info = mlflow.sklearn.log_model(afc, artifact_path='afc') 

# Очистка временных файлов
for filename in ['columns.txt', 'users_churn.csv']:
    if os.path.exists(filename):
        os.remove(filename)