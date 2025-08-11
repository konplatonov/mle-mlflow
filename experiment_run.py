import os
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, log_loss
)
import psycopg
import pandas as pd
import numpy as np

aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
bucket = os.getenv("S3_BUCKET_NAME")

assert aws_access_key_id and aws_secret_access_key and bucket

os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key

artifact_uri = f"s3://{bucket}"
if root_path:
    artifact_uri = f"{artifact_uri.rstrip('/')}/{root_path.strip('/')}/"

os.environ["MLFLOW_ARTIFACT_URI"] = artifact_uri

connection = {"sslmode": "require", "target_session_attrs": "read-write"}
postgres_credentials = {
    "host": os.getenv("DB_DESTINATION_HOST"), 
    "port": os.getenv("DB_DESTINATION_PORT"),
    "dbname": os.getenv("DB_DESTINATION_NAME"),
    "user": os.getenv("DB_DESTINATION_USER"),
    "password": os.getenv("DB_DESTINATION_PASSWORD"),
}
assert all([var_value != "" for var_value in postgres_credentials.values()])

mlflow.set_tracking_uri(
    f"postgresql://{postgres_credentials['user']}:{postgres_credentials['password']}@"
    f"{postgres_credentials['host']}:{postgres_credentials['port']}/{postgres_credentials['dbname']}"
)

connection.update(postgres_credentials)

TABLE_NAME = "clean_users_churn"

with psycopg.connect(**connection) as conn:
    with conn.cursor() as cur:
        cur.execute(f"SELECT * FROM {TABLE_NAME}")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]

df = pd.DataFrame(data, columns=columns)

# Для примера разобьем на признаки и целевую переменную
TARGET_COL = "target"
datetime_cols = df.select_dtypes(include=['datetime64[ns]', 'datetime64', 'timedelta']).columns
X = df.drop([TARGET_COL, "customer_id"] + list(datetime_cols), axis=1, errors="ignore")
y = df[TARGET_COL]
# Быстрая обработка категориальных — one-hot encoding
X = pd.get_dummies(X)

# Делим на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Обучаем модель
model = RandomForestClassifier(n_estimators=100, max_depth=6, max_features=3)
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)
try:
    y_proba = model.predict_proba(X_test)[:, 1]
except AttributeError:
    y_proba = y_pred # Логлосс и ROC AUC не посчитаются для моделей без predict_proba

# Метрики модели
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

# + старые ваши stats
counts_columns = [
    "type", "paperless_billing", "internet_service", "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies", "gender", "senior_citizen", "partner", "dependents",
    "multiple_lines", "target"
]
stats = {}
for col in counts_columns:
    column_stat = df[col].value_counts()
    column_stat = {f"{col}_{key}": value for key, value in column_stat.items()}
    stats.update(column_stat)

stats["data_length"] = df.shape[0]
stats["monthly_charges_min"] = df["monthly_charges"].min()
stats["monthly_charges_max"] = df["monthly_charges"].max()
stats["monthly_charges_mean"] = df["monthly_charges"].mean()
stats["monthly_charges_median"] = df["monthly_charges"].median()
stats["total_charges_min"] = df["total_charges"].min()
stats["total_charges_max"] = df["total_charges"].max()
stats["total_charges_mean"] = df["total_charges"].mean()
stats["total_charges_median"] = df["total_charges"].median()
stats["unique_customers_number"] = df["customer_id"].nunique()
stats["end_date_nan"] = df["end_date"].isna().sum()

# объединим оба словаря
for d in [stats, metrics]:
    for k, v in d.items():
        if isinstance(v, np.generic):
            d[k] = v.item()
        elif pd.isna(v):
            d[k] = None

EXPERIMENT_NAME = "my_own_experiment"
RUN_NAME = "model_train"

experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME, artifact_location=artifact_uri)
else:
    experiment_id = experiment.experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id
    mlflow.log_metrics(metrics)

    with open('columns.txt', 'w') as f:
        f.writelines([col + '\n' for col in df.columns])
    df.to_csv("users_churn.csv", index=False)

    mlflow.log_artifact("columns.txt", "dataframe")
    mlflow.log_artifact("users_churn.csv", "dataframe")

    # Логируем модель леса
    mlflow.sklearn.log_model(model, "model")

run = mlflow.get_run(run_id)
assert run.info.status == "FINISHED"

for filename in ['columns.txt', 'users_churn.csv']:
    if os.path.exists(filename):
        os.remove(filename)