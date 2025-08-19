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

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

TABLE_NAME = 'users_churn'
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = 'feature_selection'
RUN_NAME = "feature_selection"
REGISTRY_MODEL_NAME = 'model with sfs sbs'
FS_ASSETS = "fs_assets"

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
        cur.execute(f"SELECT * FROM {TABLE_NAME} limit 2000")
        data = cur.fetchall()
        columns = [col[0] for col in cur.description]
df = pd.DataFrame(data, columns=columns)

print(df.head())
print(len(df))

# --- Укажите ваши колонки здесь
TARGET_COL = "target"
cat_columns = ["type", "payment_method", "internet_service", "gender"]
num_columns = ["monthly_charges", "total_charges"]

X = df[cat_columns + num_columns]

y = df[TARGET_COL]

# --- Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Импутировать числа
num_imputer = SimpleImputer(strategy='median')
X_train[num_columns] = num_imputer.fit_transform(X_train[num_columns])
X_test[num_columns] = num_imputer.transform(X_test[num_columns])
print('num imputer done')

# Импутировать категории
cat_imputer = SimpleImputer(strategy='most_frequent')
X_train[cat_columns] = cat_imputer.fit_transform(X_train[cat_columns])
X_test[cat_columns] = cat_imputer.transform(X_test[cat_columns])
print('cat imputer done')

# Преобразуем категории в дамми
X_train = pd.get_dummies(X_train, columns=cat_columns, drop_first=True)
X_test  = pd.get_dummies(X_test, columns=cat_columns, drop_first=True)
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)
print('dummies done')

assert X_train.isnull().sum().sum() == 0
assert X_test.isnull().sum().sum() == 0

from sklearn.ensemble import RandomForestClassifier
from mlxtend.feature_selection import SequentialFeatureSelector

# 1. Оценщик: RandomForestClassifier (n_estimators=300)
estimator = RandomForestClassifier(n_estimators=100)

# 2. Sequential Forward Selection (SFS)
sfs = SFS(estimator, k_features=6, forward=True, floating=False, scoring='roc_auc', cv=4, n_jobs=-1)
print('sfs done')

# 3. Sequential Backward Selection (SBS)
sbs = SFS(estimator, k_features=6, forward=False, floating=False, scoring='roc_auc', cv=4, n_jobs=-1)
print('sbs done')

# 4. Обучение
sfs = sfs.fit(X_train, y_train)
sbs = sbs.fit(X_train, y_train)
print('sfs/sbs fitted')

# 5. Имена признаков
top_sfs = sfs.k_feature_names_
top_sbs = sbs.k_feature_names_

sfs_df = pd.DataFrame.from_dict(sfs.get_metric_dict()).T
sbs_df = pd.DataFrame.from_dict(sbs.get_metric_dict()).T

os.makedirs(FS_ASSETS, exist_ok=True)

sfs_df.to_csv(f"{FS_ASSETS}/sfs.csv")
sbs_df.to_csv(f"{FS_ASSETS}/sbs.csv")

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

fig = plot_sfs(sfs.get_metric_dict(), kind='std_dev')

plt.title('Sequential Forward Selection (w. StdDev)')
plt.grid()
plt.show()

plt.savefig("FS_ASSETS/sfs.png")

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

fig = plot_sfs(sbs.get_metric_dict(), kind='std_dev')

plt.title('Sequential Backward Selection (w. StdDev)')
plt.grid()
plt.show()

plt.savefig("FS_ASSETS/sbs.png") 

interc_features = list(set(top_sbs) & set(top_sfs))
union_features = list(set(top_sbs) | set(top_sfs))

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

with mlflow.start_run(run_name=f"{RUN_NAME}_intersection_and_union", experiment_id=experiment_id) as run:
    run_id = run.info.run_id
   
    mlflow.log_artifacts(FS_ASSETS)