import os

import pandas as pd
import mlflow
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder, 
    SplineTransformer, 
    QuantileTransformer, 
    RobustScaler,
    PolynomialFeatures,
    KBinsDiscretizer,
)
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
import psycopg
from dotenv import load_dotenv
load_dotenv()

TABLE_NAME = 'users_churn'

TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = 'features_experiment'
RUN_NAME = "preprocessing" 
REGISTRY_MODEL_NAME = 'features_experiment_model'

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

df = pd.DataFrame(data, columns=columns).dropna()

obj_df = df.select_dtypes(include="object")

# определение категориальных колонок, которые будут преобразованы
cat_columns = ["type", "payment_method", "internet_service", "gender"]

# создание объекта OneHotEncoder для преобразования категориальных переменных
# auto - автоматическое определение категорий
# ignore - игнорировать ошибки, если встречается неизвестная категория
# max_categories - максимальное количество уникальных категорий
# sparse_output - вывод в виде разреженной матрицы, если False, то в виде обычного массива
# drop="first" - удаляет первую категорию, чтобы избежать ловушки мультиколлинеарности
encoder_oh = OneHotEncoder(categories='auto', handle_unknown='ignore', max_categories=10, sparse_output=False, drop='first')

preprocessor = ColumnTransformer(
    transformers=[
        ('ohe', encoder_oh, cat_columns)
    ]
)

pipe = Pipeline([
    ('preprocess', preprocessor)
])

# применение OneHotEncoder к данным. Преобразование категориальных данных в массив
df_cat = df[cat_columns]
encoded_features = preprocessor.fit_transform(df_cat)

# преобразование полученных признаков в DataFrame и установка названий колонок
feature_names = preprocessor.get_feature_names_out()
encoded_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)

# конкатенация исходного DataFrame с новым DataFrame, содержащим закодированные категориальные признаки
# axis=1 означает конкатенацию по колонкам
obj_df = pd.concat([obj_df, encoded_df], axis=1)

# Предполагается, что num_df — это подтаблица из df только для числовых колонок:
num_columns = ["monthly_charges", "total_charges"]

n_knots = 3
degree_spline = 4
n_quantiles = 100
degree = 3
n_bins = 5
encode = 'ordinal'
strategy = 'uniform'
subsample = None

df_num = df[num_columns]

# ---- SplineTransformer ----
encoder_spl = SplineTransformer(n_knots=n_knots, degree=degree_spline)
encoded_features = encoder_spl.fit_transform(df_num)
encoded_df = pd.DataFrame(encoded_features, columns=encoder_spl.get_feature_names_out(num_columns))
# encoded_df = encoded_df.iloc[:, 1 + len(num_columns):]
# encoded_df.columns = [col + f'_spl' for col in num_columns]
num_df = pd.concat([df[num_columns].copy(), encoded_df], axis=1)

# ---- QuantileTransformer ----
encoder_q = QuantileTransformer(n_quantiles=n_quantiles)
encoded_features = encoder_q.fit_transform(df_num)
encoded_df = pd.DataFrame(encoded_features, columns=encoder_q.get_feature_names_out(num_columns))
# encoded_df = encoded_df.iloc[:, 1 + len(num_columns):]
# encoded_df.columns = [col + f'_q_{n_quantiles}' for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)

# ---- RobustScaler ----
encoder_rb = RobustScaler()
encoded_features = encoder_rb.fit_transform(df_num)
encoded_df = pd.DataFrame(encoded_features, columns=encoder_rb.get_feature_names_out(num_columns))
# encoded_df = encoded_df.iloc[:, 1 + len(num_columns):]
# encoded_df.columns = [col + f'_robust' for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)

# ---- PolynomialFeatures ----
encoder_pol = PolynomialFeatures(degree=degree)
encoded_features = encoder_pol.fit_transform(df_num)
encoded_df = pd.DataFrame(encoded_features, columns=encoder_pol.get_feature_names_out(num_columns))
# encoded_df = encoded_df.iloc[:, 1 + len(num_columns):]
# encoded_df.columns = [col + f'_poly' for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)

# ---- KBinsDiscretizer ----
encoder_kbd = KBinsDiscretizer(n_bins=n_bins, encode=encode, strategy=strategy)
encoded_features = encoder_kbd.fit_transform(df_num)
encoded_df = pd.DataFrame(encoded_features, columns=encoder_kbd.get_feature_names_out(num_columns))
# encoded_df = encoded_df.iloc[:, 1 + len(num_columns):]
# encoded_df.columns = [col + f'_bin' for col in num_columns]
num_df = pd.concat([num_df, encoded_df], axis=1)

# ----
print(num_df.head(2))

numeric_transformer = ColumnTransformer(
    transformers=[
        ("imputer", SimpleImputer(strategy="mean"), num_columns),
        ('features', FeatureUnion([
            ('spl', encoder_spl),
            ('q', encoder_q),
            ('rb', encoder_rb),
            ('pol', encoder_pol),
            ('kbd', encoder_kbd)
        ]), num_columns)
    ]
)

categorical_transformer = Pipeline(steps=[('encoder', encoder_oh)])

preprocessor = ColumnTransformer(
transformers=[('num', numeric_transformer, num_columns),
('cat', categorical_transformer, cat_columns)], n_jobs=-1)

encoded_features = preprocessor.fit_transform(df)

transformed_df = pd.DataFrame(encoded_features, 
                 columns=preprocessor.get_feature_names_out())

df = pd.concat([df, transformed_df], axis=1)
df.head(2)

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

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

    mlflow.sklearn.log_model(preprocessor, "column_transformer")