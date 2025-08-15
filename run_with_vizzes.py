import os
import psycopg
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow

from dotenv import load_dotenv
load_dotenv()

# --- Настройки ---
TABLE_NAME = "users_churn"
TRACKING_SERVER_HOST = "127.0.0.1"
TRACKING_SERVER_PORT = 5000

EXPERIMENT_NAME = "churn_eda_experiment"      # обязательно УКАЖИ!
RUN_NAME = "eda_run"
ASSETS_DIR = "assets"

os.makedirs(ASSETS_DIR, exist_ok=True)

pd.options.display.max_columns = 100
pd.options.display.max_rows = 64
sns.set_style("white")
sns.set_theme(style="whitegrid")

# --- Подключение к БД ---
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
print(df.head(2))

# --- Категориальные графики ---
fig, axs = plt.subplots(2, 2, figsize=(16.5, 12.5))
fig.tight_layout(pad=1.6)

for ax, (x, y) in zip(
        axs.flat,
        [("type", "customer_id"), ("internet_service", "customer_id"),
         ("payment_method", "customer_id"), ("gender", "customer_id")]):
    agg_df = df.groupby(x).agg({y: "count"}).reset_index()
    sns.barplot(data=agg_df, x=x, y="customer_id", ax=ax)
    ax.set_title(f'Count {y} by {x} in train dataframe')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

plt.savefig(os.path.join(ASSETS_DIR, 'cat_features_1'))
plt.close()

# --- Визуализация бинарных фичей ---
binary_columns = [
    "online_security", "online_backup", "device_protection", "tech_support",
    "streaming_tv", "streaming_movies", "senior_citizen", "partner", "dependents"
]

heat_df = (df[binary_columns] == 'Yes').astype(int)
sns.heatmap(heat_df.T, cbar=False)
plt.title("Binary feature heatmap")
plt.savefig(os.path.join(ASSETS_DIR, 'cat_features_2_binary_heatmap'))
plt.close()

# --- Агрегация по begin_date ---
x = "begin_date"
charges_columns = ["monthly_charges", "total_charges"]
df = df.dropna(subset=charges_columns, how='any') # уберём NaN для платежей

stats = ["mean", "median", lambda x: x.mode().iloc[0]]

charges_monthly_agg = df[[x, charges_columns[0]]].groupby([x]).agg(stats).reset_index()
charges_monthly_agg.columns = [x, "monthly_mean", "monthly_median", "monthly_mode"]

charges_total_agg = df[[x, charges_columns[1]]].groupby([x]).agg(stats).reset_index()
charges_total_agg.columns = [x, "total_mean", "total_median", "total_mode"]

fig, axs = plt.subplots(2, 1, figsize=(6.5, 5.5))
fig.tight_layout(pad=2.5)

sns.lineplot(data=charges_monthly_agg, x=x, y='monthly_mean', ax=axs[0], label="mean")
sns.lineplot(data=charges_monthly_agg, x=x, y='monthly_median', ax=axs[0], label="median")
sns.lineplot(data=charges_monthly_agg, x=x, y='monthly_mode', ax=axs[0], label="mode")
axs[0].set_title(f"Count statistics for {charges_columns[0]} by {x}")
axs[0].legend()

sns.lineplot(data=charges_total_agg, x=x, y='total_mean', ax=axs[1], label="mean")
sns.lineplot(data=charges_total_agg, x=x, y='total_median', ax=axs[1], label="median")
sns.lineplot(data=charges_total_agg, x=x, y='total_mode', ax=axs[1], label="mode")
axs[1].set_title(f"Count statistics for {charges_columns[1]} by {x}")
axs[1].legend()

plt.savefig(os.path.join(ASSETS_DIR, 'charges_by_date'))
plt.close()

# --- Target distribution ---
x = "target"
target_agg = df[x].value_counts().reset_index()
target_agg.columns = [x, 'count']

sns.barplot(data=target_agg, x=x, y='count')
plt.title(f"{x} total distribution")
plt.savefig(os.path.join(ASSETS_DIR, 'target_count'))
plt.close()

# --- Анализ конверсии по датам и по полу ---
x = "begin_date"
target = "target"
stat = ["count"]

target_agg_by_date = df[[x, target]].groupby([x]).agg(stat).reset_index()
target_agg_by_date.columns = [x, "target_count"]

target_agg = df[[x, target, 'customer_id']].groupby([x, target]).count().reset_index()

conversion_agg = df[[x, target]].groupby([x])['target'].agg(['sum', 'count']).reset_index()
conversion_agg['conv'] = (conversion_agg['sum'] / conversion_agg['count']).round(2)

conversion_agg_gender = df[[x, target, 'gender']].groupby([x, 'gender'])[target].agg(['sum', 'count']).reset_index()
conversion_agg_gender['conv'] = (conversion_agg_gender['sum'] / conversion_agg_gender['count']).round(2)

fig, axs = plt.subplots(2, 2, figsize=(16.5, 12.5))
fig.tight_layout(pad=1.6)

sns.lineplot(data=target_agg_by_date, x=x, y="target_count", ax=axs[0, 0])
axs[0, 0].set_title("Target count by begin date")

sns.lineplot(data=target_agg, x=x, y="customer_id", hue=target, ax=axs[0, 1])
axs[0, 1].set_title("Target count type by begin date")

sns.lineplot(data=conversion_agg, x=x, y="conv", ax=axs[1, 0])
axs[1, 0].set_title("Conversion value")

sns.lineplot(data=conversion_agg_gender, x=x, y="conv", hue="gender", ax=axs[1, 1])
axs[1, 1].set_title("Conversion value by gender")

plt.savefig(os.path.join(ASSETS_DIR, 'target_by_date'))
plt.close()

# --- Распределения платёжных фичей по target ---
charges = ["monthly_charges", "total_charges"]
target = "target"

fig, axs = plt.subplots(2, 1, figsize=(6.5, 6.5))
fig.tight_layout(pad=1.5)

sns.histplot(data=df, x=charges[0], hue=target, kde=True, ax=axs[0])
axs[0].set_title(f"{charges[0]} distribution")

sns.histplot(data=df, x=charges[1], hue=target, kde=True, ax=axs[1])
axs[1].set_title(f"{charges[1]} distribution")

plt.savefig(os.path.join(ASSETS_DIR, 'charges_by_target_dist'))
plt.close()

# --- Логирование в MLflow ---
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
    mlflow.log_artifacts(ASSETS_DIR)

print(f"EDA графики логированы в MLflow и сохранены в {ASSETS_DIR}")