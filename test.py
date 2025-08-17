import os
print("AWS_ACCESS_KEY_ID:", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET_ACCESS_KEY:", os.getenv("AWS_SECRET_ACCESS_KEY"))
print("S3_BUCKET_NAME:", os.getenv("S3_BUCKET_NAME"))
print("MLFLOW_S3_ENDPOINT_URL:", os.getenv("MLFLOW_S3_ENDPOINT_URL"))

import pandas as pd
from sqlalchemy import create_engine
engine = create_engine("postgresql://mle_20250619_6c56b203e2_freetrack:93034d5fc7414b029650a561585a31a5@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250619_6c56b203e2")
df = pd.read_sql("SELECT name,version,run_id,status FROM model_versions", engine)
# df = pd.read_sql("select * from model_versions where name = 'model_with_prepro' and version = 3 and run_id = 'd75d484312bc49579f55e6cfb677c700' and status = 'READY';", engine)
print(df)