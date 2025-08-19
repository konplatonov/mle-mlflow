export MLFLOW_S3_ENDPOINT_URL=https://storage.yandexcloud.net
export AWS_ACCESS_KEY_ID=$S3_ACCESS_KEY
export AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
export AWS_BUCKET_NAME=$S3_BUCKET_NAME

# sudo apt-get update
# sudo apt-get install python3.10-venv
# python3.10 -m venv .venv_mlflow_server
# source .venv_mlflow_server/bin/activate
# pip install -r requirements.txt
set -a
source .env
set +a

mlflow server \
  --backend-store-uri postgresql://mle_20250619_6c56b203e2_freetrack:93034d5fc7414b029650a561585a31a5@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250619_6c56b203e2\
  --registry-store-uri  postgresql://mle_20250619_6c56b203e2_freetrack:93034d5fc7414b029650a561585a31a5@rc1b-uh7kdmcx67eomesf.mdb.yandexcloud.net:6432/playground_mle_20250619_6c56b203e2\
  --default-artifact-root s3://s3-student-mle-20250619-6c56b203e2-freetrack\
  --serve-artifacts