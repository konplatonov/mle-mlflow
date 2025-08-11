import os

import mlflow

EXPERIMENT_NAME = "churn_platonov"
RUN_NAME = "model_0_registry"
REGISTRY_MODEL_NAME = "churn_model_platonov"


os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://storage.yandexcloud.net"
os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")


pip_requirements = "../requirements.txt"
signature = mlflow.models.infer_signature(X_test, prediction)
input_example = [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
metadata = {'model_type': 'monthly'}
code_paths = ["train.py", "val_model.py"]
input_example = X_test[:10]

experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id

with mlflow.start_run(run_name=RUN_NAME, experiment_id=experiment_id) as run:
    run_id = run.info.run_id

model_info = mlflow.catboost.log_model(
    run_id = run.info.run_id,
    pip_requirements=pip_requirements,
    input_example=input_example,
    metadata=metadata,
    signature=signature,
    code_paths=code_paths,
    cb_model=model,
    artifact_path='models',
    registered_model_name=REGISTRY_MODEL_NAME,
    await_registration_for=60
    )

loaded_model = mlflow.pyfunc.load_model(model_uri=model_info.model_uri)
model_predictions = loaded_model.predict(X_test)
