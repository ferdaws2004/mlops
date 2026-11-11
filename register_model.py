# ===== 1. Import the libraries =====
import mlflow # the main MLflow module
from mlflow import MlflowClient # the "client": lets us manage the Registry
# ===== 2. Connect to the same database as Lab 3 =====
# It is the mlflow.db file of the current folder, which holds all our runs
mlflow.set_tracking_uri("sqlite:///mlflow.db")
# ===== 3. Find the best run (same logic as in Lab 3) =====
runs = mlflow.search_runs(
experiment_names=["churn-prediction"],
order_by=["metrics.f1 DESC"]) # f1 descending: the best one first
best = runs.iloc[0] # first row = best run
run_id = best["run_id"] # its unique identifier
print("Best run :", best["tags.mlflow.runName"], "| id :", run_id[:8], "...")
# ===== 4. Register the model of this run in the Registry =====
# "runs:/<id>/model" = the 'model' folder of this run
# "churn-model" = the name of the model in the catalog (created if it does not exist)
result = mlflow.register_model(f"runs:/{run_id}/model", "churn-model")
print("Model registered : churn-model | version", result.version)
# ===== 5. Put the alias "champion" on this version =====
client = MlflowClient()
client.set_registered_model_alias("churn-model", "champion", result.version)
print("Alias 'champion' -> version", result.version)