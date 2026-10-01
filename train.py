import os
from io import StringIO

import boto3
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------- Config ----------------
BUCKET = os.getenv("S3_BUCKET", "mlops-house-prediction")
KEY = os.getenv("S3_KEY", "proccessed/2026-09-25/Mlops_house_predication_clean_v1.csv")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

EXPERIMENT_NAME = "mlops-house-prediction"
MODEL_NAME = "house-price-predictor"   # serve.py isi naam se model load karta hai
MODEL_ALIAS = "champion"

FEATURES = ["sqft", "bedrooms", "bathrooms", "age_years", "garage", "location_score"]
TARGET = "price"

N_ESTIMATORS = 150
MAX_DEPTH = 8


# ---------------- Data ----------------
def fetch_data() -> pd.DataFrame:
    """S3 se cleaned CSV fetch karta hai (AWS credentials chahiye)."""
    s3 = boto3.client("s3")
    obj = s3.get_object(Bucket=BUCKET, Key=KEY)
    return pd.read_csv(StringIO(obj["Body"].read().decode("utf-8")))


# ---------------- Training ----------------
def main():
    df = fetch_data()
    print(f"Fetched shape: {df.shape}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {TRACKING_URI}")

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=42,
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("max_depth", MAX_DEPTH)
        mlflow.log_param("data_source", f"s3://{BUCKET}/{KEY}")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        # Model log + registry mein register
        mlflow.sklearn.log_model(
            model,
            "model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        print(f"\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.4f}")

    # Latest version ko 'champion' alias do
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    latest = max(int(v.version) for v in versions)
    client.set_registered_model_alias(MODEL_NAME, MODEL_ALIAS, str(latest))
    print(f"Alias '{MODEL_ALIAS}' -> {MODEL_NAME} version {latest}")


if __name__ == "__main__":
    main()