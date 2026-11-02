"""Avalia o modelo e registra as métricas no mesmo run do MLflow."""

import argparse
import json

import joblib
import mlflow
from common import (
    DEFAULT_CONFIG,
    configure_mlflow,
    ensure_parent,
    load_config,
    load_run_id,
    project_path,
)

from ecommerce_purchase_propensity_mlops.training import evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)

    split = joblib.load(project_path(config["paths"]["split_data"]))
    model = joblib.load(project_path(config["paths"]["model"]))
    metrics_path = project_path(config["paths"]["metrics"])

    configure_mlflow(config)
    with mlflow.start_run(run_id=load_run_id(config)):
        mlflow.sklearn.autolog()
        metrics = evaluate_model(config["model"]["name"], model, split["X_test"], split["y_test"])

    ensure_parent(metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
