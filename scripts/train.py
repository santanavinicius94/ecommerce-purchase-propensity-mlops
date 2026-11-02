"""Treina e persiste o pipeline usando a divisão produzida por prepare_data."""

import argparse
import json

import joblib
import mlflow
from common import (
    DEFAULT_CONFIG,
    configure_mlflow,
    ensure_parent,
    load_config,
    project_path,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from ecommerce_purchase_propensity_mlops.preprocessing import (
    assemble_full_pipeline,
    create_preprocessing_pipeline,
)
from ecommerce_purchase_propensity_mlops.training import save_artifact, train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)

    split = joblib.load(project_path(config["paths"]["split_data"]))
    model_config = config["model"]
    model_path = project_path(config["paths"]["model"])
    run_path = project_path(config["paths"]["mlflow_run"])
    features = config["features"]

    if model_config["type"] == "random_forest":
        model = RandomForestClassifier(
            n_estimators=model_config["n_estimators"],
            max_depth=model_config["max_depth"],
            class_weight=model_config["class_weight"],
            random_state=model_config["random_state"],
            n_jobs=model_config["n_jobs"],
        )
    elif model_config["type"] == "logistic_regression":
        model = LogisticRegression(
            random_state=model_config["random_state"],
            class_weight=model_config["class_weight"],
        )
    else:
        raise ValueError(f"Modelo não suportado: {model_config['type']}")
    
    preprocessor = create_preprocessing_pipeline(
        features["numeric"], features["categorical"]
    )
    pipeline = assemble_full_pipeline(preprocessor, model)

    configure_mlflow(config)
    mlflow.sklearn.autolog()
    with mlflow.start_run() as run:
        mlflow.log_params(
            {
                "target": config["split"]["target"],
                "test_size": config["split"]["test_size"],
                "split_random_state": config["split"]["random_state"],
                "stratify": config["split"]["stratify"],
            }
        )
        trained_pipeline = train_model(pipeline, split["X_train"], split["y_train"])
        ensure_parent(model_path)
        save_artifact(trained_pipeline, model_path)
        ensure_parent(run_path)
        run_path.write_text(json.dumps({"run_id": run.info.run_id}), encoding="utf-8")


if __name__ == "__main__":
    main()
