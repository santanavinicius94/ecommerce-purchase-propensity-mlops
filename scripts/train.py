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
from sklearn.model_selection import StratifiedKFold, cross_validate

from ecommerce_purchase_propensity_mlops.preprocessing import (
    assemble_full_pipeline,
    create_preprocessing_pipeline,
)
from ecommerce_purchase_propensity_mlops.training import save_artifact, train_model

CROSS_VALIDATION_METRICS = (
    "average_precision",
    "precision",
    "recall",
    "f1",
    "roc_auc",
)


def cross_validate_model(pipeline, X_train, y_train, random_state: int) -> dict[str, float]:
    """Avalia o pipeline nos folds estratificados do conjunto de treino."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=folds,
        scoring=CROSS_VALIDATION_METRICS,
        n_jobs=1,
        error_score="raise",
    )
    metrics = {
        f"cv_{metric}_{stat}": float(getattr(results[f"test_{metric}"], stat)())
        for metric in CROSS_VALIDATION_METRICS
        for stat in ("mean", "std")
    }
    mlflow.log_metrics(metrics)
    return metrics


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
        cross_validate_model(
            pipeline,
            split["X_train"],
            split["y_train"],
            config["split"]["random_state"],
        )
        trained_pipeline = train_model(pipeline, split["X_train"], split["y_train"])
        ensure_parent(model_path)
        save_artifact(trained_pipeline, model_path)
        ensure_parent(run_path)
        run_path.write_text(json.dumps({"run_id": run.info.run_id}), encoding="utf-8")


if __name__ == "__main__":
    main()
