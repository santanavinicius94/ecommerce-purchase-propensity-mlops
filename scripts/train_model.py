"""Executa validação, treinamento e avaliação com tracking no MLflow."""

import argparse
from pathlib import Path

import mlflow
from sklearn.linear_model import LogisticRegression

from ecommerce_purchase_propensity_mlops.preprocessing import (
    assemble_full_pipeline,
    create_preprocessing_pipeline,
    load_data,
    normalize_columns,
    split_data,
)
from ecommerce_purchase_propensity_mlops.training import (
    evaluate_model,
    save_artifact,
    train_model,
)

if __package__:
    from scripts.validate_data import validate_dataset
else:
    from validate_data import validate_dataset

# Configurações do treinamento
# ==========

basepath = Path(__file__).resolve().parents[1]

DATA_FILEPATH = basepath / "data/raw/online_shoppers_intention.csv"
MODEL_FILEPATH = basepath / "models/logistic_regression.joblib"

RANDOM_STATE = 7
MODEL_NAME = "logistic_regression"


TARGET_COL = "revenue"
NUM_FEATURES = [
    "administrative",
    "administrative_duration",
    "informational",
    "informational_duration",
    "productrelated",
    "productrelated_duration",
    "bouncerates",
    "exitrates",
    "pagevalues",
    "specialday",
]

CAT_FEATURES = [
    "month",
    "operatingsystems",
    "browser",
    "region",
    "traffictype",
    "visitortype",
    "weekend",
]


def run_training(data_path: Path, model_path: Path, experiment: str) -> str:
    """Treina com dados válidos e retorna o identificador da execução MLflow."""
    validate_dataset(data_path)
    df = normalize_columns(load_data(data_path))
    if df[TARGET_COL].nunique() != 2:
        raise ValueError("Revenue deve conter as duas classes para treinamento")
    X_train, X_test, y_train, y_test = split_data(df, TARGET_COL, stratify=True)
    mlflow.set_experiment(experiment)
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name=MODEL_NAME) as run:
        preprocessor = create_preprocessing_pipeline(NUM_FEATURES, CAT_FEATURES)
        model = LogisticRegression(random_state=RANDOM_STATE, class_weight="balanced")
        pipeline = assemble_full_pipeline(preprocessor, model)
        trained_pipeline = train_model(pipeline, X_train, y_train)
        evaluate_model(MODEL_NAME, trained_pipeline, X_test, y_test)
        save_artifact(trained_pipeline, model_path)
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_FILEPATH)
    parser.add_argument("--model", type=Path, default=MODEL_FILEPATH)
    parser.add_argument("--experiment", default="online_shoppers_intention_experiment")
    args = parser.parse_args()
    run_id = run_training(args.data, args.model, args.experiment)
    print(f"MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
