"""Prepara a divisão de dados usada pelas etapas de treino e avaliação."""

import argparse

import joblib
from common import (
    DEFAULT_CONFIG,
    ensure_parent,
    load_config,
    project_path,
)

from ecommerce_purchase_propensity_mlops.preprocessing import (
    load_data,
    normalize_columns,
    split_data,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)

    paths = config["paths"]
    split_config = config["split"]
    raw_data_path = project_path(paths["raw_data"])
    split_path = project_path(paths["split_data"])

    data = load_data(raw_data_path)
    data = normalize_columns(data)
    X_train, X_test, y_train, y_test = split_data(
        data,
        target_col=split_config["target"],
        test_size=split_config["test_size"],
        random_state=split_config["random_state"],
        stratify=split_config["stratify"],
    )

    ensure_parent(split_path)
    joblib.dump(
        {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        split_path,
    )


if __name__ == "__main__":
    main()
