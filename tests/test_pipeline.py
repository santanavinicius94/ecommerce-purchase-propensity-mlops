"""Testes com dados sintéticos e tracking isolado do ambiente local."""

import os
import subprocess
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
from scripts.train_model import CAT_FEATURES, NUM_FEATURES, run_training
from scripts.validate_data import COLUMNS, validate_dataset


@pytest.fixture
def csv_path(tmp_path):
    rows = []
    for i in range(80):
        rows.append(
            [
                i % 4,
                i * 2,
                i % 3,
                i * 3,
                i + 1,
                i * 5,
                0.02,
                0.04,
                i % 7,
                0,
                "Feb" if i % 3 else "Mar",
                1,
                1,
                1,
                1,
                "Returning_Visitor" if i % 3 else "New_Visitor",
                "TRUE" if i % 2 else "FALSE",
                "TRUE" if i % 4 == 0 else "FALSE",
            ]
        )
    path = tmp_path / "sessions.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(path, index=False)
    return path


@pytest.fixture
def frame(csv_path):
    return normalize_columns(load_data(csv_path))


def test_csv_contract_and_normalization(csv_path):
    original = load_data(csv_path)
    copy = original.copy(deep=True)
    normalized = normalize_columns(original)
    pd.testing.assert_frame_equal(original, copy)
    assert list(normalized.columns) == [column.lower() for column in COLUMNS]
    assert validate_dataset(csv_path)["target_counts"] == {"FALSE": 60, "TRUE": 20}


def test_split_is_stratified_reproducible_and_excludes_target(frame):
    first = split_data(frame, "revenue", stratify=True)
    second = split_data(frame, "revenue", stratify=True)
    for left, right in zip(first, second):
        assert left.equals(right)
    X_train, X_test, y_train, y_test = first
    assert "revenue" not in X_train and "revenue" not in X_test
    assert set(X_train.index).isdisjoint(X_test.index)
    assert set(X_train.index) | set(X_test.index) == set(frame.index)
    assert y_train.mean() == y_test.mean() == 0.25


def test_preprocessing_fits_only_training_and_accepts_unseen_categories(frame):
    X_train, X_test, _, _ = split_data(frame, "revenue", stratify=True)
    preprocessor = create_preprocessing_pipeline(NUM_FEATURES, CAT_FEATURES)
    transformed = preprocessor.fit_transform(X_train)
    scaler = preprocessor.named_transformers_["num"].named_steps["standardscaler"]
    np.testing.assert_allclose(scaler.mean_, np.log1p(X_train[NUM_FEATURES]).mean())
    original_means = scaler.mean_.copy()
    unseen = X_test.copy()
    unseen["month"] = "UNSEEN"
    result = preprocessor.transform(unseen)
    result = result.toarray() if hasattr(result, "toarray") else result
    assert result.shape == (len(X_test), transformed.shape[1])
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(scaler.mean_, original_means)


def test_training_evaluation_and_serialization(frame, tmp_path, capsys):
    X_train, X_test, y_train, y_test = split_data(frame, "revenue", stratify=True)
    pipeline = assemble_full_pipeline(
        create_preprocessing_pipeline(NUM_FEATURES, CAT_FEATURES),
        LogisticRegression(random_state=7, class_weight="balanced"),
    )
    assert train_model(pipeline, X_train, y_train) is pipeline
    predictions = evaluate_model("test", pipeline, X_test, y_test)
    assert predictions.shape == (len(X_test),)
    assert set(predictions) <= {False, True}
    assert "accuracy:" in capsys.readouterr().out
    probabilities = pipeline.predict_proba(X_test)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1)
    destination = tmp_path / "nested/models/model.joblib"
    save_artifact(pipeline, destination)
    restored = joblib.load(destination)
    np.testing.assert_allclose(restored.predict_proba(X_test), probabilities)


def test_training_cli_tracks_model_and_test_metrics(csv_path, tmp_path):
    root = Path(__file__).resolve().parents[1]
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    artifact = tmp_path / "models/model.joblib"
    environment = os.environ.copy()
    environment["MLFLOW_TRACKING_URI"] = tracking_uri
    environment["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
    result = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/train_model.py"),
            "--data",
            str(csv_path),
            "--model",
            str(artifact),
            "--experiment",
            "pipeline-test",
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("pipeline-test")
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]
    assert run.info.status == "FINISHED"
    assert {
        "test_precision",
        "test_recall",
        "test_accuracy",
        "test_f1",
    } <= run.data.metrics.keys()
    model = joblib.load(artifact)
    _, X_test, _, y_test = split_data(
        normalize_columns(load_data(csv_path)),
        "revenue",
        stratify=True,
    )
    assert run.data.metrics["test_accuracy"] == round(
        accuracy_score(y_test, model.predict(X_test)),
        4,
    )
    logged_models = client.search_logged_models(
        experiment_ids=[experiment.experiment_id]
    )
    assert len(logged_models) == 1
    restored = mlflow.sklearn.load_model(logged_models[0].artifact_location)
    np.testing.assert_allclose(
        restored.predict_proba(X_test), model.predict_proba(X_test)
    )


@pytest.mark.parametrize("fault", ["missing_column", "negative", "one_class"])
def test_training_rejects_invalid_data_before_tracking(
    csv_path, tmp_path, monkeypatch, fault
):
    df = pd.read_csv(csv_path)
    if fault == "missing_column":
        df = df.drop(columns="Revenue")
    elif fault == "negative":
        df.loc[0, "PageValues"] = -1
    else:
        df["Revenue"] = "FALSE"
    for column in ("Revenue", "Weekend"):
        if column in df and df[column].dtype == bool:
            df[column] = df[column].map({True: "TRUE", False: "FALSE"})
    df.to_csv(csv_path, index=False)

    def unexpected_tracking(*args, **kwargs):
        pytest.fail("Entrada inválida não deve iniciar tracking")

    monkeypatch.setattr(mlflow, "set_experiment", unexpected_tracking)
    destination = tmp_path / "model.joblib"
    with pytest.raises(ValueError):
        run_training(csv_path, destination, "invalid")
    assert not destination.exists()
