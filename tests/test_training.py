"""Testes unitários das funções de treinamento e avaliação."""

import joblib
import numpy as np

from ecommerce_purchase_propensity_mlops import training


class TrainablePipeline:
    def fit(self, features, target):
        self.features = features
        self.target = target
        return self


class ScoringPipeline:
    def predict(self, features):
        return np.array([True, False, False, False])

    def predict_proba(self, features):
        return np.array([[0.1, 0.9], [0.9, 0.1], [0.6, 0.4], [0.8, 0.2]])


def test_train_model_fits_and_returns_pipeline() -> None:
    pipeline = TrainablePipeline()
    features = [1, 2]
    target = [False, True]

    result = training.train_model(pipeline, features, target)

    assert result is pipeline
    assert pipeline.features == features
    assert pipeline.target == target


def test_evaluate_model_returns_metrics_and_logs_them(monkeypatch, capsys) -> None:
    logged = {}
    monkeypatch.setattr(training.mlflow, "log_metric", logged.__setitem__)

    metrics = training.evaluate_model(
        "unit-test",
        ScoringPipeline(),
        X_test=[1, 2, 3, 4],
        y_test=np.array([True, False, True, False]),
    )

    assert metrics == {
        "precision": 1.0,
        "recall": 0.5,
        "accuracy": 0.75,
        "f1": 0.6667,
        "roc_auc": 1.0,
    }
    assert set(logged) == {f"test_{name}" for name in metrics}
    assert np.isclose(logged["test_f1"], 2 / 3)
    assert "=== unit-test ===" in capsys.readouterr().out


def test_save_artifact_creates_parent_and_serializes(tmp_path) -> None:
    destination = tmp_path / "models" / "pipeline.joblib"
    artifact = {"model": "unit-test"}

    training.save_artifact(artifact, destination)

    assert joblib.load(destination) == artifact
