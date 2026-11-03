import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from scripts import train


def test_cross_validation_logs_mean_and_standard_deviation(monkeypatch) -> None:
    captured = {}
    results = {
        f"test_{metric}": np.array([0.6, 0.8]) for metric in train.CROSS_VALIDATION_METRICS
    }
    monkeypatch.setattr(train, "cross_validate", lambda *args, **kwargs: results)
    monkeypatch.setattr(train.mlflow, "log_metrics", captured.update)

    metrics = train.cross_validate_model(object(), [1, 2], [False, True], random_state=7)

    assert metrics == captured
    assert metrics["cv_average_precision_mean"] == 0.7
    assert metrics["cv_average_precision_std"] == pytest.approx(0.1)
