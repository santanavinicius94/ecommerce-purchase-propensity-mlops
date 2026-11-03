"""Testes unitários das transformações de pré-processamento."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ecommerce_purchase_propensity_mlops.preprocessing import (
    _create_custom_num_features,
    assemble_full_pipeline,
    create_preprocessing_pipeline,
    load_data,
    normalize_columns,
    split_data,
)


def test_load_data_and_normalize_columns(tmp_path) -> None:
    path = tmp_path / "data.csv"
    source = pd.DataFrame({"Page Values": [0, 1], "Revenue": [False, True]})
    source.to_csv(path, index=False)

    loaded = load_data(path)
    normalized = normalize_columns(loaded)

    pd.testing.assert_frame_equal(loaded, source)
    assert list(normalized.columns) == ["page_values", "revenue"]


def test_split_data_is_stratified_and_reproducible() -> None:
    frame = pd.DataFrame({"feature": range(20), "revenue": [False] * 15 + [True] * 5})

    first = split_data(frame, "revenue", stratify=True)
    second = split_data(frame, "revenue", stratify=True)

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])
    pd.testing.assert_series_equal(first[2], second[2])
    pd.testing.assert_series_equal(first[3], second[3])
    assert "revenue" not in first[0]
    assert first[2].mean() == first[3].mean() == 0.25


def test_numeric_and_full_pipeline_transform_and_predict() -> None:
    features = pd.DataFrame({"count": [0, 1, 2, 3], "month": ["Feb", "Feb", "Mar", "Mar"]})
    target = pd.Series([False, False, True, True])

    transformed_numbers = _create_custom_num_features(features[["count"]])
    preprocessor = create_preprocessing_pipeline(["count"], ["month"])
    pipeline = assemble_full_pipeline(preprocessor, LogisticRegression(random_state=7))
    pipeline.fit(features, target)

    np.testing.assert_allclose(transformed_numbers["count"], np.log1p(features["count"]))
    assert len(pipeline.predict(features.assign(month="UNSEEN"))) == len(features)
