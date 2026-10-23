from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from wastevision.baseline.train import (
    PARAMETER_GRIDS,
    BaselineError,
    build_pipeline,
    evaluate,
    load_model,
    load_split_features,
    save_model,
    select_baseline,
)
from wastevision.labels import CLASS_NAMES


def separable_features(samples_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = []
    labels = []
    for index, class_name in enumerate(CLASS_NAMES):
        center = np.zeros(len(CLASS_NAMES))
        center[index] = 10.0
        features.append(center + rng.normal(0.0, 0.5, size=(samples_per_class, len(CLASS_NAMES))))
        labels.extend([class_name] * samples_per_class)
    return np.vstack(features), np.array(labels)


def candidate_count() -> int:
    return sum(
        int(np.prod([len(values) for values in grid.values()])) for grid in PARAMETER_GRIDS.values()
    )


def test_build_pipeline_scales_before_the_model():
    pipeline = build_pipeline("svm_rbf", {"C": 10.0, "gamma": "scale"})

    assert [name for name, _ in pipeline.steps] == ["scaler", "model"]
    assert isinstance(pipeline.named_steps["model"], SVC)
    assert pipeline.named_steps["model"].C == 10.0


def test_build_pipeline_creates_logistic_regression():
    pipeline = build_pipeline("logistic_regression", {"C": 0.1})

    assert isinstance(pipeline.named_steps["model"], LogisticRegression)


def test_build_pipeline_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        build_pipeline("random_forest", {})


def test_select_baseline_tries_every_candidate_on_validation():
    train_features, train_labels = separable_features(10, seed=0)
    validation_features, validation_labels = separable_features(5, seed=1)

    selection = select_baseline(
        train_features, train_labels, validation_features, validation_labels
    )

    assert len(selection.candidates) == candidate_count()
    assert {candidate.model_name for candidate in selection.candidates} == set(PARAMETER_GRIDS)
    assert selection.best.validation.macro_f1 == max(
        candidate.validation.macro_f1 for candidate in selection.candidates
    )
    assert selection.best.validation.accuracy == 1.0
    assert evaluate(selection.pipeline, validation_features, validation_labels).accuracy == 1.0


def test_select_baseline_keeps_the_first_of_equal_candidates():
    train_features, train_labels = separable_features(10, seed=0)
    validation_features, validation_labels = separable_features(5, seed=1)

    selection = select_baseline(
        train_features, train_labels, validation_features, validation_labels
    )

    assert selection.best is selection.candidates[0]


class FixedPredictions:
    def __init__(self, predictions: list[str]):
        self.predictions = np.array(predictions)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predictions


def test_evaluate_computes_accuracy_macro_f1_and_per_class_scores():
    labels = np.array(["glass", "glass", "metal", "metal"])
    pipeline = FixedPredictions(["glass", "metal", "metal", "metal"])

    result = evaluate(pipeline, np.zeros((4, 1)), labels)

    assert result.accuracy == pytest.approx(0.75)
    assert result.per_class["glass"].precision == pytest.approx(1.0)
    assert result.per_class["glass"].recall == pytest.approx(0.5)
    assert result.per_class["metal"].precision == pytest.approx(2 / 3)
    assert result.per_class["metal"].support == 2
    assert result.per_class["trash"].support == 0
    assert list(result.per_class) == list(CLASS_NAMES)
    glass_f1 = 2 * 1.0 * 0.5 / 1.5
    metal_f1 = 2 * (2 / 3) * 1.0 / (2 / 3 + 1.0)
    assert result.macro_f1 == pytest.approx((glass_f1 + metal_f1) / len(CLASS_NAMES))


def test_load_split_features_reads_only_the_requested_split(tiny_dataset: tuple[Path, Path]):
    images_dir, split_path = tiny_dataset

    features, labels = load_split_features(split_path, images_dir, "train")

    assert features.shape[0] == 3 * len(CLASS_NAMES)
    assert sorted(set(labels)) == list(CLASS_NAMES)
    assert np.isfinite(features).all()


def test_load_split_features_rejects_an_empty_split(tmp_path: Path):
    split_path = tmp_path / "split.csv"
    split_path.write_text("file,class,split\nglass/a.png,glass,train\n", encoding="utf-8")

    with pytest.raises(BaselineError, match="No images in the test split"):
        load_split_features(split_path, tmp_path, "test")


def test_load_split_features_fails_for_missing_image(tmp_path: Path):
    split_path = tmp_path / "split.csv"
    split_path.write_text("file,class,split\nglass/a.png,glass,train\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        load_split_features(split_path, tmp_path, "train")


def test_saved_model_predicts_like_the_original(tmp_path: Path):
    features, labels = separable_features(5, seed=2)
    pipeline = build_pipeline("logistic_regression", {"C": 1.0}).fit(features, labels)
    path = tmp_path / "runs" / "model.joblib"

    save_model(pipeline, path)

    np.testing.assert_array_equal(load_model(path).predict(features), pipeline.predict(features))


def test_load_model_reports_missing_file(tmp_path: Path):
    with pytest.raises(BaselineError, match="wastevision baseline train"):
        load_model(tmp_path / "missing.joblib")
