import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from wastevision.inference.predictor import Predictor, load_metadata, softmax
from wastevision.labels import CATEGORY_BY_CLASS, CLASS_NAMES

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH = MODELS_DIR / "waste-mobilenetv3.onnx"


@pytest.fixture(scope="module")
def metadata() -> dict:
    return load_metadata(MODELS_DIR / "metadata.json")


@pytest.fixture(scope="module")
def predictor(metadata: dict) -> Predictor:
    return Predictor(MODEL_PATH, metadata)


def random_image(seed: int = 0, size: tuple[int, int] = (320, 240)) -> Image.Image:
    pixels = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), np.uint8)
    return Image.fromarray(pixels)


def test_softmax_rows_sum_to_one():
    result = softmax(np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]]))

    assert np.allclose(result.sum(axis=1), 1.0)
    assert np.allclose(result[1], 1 / 3)
    assert result[0].argmax() == 2


def test_softmax_matches_the_hand_computed_values():
    result = softmax(np.array([0.0, np.log(3.0)]))

    assert np.allclose(result, [0.25, 0.75])


def test_softmax_is_stable_for_large_logits():
    result = softmax(np.array([1000.0, 1001.0, 999.0]))

    assert np.isfinite(result).all()
    assert np.isclose(result.sum(), 1.0)
    assert result.argmax() == 1


def test_probabilities_form_a_distribution_over_the_classes(predictor: Predictor):
    probabilities = predictor.probabilities(random_image())

    assert probabilities.shape == (len(CLASS_NAMES),)
    assert np.isclose(probabilities.sum(), 1.0)
    assert (probabilities >= 0).all()


def test_prediction_lists_the_three_most_likely_classes_in_order(predictor: Predictor):
    prediction = predictor.predict(random_image())

    labels = [label for label, _ in prediction.top]
    probabilities = [probability for _, probability in prediction.top]
    assert len(prediction.top) == 3
    assert len(set(labels)) == 3
    assert set(labels) <= set(CLASS_NAMES)
    assert probabilities == sorted(probabilities, reverse=True)
    assert sum(probabilities) <= 1.0 + 1e-9


def test_prediction_matches_the_most_likely_class(predictor: Predictor):
    image = random_image(4)
    probabilities = predictor.probabilities(image)

    prediction = predictor.predict(image)

    assert prediction.label == CLASS_NAMES[int(probabilities.argmax())]
    assert prediction.confidence == pytest.approx(float(probabilities.max()))
    assert prediction.category == CATEGORY_BY_CLASS[prediction.label]


def test_the_same_image_gives_the_same_prediction(predictor: Predictor):
    assert predictor.predict(random_image(2)) == predictor.predict(random_image(2))


@pytest.mark.parametrize("size", [(64, 48), (48, 64), (1200, 900)])
def test_images_of_any_size_can_be_classified(predictor: Predictor, size: tuple[int, int]):
    assert predictor.predict(random_image(size=size)).label in CLASS_NAMES


def test_low_confidence_is_unsure(metadata: dict):
    strict = Predictor(MODEL_PATH, metadata, threshold=1.0)

    assert strict.predict(random_image()).unsure is True


def test_zero_threshold_is_never_unsure(metadata: dict):
    lenient = Predictor(MODEL_PATH, metadata, threshold=0.0)

    assert lenient.predict(random_image()).unsure is False


def test_confidence_equal_to_the_threshold_is_answered(metadata: dict, predictor: Predictor):
    image = random_image(6)
    confidence = predictor.predict(image).confidence

    at_threshold = Predictor(MODEL_PATH, metadata, threshold=confidence)
    just_above = Predictor(MODEL_PATH, metadata, threshold=min(1.0, confidence + 1e-6))

    assert at_threshold.predict(image).unsure is False
    assert just_above.predict(image).unsure is True


def test_threshold_comes_from_the_metadata(metadata: dict, predictor: Predictor):
    assert predictor.threshold == metadata["confidence_threshold"]["threshold"]


def test_threshold_in_the_metadata_matches_its_own_numbers(metadata: dict):
    choice = metadata["confidence_threshold"]

    assert 0.0 < choice["threshold"] < 1.0
    assert choice["accuracy"] >= choice["target_accuracy"]
    assert choice["coverage"] == pytest.approx(choice["answered"] / choice["images"])


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_threshold_outside_zero_and_one_is_rejected(metadata: dict, threshold: float):
    with pytest.raises(ValueError, match="between 0 and 1"):
        Predictor(MODEL_PATH, metadata, threshold=threshold)


def test_metadata_without_a_threshold_is_rejected(metadata: dict):
    without = {key: value for key, value in metadata.items() if key != "confidence_threshold"}

    with pytest.raises(ValueError, match="no confidence threshold"):
        Predictor(MODEL_PATH, without)


def test_missing_model_is_rejected(metadata: dict, tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        Predictor(tmp_path / "missing.onnx", metadata)


def test_predictor_can_be_built_from_the_models_folder():
    predictor = Predictor.from_directory(MODELS_DIR)

    assert predictor.class_names == CLASS_NAMES
    assert predictor.predict(random_image()).label in CLASS_NAMES


def test_metadata_is_read_from_json(tmp_path: Path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"a": 1}), encoding="utf-8")

    assert load_metadata(path) == {"a": 1}
