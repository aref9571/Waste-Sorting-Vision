import json
from pathlib import Path

import pytest

from wastevision.evaluation.threshold import (
    ThresholdChoice,
    ThresholdError,
    add_threshold_to_metadata,
    coverage_and_accuracy,
    select_threshold,
)

CONFIDENCES = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
CORRECT = [True, True, True, True, True, False, True, False]


def test_coverage_and_accuracy_count_only_answered_images():
    answered, coverage, accuracy = coverage_and_accuracy(CONFIDENCES, CORRECT, 0.7)

    assert answered == 4
    assert coverage == 0.5
    assert accuracy == 1.0


def test_threshold_is_inclusive():
    answered, _, _ = coverage_and_accuracy([0.5, 0.4], [True, False], 0.5)

    assert answered == 1


def test_nothing_answered_gives_zero_accuracy():
    assert coverage_and_accuracy([0.5, 0.4], [True, False], 0.9) == (0, 0.0, 0.0)


def test_select_threshold_takes_the_lowest_value_that_reaches_the_target():
    choice = select_threshold(CONFIDENCES, CORRECT, target_accuracy=0.8)

    assert choice.threshold == 0.4
    assert choice.answered == 7
    assert choice.coverage == 7 / 8
    assert choice.accuracy == 6 / 7
    assert choice.images == 8
    assert choice.target_accuracy == 0.8


def test_select_threshold_with_a_stricter_target_answers_less():
    choice = select_threshold(CONFIDENCES, CORRECT, target_accuracy=1.0)

    assert choice.threshold == 0.6
    assert choice.answered == 5
    assert choice.accuracy == 1.0


def test_all_correct_predictions_are_all_answered():
    choice = select_threshold([0.9, 0.5, 0.3], [True, True, True])

    assert choice.threshold == 0.3
    assert choice.coverage == 1.0


def test_target_that_cannot_be_reached_is_an_error():
    with pytest.raises(ThresholdError, match="No threshold reaches"):
        select_threshold([0.9, 0.8], [False, False])


@pytest.mark.parametrize("target", [0.0, -0.1, 1.5])
def test_invalid_target_is_rejected(target: float):
    with pytest.raises(ThresholdError, match="target_accuracy"):
        select_threshold(CONFIDENCES, CORRECT, target)


def test_empty_input_is_rejected():
    with pytest.raises(ThresholdError, match="without any predictions"):
        select_threshold([], [])


def test_different_lengths_are_rejected():
    with pytest.raises(ThresholdError, match="same length"):
        coverage_and_accuracy([0.5], [True, False], 0.5)


def test_threshold_is_added_to_the_metadata_without_touching_other_keys(tmp_path: Path):
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps({"model_sha256": "abc", "opset": 18}), encoding="utf-8")
    choice = ThresholdChoice(0.4, 0.95, 378, 358, 358 / 378, 0.95)

    add_threshold_to_metadata(path, choice)

    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["model_sha256"] == "abc"
    assert stored["opset"] == 18
    assert stored["confidence_threshold"]["threshold"] == 0.4
    assert stored["confidence_threshold"]["answered"] == 358
    assert path.read_text(encoding="utf-8").endswith("}\n")


def test_adding_a_threshold_twice_replaces_the_first(tmp_path: Path):
    path = tmp_path / "metadata.json"
    path.write_text("{}", encoding="utf-8")

    add_threshold_to_metadata(path, ThresholdChoice(0.4, 0.95, 10, 9, 0.9, 1.0))
    add_threshold_to_metadata(path, ThresholdChoice(0.6, 0.95, 10, 7, 0.7, 1.0))

    assert json.loads(path.read_text(encoding="utf-8"))["confidence_threshold"]["threshold"] == 0.6
