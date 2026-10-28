import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

from wastevision.evaluation.metrics import classification_report, confusion_matrix

THREE_CLASSES = ("a", "b", "c")


def test_confusion_matrix_counts_true_rows_and_predicted_columns():
    labels = [0, 0, 1, 1, 2, 2]
    predictions = [0, 1, 1, 1, 0, 2]

    matrix = confusion_matrix(labels, predictions, 3)

    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert matrix.dtype == np.int64


def test_confusion_matrix_includes_classes_that_never_occur():
    matrix = confusion_matrix([0, 0], [0, 0], 3)

    assert matrix.tolist() == [[2, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_report_matches_hand_computed_values():
    labels = [0, 0, 0, 0, 1, 1, 2, 2]
    predictions = [0, 0, 0, 1, 1, 2, 2, 2]

    report = classification_report(labels, predictions, THREE_CLASSES)

    assert report.accuracy == pytest.approx(6 / 8)
    assert report.per_class["a"].precision == pytest.approx(1.0)
    assert report.per_class["a"].recall == pytest.approx(3 / 4)
    assert report.per_class["a"].f1 == pytest.approx(6 / 7)
    assert report.per_class["b"].precision == pytest.approx(1 / 2)
    assert report.per_class["b"].recall == pytest.approx(1 / 2)
    assert report.per_class["b"].f1 == pytest.approx(1 / 2)
    assert report.per_class["c"].precision == pytest.approx(2 / 3)
    assert report.per_class["c"].recall == pytest.approx(1.0)
    assert report.per_class["c"].f1 == pytest.approx(4 / 5)
    assert [scores.support for scores in report.per_class.values()] == [4, 2, 2]
    assert report.macro_f1 == pytest.approx((6 / 7 + 1 / 2 + 4 / 5) / 3)
    assert report.confusion.tolist() == [[3, 1, 0], [0, 1, 1], [0, 0, 2]]


def test_class_that_is_never_predicted_scores_zero():
    labels = [0, 1, 2, 2]
    predictions = [0, 1, 1, 1]

    report = classification_report(labels, predictions, THREE_CLASSES)

    assert report.per_class["c"].precision == 0.0
    assert report.per_class["c"].recall == 0.0
    assert report.per_class["c"].f1 == 0.0
    assert report.macro_f1 == pytest.approx((1.0 + 0.5) / 3)


def test_class_without_labels_scores_zero_and_counts_in_the_macro_average():
    report = classification_report([0, 1], [0, 1], THREE_CLASSES)

    assert report.per_class["c"].support == 0
    assert report.per_class["c"].recall == 0.0
    assert report.per_class["c"].f1 == 0.0
    assert report.accuracy == 1.0
    assert report.macro_f1 == pytest.approx(2 / 3)


def test_perfect_predictions():
    labels = [0, 1, 2, 1]

    report = classification_report(labels, labels, THREE_CLASSES)

    assert report.accuracy == 1.0
    assert report.macro_f1 == 1.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_report_agrees_with_scikit_learn(seed: int):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 6, size=200)
    predictions = np.where(rng.random(200) < 0.6, labels, rng.integers(0, 5, size=200))
    class_names = tuple(f"class{index}" for index in range(6))
    indices = list(range(6))

    report = classification_report(labels.tolist(), predictions.tolist(), class_names)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=indices, zero_division=0
    )
    assert report.accuracy == pytest.approx(accuracy_score(labels, predictions))
    assert report.macro_f1 == pytest.approx(
        f1_score(labels, predictions, labels=indices, average="macro", zero_division=0)
    )
    assert [scores.precision for scores in report.per_class.values()] == pytest.approx(precision)
    assert [scores.recall for scores in report.per_class.values()] == pytest.approx(recall)
    assert [scores.f1 for scores in report.per_class.values()] == pytest.approx(f1)
    assert [scores.support for scores in report.per_class.values()] == support.tolist()
    assert np.array_equal(
        report.confusion, sklearn_confusion_matrix(labels, predictions, labels=indices)
    )


def test_rejects_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        confusion_matrix([0, 1], [0], 3)


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="without any labels"):
        classification_report([], [], THREE_CLASSES)


@pytest.mark.parametrize(
    ("labels", "predictions"), [([0, 3], [0, 1]), ([0, 1], [0, -1]), ([0, 1], [5, 1])]
)
def test_rejects_class_indices_out_of_range(labels: list[int], predictions: list[int]):
    with pytest.raises(ValueError, match="between 0 and 2"):
        confusion_matrix(labels, predictions, 3)


def test_rejects_nested_sequences():
    with pytest.raises(ValueError, match="flat sequences"):
        confusion_matrix([[0, 1]], [[0, 1]], 3)
