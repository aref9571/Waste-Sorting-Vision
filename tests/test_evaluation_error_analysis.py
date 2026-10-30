import numpy as np
import pytest

from wastevision.evaluation.error_analysis import (
    ConfusedPair,
    gradcam_pairs,
    most_confident_correct,
    most_confused_pairs,
    worst_mistakes,
)

LABELS = np.array([0, 0, 1, 1, 2, 2, 0])
PROBABILITIES = np.array(
    [
        [0.9, 0.05, 0.05],
        [0.2, 0.7, 0.1],
        [0.1, 0.8, 0.1],
        [0.6, 0.3, 0.1],
        [0.05, 0.05, 0.9],
        [0.1, 0.85, 0.05],
        [0.5, 0.3, 0.2],
    ]
)


def test_worst_mistakes_are_the_most_confident_wrong_predictions():
    assert worst_mistakes(LABELS, PROBABILITIES, count=10) == [5, 1, 3]


def test_worst_mistakes_keeps_only_count():
    assert worst_mistakes(LABELS, PROBABILITIES, count=2) == [5, 1]


def test_worst_mistakes_keeps_the_order_of_equal_confidences():
    labels = np.array([1, 1, 1])
    probabilities = np.array([[0.6, 0.4], [0.7, 0.3], [0.6, 0.4]])

    assert worst_mistakes(labels, probabilities, count=3) == [1, 0, 2]


def test_worst_mistakes_is_empty_without_mistakes():
    labels = np.array([0, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8]])

    assert worst_mistakes(labels, probabilities, count=5) == []


def test_most_confident_correct_picks_the_highest_confidence_of_the_class():
    assert most_confident_correct(LABELS, PROBABILITIES, class_index=0) == 0
    assert most_confident_correct(LABELS, PROBABILITIES, class_index=2) == 4


def test_most_confident_correct_rejects_a_class_without_correct_predictions():
    labels = np.array([0, 1])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2]])

    with pytest.raises(ValueError, match="No correct prediction for class 1"):
        most_confident_correct(labels, probabilities, class_index=1)


def test_gradcam_pairs_match_each_worst_mistake_with_a_correct_image_of_its_class():
    assert gradcam_pairs(LABELS, PROBABILITIES, count=3) == [(4, 5), (0, 1), (2, 3)]


def test_gradcam_pairs_use_each_true_class_once():
    labels = np.array([0, 0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4], [0.1, 0.9]])

    assert gradcam_pairs(labels, probabilities, count=3) == [(0, 1), (4, 3)]


def test_gradcam_pairs_skip_classes_without_a_correct_image():
    labels = np.array([0, 1, 1])
    probabilities = np.array([[0.1, 0.9], [0.8, 0.2], [0.4, 0.6]])

    assert gradcam_pairs(labels, probabilities, count=2) == [(2, 1)]


def test_most_confused_pairs_sorts_off_diagonal_cells():
    confusion = np.array([[5, 1, 0], [3, 7, 2], [0, 3, 9]])

    assert most_confused_pairs(confusion, count=3) == [
        ConfusedPair(1, 0, 3),
        ConfusedPair(2, 1, 3),
        ConfusedPair(1, 2, 2),
    ]


def test_most_confused_pairs_leaves_out_empty_cells():
    confusion = np.array([[4, 0], [1, 6]])

    assert most_confused_pairs(confusion, count=5) == [ConfusedPair(1, 0, 1)]


def test_most_confused_pairs_rejects_a_matrix_that_is_not_square():
    with pytest.raises(ValueError, match="square"):
        most_confused_pairs(np.zeros((2, 3)), count=1)


@pytest.mark.parametrize(
    "select",
    [
        lambda: worst_mistakes(LABELS, PROBABILITIES, count=0),
        lambda: gradcam_pairs(LABELS, PROBABILITIES, count=0),
        lambda: most_confused_pairs(np.eye(3), count=0),
    ],
)
def test_count_must_be_positive(select):
    with pytest.raises(ValueError, match="count must be at least 1"):
        select()


@pytest.mark.parametrize(
    ("labels", "probabilities"),
    [
        (np.array([0, 1]), np.array([0.5, 0.5])),
        (np.array([0, 1, 1]), np.array([[0.5, 0.5], [0.4, 0.6]])),
    ],
)
def test_labels_and_probabilities_must_match(labels: np.ndarray, probabilities: np.ndarray):
    with pytest.raises(ValueError, match="one label per row"):
        worst_mistakes(labels, probabilities, count=1)
    with pytest.raises(ValueError, match="one label per row"):
        gradcam_pairs(labels, probabilities, count=1)
