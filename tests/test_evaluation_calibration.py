import pytest

from wastevision.evaluation.calibration import expected_calibration_error, reliability_bins


def test_ece_is_zero_for_perfectly_calibrated_predictions():
    confidences = [0.8] * 10 + [0.5] * 4 + [1.0] * 3
    correct = [True] * 8 + [False] * 2 + [True, False, True, False] + [True] * 3

    bins = reliability_bins(confidences, correct)

    assert expected_calibration_error(bins) == pytest.approx(0.0)


def test_ece_of_overconfident_predictions():
    confidences = [0.9] * 10
    correct = [True] * 6 + [False] * 4

    assert expected_calibration_error(reliability_bins(confidences, correct)) == pytest.approx(0.3)


def test_ece_weights_each_bin_by_its_share_of_predictions():
    confidences = [0.95] * 3 + [0.55]
    correct = [True, True, True, False]

    ece = expected_calibration_error(reliability_bins(confidences, correct))

    assert ece == pytest.approx(3 / 4 * 0.05 + 1 / 4 * 0.55)


def test_bins_have_equal_width_and_cover_zero_to_one():
    bins = reliability_bins([0.5], [True])

    assert len(bins) == 10
    assert [calibration_bin.lower for calibration_bin in bins] == pytest.approx(
        [index / 10 for index in range(10)]
    )
    assert [calibration_bin.upper for calibration_bin in bins] == pytest.approx(
        [(index + 1) / 10 for index in range(10)]
    )


def test_a_confidence_on_an_edge_belongs_to_the_lower_bin():
    bins = reliability_bins([0.0, 0.1, 0.3, 0.30001, 1.0], [True] * 5)

    assert [calibration_bin.count for calibration_bin in bins] == [2, 0, 1, 1, 0, 0, 0, 0, 0, 1]


def test_bin_statistics():
    bins = reliability_bins([0.62, 0.68, 0.65, 0.2], [True, False, True, False], num_bins=5)

    fourth = bins[3]
    assert fourth.count == 3
    assert fourth.mean_confidence == pytest.approx(0.65)
    assert fourth.accuracy == pytest.approx(2 / 3)
    assert bins[0].count == 1
    assert bins[4].count == 0
    assert bins[4].mean_confidence == 0.0
    assert bins[4].accuracy == 0.0


def test_rejects_different_lengths():
    with pytest.raises(ValueError, match="same length"):
        reliability_bins([0.5, 0.6], [True])


def test_rejects_empty_input():
    with pytest.raises(ValueError, match="without any predictions"):
        reliability_bins([], [])


def test_ece_rejects_bins_without_predictions():
    bins = reliability_bins([0.5], [True])
    empty_bins = [bins[0]]

    with pytest.raises(ValueError, match="without any predictions"):
        expected_calibration_error(empty_bins)


@pytest.mark.parametrize("confidence", [-0.1, 1.5, float("nan")])
def test_rejects_confidences_outside_zero_to_one(confidence: float):
    with pytest.raises(ValueError, match="between 0 and 1"):
        reliability_bins([0.5, confidence], [True, False])


def test_rejects_zero_bins():
    with pytest.raises(ValueError, match="num_bins"):
        reliability_bins([0.5], [True], num_bins=0)
