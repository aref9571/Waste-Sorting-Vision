import pytest

from wastevision.labels import CATEGORY_BY_CLASS, CLASS_NAMES, category_for


def test_there_are_six_classes():
    assert len(CLASS_NAMES) == 6


def test_class_names_are_unique():
    assert len(set(CLASS_NAMES)) == len(CLASS_NAMES)


def test_class_names_are_in_alphabetical_order():
    assert list(CLASS_NAMES) == sorted(CLASS_NAMES)


def test_every_class_has_a_category():
    assert set(CATEGORY_BY_CLASS) == set(CLASS_NAMES)


@pytest.mark.parametrize(
    ("class_name", "category"),
    [
        ("cardboard", "Cardboard"),
        ("glass", "Glass"),
        ("metal", "Metal and cans"),
        ("paper", "Paper"),
        ("plastic", "Plastic"),
        ("trash", "Other waste"),
    ],
)
def test_category_for_known_class(class_name, category):
    assert category_for(class_name) == category


def test_category_for_unknown_class_raises():
    with pytest.raises(ValueError, match="Unknown class name"):
        category_for("wood")
