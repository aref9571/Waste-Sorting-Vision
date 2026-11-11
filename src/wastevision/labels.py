CLASS_NAMES: tuple[str, ...] = ("cardboard", "glass", "metal", "paper", "plastic", "trash")

CATEGORY_BY_CLASS: dict[str, str] = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal and cans",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Other waste",
}


def category_for(class_name: str) -> str:
    try:
        return CATEGORY_BY_CLASS[class_name]
    except KeyError:
        raise ValueError(f"Unknown class name: {class_name!r}") from None


SORTING_HINT_BY_CLASS: dict[str, str] = {
    "cardboard": "Flatten boxes and keep them dry. Remove tape and packing material.",
    "glass": "Empty and rinse bottles and jars. Lids and caps are often collected separately.",
    "metal": "Empty and rinse cans and tins before you put them in the recycling.",
    "paper": "Keep paper clean and dry. Greasy or soiled paper usually does not belong here.",
    "plastic": "Empty the container and check the recycling symbol on the item.",
    "trash": "This item probably belongs in general waste. Check whether parts can be separated.",
}


def sorting_hint_for(class_name: str) -> str:
    try:
        return SORTING_HINT_BY_CLASS[class_name]
    except KeyError:
        raise ValueError(f"Unknown class name: {class_name!r}") from None
