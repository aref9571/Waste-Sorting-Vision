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
