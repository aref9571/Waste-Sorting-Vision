import numpy as np
import pytest
from PIL import Image

pytest.importorskip("torch")

from wastevision.data.transforms import evaluation_transforms
from wastevision.inference.preprocess import preprocess_image

SIZES = [(512, 384), (384, 512), (300, 300), (257, 256), (1000, 333), (2000, 1500), (224, 224)]


@pytest.mark.parametrize(("width", "height"), SIZES)
def test_numpy_preprocessing_matches_the_torchvision_evaluation_transform(width: int, height: int):
    pixels = np.random.default_rng(width + height).integers(0, 256, (height, width, 3), np.uint8)
    image = Image.fromarray(pixels)

    expected = evaluation_transforms()(image).numpy()

    assert np.abs(preprocess_image(image) - expected).max() < 1e-5


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_other_modes_match_after_conversion_to_rgb(mode: str):
    pixels = np.random.default_rng(1).integers(0, 256, (90, 120, 3), np.uint8)
    image = Image.fromarray(pixels).convert(mode)

    expected = evaluation_transforms()(image.convert("RGB")).numpy()

    assert np.abs(preprocess_image(image) - expected).max() < 1e-5


def test_predictor_agrees_with_pytorch_on_the_same_image(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxscript")
    from wastevision.config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, RESIZE_SIZE
    from wastevision.inference.predictor import Predictor
    from wastevision.labels import CLASS_NAMES
    from wastevision.training.export import export_onnx
    from wastevision.training.model import build_model

    torch.manual_seed(0)
    model = build_model(pretrained=False).eval()
    model_path = tmp_path / "model.onnx"
    export_onnx(model, model_path)
    metadata = {
        "class_names": list(CLASS_NAMES),
        "input": {
            "name": "image",
            "resize": RESIZE_SIZE,
            "crop": IMAGE_SIZE,
            "mean": list(IMAGENET_MEAN),
            "std": list(IMAGENET_STD),
        },
        "output": {"name": "logits"},
    }
    predictor = Predictor(model_path, metadata, threshold=0.5)
    pixels = np.random.default_rng(5).integers(0, 256, (300, 400, 3), np.uint8)
    image = Image.fromarray(pixels)

    with torch.no_grad():
        expected = torch.softmax(model(evaluation_transforms()(image)[None]), dim=1)[0].numpy()

    assert np.abs(predictor.probabilities(image) - expected).max() < 1e-4
    assert predictor.predict(image).label == CLASS_NAMES[int(expected.argmax())]
