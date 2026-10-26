import pytest

torch = pytest.importorskip("torch")

from torch import nn

from wastevision.labels import CLASS_NAMES
from wastevision.training.model import build_model, freeze_backbone


def test_build_model_outputs_one_score_per_class():
    model = build_model(pretrained=False).eval()

    with torch.no_grad():
        logits = model(torch.zeros(2, 3, 64, 64))

    assert logits.shape == (2, len(CLASS_NAMES))


def test_build_model_replaces_only_the_last_layer():
    model = build_model(num_classes=4, pretrained=False)

    last_layer = model.classifier[-1]
    assert isinstance(last_layer, nn.Linear)
    assert last_layer.out_features == 4
    assert last_layer.in_features == model.classifier[0].out_features


def test_freeze_backbone_leaves_only_the_classifier_trainable():
    model = build_model(pretrained=False)

    freeze_backbone(model)

    assert not any(parameter.requires_grad for parameter in model.features.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())
