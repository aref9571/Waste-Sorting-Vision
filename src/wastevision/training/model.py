from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, MobileNetV3, mobilenet_v3_small

from wastevision.labels import CLASS_NAMES


def build_model(num_classes: int = len(CLASS_NAMES), pretrained: bool = True) -> MobileNetV3:
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: MobileNetV3) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = False
