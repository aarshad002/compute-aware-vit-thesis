"""ViT model construction via timm."""

import timm
import torch.nn as nn


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Load a timm model and replace the head for num_classes output.

    Args:
        model_name: timm model identifier, e.g. 'deit_tiny_patch16_224'.
        num_classes: number of output classes.
        pretrained: whether to load ImageNet pretrained weights.

    Returns:
        nn.Module ready for fine-tuning.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model
