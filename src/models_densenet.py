"""
DenseNet-121 classifier builder, matching the project spec:
ImageNet-pretrained backbone + Dropout(0.30) + Linear(1024, num_classes).
"""
import torch.nn as nn
from torchvision.models import DenseNet121_Weights, densenet121

from config import CLF_DROPOUT, NUM_CLASSES


def build_densenet121(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = densenet121(weights=weights)
    model.classifier = nn.Sequential(
        nn.Dropout(p=CLF_DROPOUT),
        nn.Linear(model.classifier.in_features, num_classes),
    )
    return model


def freeze_backbone(model):
    """Stage A: freeze everything except the new classifier head."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")
    return model


def unfreeze_final_block(model):
    """
    Stage B: unfreeze the final dense block + transition layer + classifier.
    torchvision's DenseNet121 features are named:
    denseblock1..4, transition1..3, norm5. We unfreeze denseblock4 + norm5 + classifier.
    """
    for name, param in model.named_parameters():
        if (
            name.startswith("classifier")
            or "denseblock4" in name
            or "norm5" in name
        ):
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


def unfreeze_all(model):
    """Sensitivity-run only: unfreeze the entire backbone."""
    for param in model.parameters():
        param.requires_grad = True
    return model


def get_param_groups(model, head_lr, backbone_lr):
    """Discriminative learning rates: classifier head vs. rest of backbone."""
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (head_params if name.startswith("classifier") else backbone_params).append(param)
    groups = []
    if head_params:
        groups.append({"params": head_params, "lr": head_lr})
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return groups
