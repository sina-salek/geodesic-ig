import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    convnext_base,
    ConvNeXt_Base_Weights,
    efficientnet_v2_l,
    EfficientNet_V2_L_Weights,
)
from torchvision.models import (
    mobilenet_v3_small,
    MobileNet_V3_Small_Weights,
    shufflenet_v2_x0_5,
    ShuffleNet_V2_X0_5_Weights,
)
import timm
from experiments.voc.constants import VALID_BACKBONE_NAMES


class VocClassifier(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(VocClassifier, self).__init__()
        if backbone_name not in VALID_BACKBONE_NAMES:
            raise ValueError(
                f"Invalid backbone name: {backbone_name}. Valid names are: {VALID_BACKBONE_NAMES}"
            )

        # Use latest pretrained weights
        if backbone_name == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, 20)
        elif backbone_name == "convnext_base":
            self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.DEFAULT)
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(num_features, 20)
        elif backbone_name == "mobilenet_v3_small":
            self.backbone = mobilenet_v3_small(
                weights=MobileNet_V3_Small_Weights.DEFAULT
            )
            self.backbone = modify_mobilenet_activations(self.backbone)
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(num_features, 20)
        elif backbone_name == "shufflenet_v2_x0_5":
            self.backbone = shufflenet_v2_x0_5(
                weights=ShuffleNet_V2_X0_5_Weights.DEFAULT
            )
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, 20)
        elif backbone_name == "resnet10t.c3_in1k":  # Tiny ResNet10
            self.backbone = timm.create_model("resnet10t.c3_in1k", pretrained=True)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, 20)
        elif backbone_name == "efficientnet_v2_l":
            self.backbone = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.DEFAULT)
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(num_features, 20)

    def forward(self, x):
        x = self.backbone(x)
        return F.log_softmax(x, dim=-1)


def modify_mobilenet_activations(model):
    """Replace hardsigmoid/hardswish with differentiable alternatives in MobileNetV3"""

    def _replace_activation(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Hardswish):
                setattr(module, name, nn.ReLU())
            elif isinstance(child, nn.Hardsigmoid):
                setattr(module, name, nn.Sigmoid())
            else:
                _replace_activation(child)

    _replace_activation(model)
    return model
