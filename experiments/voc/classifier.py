import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights, convnext_base, ConvNeXt_Base_Weights
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from experiments.voc.constants import VALID_BACKBONE_NAMES

class VocClassifier(nn.Module):
    def __init__(self, backbone_name="resnet18"):
        super(VocClassifier, self).__init__()
        if backbone_name not in VALID_BACKBONE_NAMES:
            raise ValueError(f"Invalid backbone name: {backbone_name}. Valid names are: {VALID_BACKBONE_NAMES}")
            
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
            self.backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
            num_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(num_features, 20)
    
    def forward(self, x):
        x = self.backbone(x)
        return F.log_softmax(x, dim=-1)