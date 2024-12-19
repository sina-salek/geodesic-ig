import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet18, ResNet18_Weights


class VocClassifier(nn.Module):
    def __init__(self):
        super(VocClassifier, self).__init__()
        # Use latest pretrained weights
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # Replace final layer to match VOC classes
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 20)  # 20 VOC classes
        
    def forward(self, x):
        x = self.resnet(x)
        return F.log_softmax(x, dim=-1)