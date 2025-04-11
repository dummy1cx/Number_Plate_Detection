import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models import resnet50
from attention import ChannelAttention, SpatialAttention

## -----------------------------------------------------------------
## tried to integrate attention with Resnet backbone for FRCNN 
## -----------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.base_layers = nn.Sequential(*list(base_model.children())[:-2])
        self.ca = ChannelAttention(2048)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.base_layers(x)
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

def build_frcnn_with_attention(num_classes):
    backbone = AttentionResNetBackbone()
    backbone.out_channels = 2048
    model = FasterRCNN(backbone, num_classes=num_classes)
    return model.to(DEVICE)

def print_model_size(model):
    total_pars = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_pars}")

def collate_fn(batch):
    from torchvision.transforms import ToTensor
    images, targets = tuple(zip(*batch))
    images = [ToTensor()(img) for img in images]
    return images, targets
