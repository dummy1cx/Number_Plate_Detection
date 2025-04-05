# model.py
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##  -----------------------------------------------------------------
##  pretrained resnet50 was used for Faster RCNN base line training 
##  Objectve is to improve the perfomance in next architecture
##  Attention modules to be implemented in further experiments
##  This code is referneced from Pytorch - torchvision documentations
##  -------------------------------------------------------------------


def build_resnet50_frcnn(num_classes):    
    backbone = resnet_fpn_backbone(
        backbone_name="resnet50",
        weights=torchvision.models.ResNet50_Weights.DEFAULT,
        trainable_layers=3
    )
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)
    return model.to(DEVICE)


def print_model_size(model):
    total_pars = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_pars}")
    return


def collate_fn(batch):
    from torchvision.transforms import ToTensor
    images, targets = tuple(zip(*batch))
    images = [ToTensor()(img) for img in images]
    return images, targets
