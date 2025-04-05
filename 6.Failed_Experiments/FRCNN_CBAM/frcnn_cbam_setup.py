
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from cbam import CBAMBlock

def change_the_predictor(model, num_desired_classes=2):
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_desired_classes)
    return model

def change_the_backbone(num_desired_classes=2):
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    cbam_backbone = []
    for layer in backbone:
        if isinstance(layer, nn.Sequential):
            children = list(layer.children())
            if len(children) > 0 and isinstance(children[-1], nn.Conv2d):
                out_channels = children[-1].out_channels
                cbam_backbone.append(CBAMBlock(layer, out_channels))
            else:
                cbam_backbone.append(layer)
        elif isinstance(layer, nn.Conv2d):
            cbam_backbone.append(CBAMBlock(layer, layer.out_channels))
        else:
            cbam_backbone.append(layer)
    backbone = nn.Sequential(*cbam_backbone)
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))
    model = FasterRCNN(backbone,
                       num_classes=num_desired_classes,
                       rpn_anchor_generator=anchor_generator)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_desired_classes)
    return model
