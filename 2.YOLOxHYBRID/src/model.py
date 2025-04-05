import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

## ----------------------------------------------------------------------------
## The codes are referenced from the official research papers of CBAM author
## Multiple github repos are referenced for this architure
## I do not claim that this code is written by me 
## The codes are referenced only for educational purpose
## All the referenced sources are outlined in the written report
## ----------------------------------------------------------------------------

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        # Spatial attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        # Combining average and max pool paths
        scale = self.sigmoid(avg_out + max_out)
        # Channel attention scaling 
        x = x * scale
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.spatial_conv(spatial_att))
        x = x * scale
        return x


class YOLOv4DenseNetCBAM(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv4DenseNetCBAM, self).__init__()
        base_model = models.densenet121(pretrained=False)
        self.initial = nn.Sequential(
            base_model.features.conv0,   
            base_model.features.norm0,   
            base_model.features.relu0,   
            base_model.features.pool0    
        )
        self.block1 = base_model.features.denseblock1      
        self.trans1 = base_model.features.transition1      
        self.block2 = base_model.features.denseblock2      
        self.trans2 = base_model.features.transition2      
        self.block3 = base_model.features.denseblock3      
        self.trans3 = base_model.features.transition3      
        self.block4 = base_model.features.denseblock4      
        self.norm5  = base_model.features.norm5            
        del base_model

        self.cbam2 = CBAM(channels=512)    
        self.cbam3 = CBAM(channels=1024)   
        self.cbam4 = CBAM(channels=1024)   

        self.num_classes = num_classes
        self.num_anchors = 3  

        self.anchor_boxes = {
            'large':  [(10, 13), (16, 30), (33, 23)],    
            'medium': [(30, 61), (62, 45), (59, 119)],  
            'small':  [(116, 90), (156, 198), (373, 326)]
        }

        self.anchor_masks = {
            'small': [0, 1, 2],
            'medium': [3, 4, 5],
            'large': [6, 7, 8]
        }

        self.conv_route_small = nn.Conv2d(1024, 512, kernel_size=1)
        self.conv_pred_small  = nn.Conv2d(512, self.num_anchors * (5 + self.num_classes), kernel_size=1)
        self.conv_route_medium = nn.Conv2d(1536, 256, kernel_size=1)  
        self.conv_pred_medium  = nn.Conv2d(256, self.num_anchors * (5 + self.num_classes), kernel_size=1)
        self.conv_route_large = nn.Conv2d(768, 128, kernel_size=1)  
        self.conv_pred_large  = nn.Conv2d(128, self.num_anchors * (5 + self.num_classes), kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.initial(x)         
        x = self.block1(x)          
        x = self.trans1(x)          
        x = self.block2(x)          
        out_large = x               
        out_large = self.cbam2(out_large)   
        x = self.trans2(x)          
        x = self.block3(x)          
        out_medium = x              
        out_medium = self.cbam3(out_medium) 
        x = self.trans3(x)          
        x = self.block4(x)          
        x = self.norm5(x)           
        x = F.relu(x, inplace=True) 
        out_small = x               
        out_small = self.cbam4(out_small)
        route_small = self.conv_route_small(out_small) 
        pred_small  = self.conv_pred_small(route_small) 
        up_small   = self.upsample(route_small)                  
        fusion_mid = torch.cat([out_medium, up_small], dim=1)    
        route_medium = self.conv_route_medium(fusion_mid)
        pred_medium  = self.conv_pred_medium(route_medium) 
        up_medium  = self.upsample(route_medium)                 
        fusion_large = torch.cat([out_large, up_medium], dim=1)   
        route_large = self.conv_route_large(fusion_large)   
        pred_large  = self.conv_pred_large(route_large)     
        return [pred_small, pred_medium, pred_large]


if __name__ == "__main__":
    model = YOLOv4DenseNetCBAM(num_classes=80)  ## Testing purpose 
    model.eval()                                ## Output dimension shape
    dummy_input = torch.randn(1, 3, 416, 416)
    outputs = model(dummy_input)
    for name, output in zip(["Small (13x13)", "Medium (26x26)", "Large (52x52)"], outputs):
        print(f"{name} output shape: {output.shape}")
