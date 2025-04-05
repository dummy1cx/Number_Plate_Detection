
import torch
import torch.nn as nn

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        scale = self.sigmoid(avg_out + max_out)
        x = x * scale
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.spatial_conv(spatial_att))
        x = x * scale
        return x

class CBAMBlock(nn.Module):
    def __init__(self, module, channels):
        super(CBAMBlock, self).__init__()
        self.module = module
        self.cbam = CBAM(channels)

    def forward(self, x):
        x = self.module(x)
        x = self.cbam(x)
        return x
