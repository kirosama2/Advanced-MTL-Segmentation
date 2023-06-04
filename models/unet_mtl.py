
""" Full assembly of the parts to form the complete network """

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.conv2d_mtl import Conv2dMtl, ConvTranspose2dMtl


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


def conv3x3mtl(in_planes, out_planes, stride=1):
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class DoubleConvMtl(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = conv3x3mtl(in_channels, mid_channels)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3mtl(mid_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(mid_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, mtl):
        super().__init__()
        if mtl:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConvMtl(in_channels, out_channels)
            )
        else:
            self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(2),
                DoubleConv(in_channels, out_channels)
            )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mtl):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        # if bilinear:
        #     self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #     self.conv = DoubleConv(in_channels, out_channels // 2, in_channels // 2)
        if mtl:
            self.up = ConvTranspose2dMtl(in_channels , in_channels // 2, 2, 2)
            self.conv = DoubleConvMtl(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
