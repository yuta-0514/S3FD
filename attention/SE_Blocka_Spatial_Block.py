import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from matplotlib import pyplot as plt

from collections import OrderedDict


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

# Spatial Attention
class AttentionNetwork(nn.Module):
    def __init__(self, channel):
        super(AttentionNetwork, self).__init__()

        self.channel = channel
        self.attn_conv = nn.Sequential(
            nn.Conv2d(self.channel, 1, 1),
            nn.Sigmoid()
        )

        self.mask_ = None

    def forward(self, x):
        attn = self.attn_conv(x)  # [B, 1, H, W]
        B, _, H, W = attn.shape
        self.mask_ = attn.detach()
        x = x * attn
        return x


x = torch.rand(1,256,160,160)
Attention = SEBlock(channel=256)
y = Attention(x)
print(y.shape)