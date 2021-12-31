import numpy as np
import torch
from torch import nn
from torch.nn import init

from ECA import ECAAttention #Channel Attention
from EPSA import PSA #Channel Attention
from SGE import SpatialGroupEnhance  #Spatial Attention


# BAMベース
class Attention_1(nn.Module):

    def __init__(self, channel):
        super().__init__()
        self.ca=ECAAttention(kernel_size=3)
        self.sa=SpatialGroupEnhance(groups=8)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        sa_out=self.sa(x)
        ca_out=self.ca(x)
        weight=self.sigmoid(sa_out+ca_out)
        out=(1+weight)*x
        return out

# CBAMベース
class Attention_2(nn.Module):

    def __init__(self,channel):
        super().__init__()
        self.ca=PSA(channel=channel)
        self.sa=SpatialGroupEnhance(groups=8)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual


from CBAM import ChannelAttention,SpatialAttention
# shuffleベース
class Attention_3(nn.Module):

    def __init__(self, channel,reduction=16,G=8):
        super().__init__()
        self.G=G
        self.channel=channel
        self.ca=ChannelAttention(channel=channel//self.G,reduction=reduction)
        self.sa=SpatialAttention()

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape
        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x):
        b, c, h, w = x.size()

        #group into subfeatures
        x=x.view(b*self.G,-1,h,w) #bs*G,c//G,h,w
        residual=x

        out=x*self.ca(x)
        out=out*self.sa(out)
        out=out+residual

        # concatenate along channel axis
        out=out.contiguous().view(b,-1,h,w)

        # channel shuffle
        out = self.channel_shuffle(out, 2)
        return out


input=torch.randn(1,256,160,160)
se = Attention_3(channel=256)
output=se(input)
print(output.shape)

