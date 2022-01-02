import numpy as np
import torch
from torch import nn
from torch.nn import init


# 1. PSA+ECA
def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

class ECAWeight(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        self.conv=nn.Conv1d(1,1,kernel_size=kernel_size,padding=(kernel_size-1)//2)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out=out.squeeze(-1).permute(0,2,1) #bs,1,c
        out=self.conv(out) #bs,1,c
        out=self.sigmoid(out) #bs,1,c
        weight=out.permute(0,2,1).unsqueeze(-1) #bs,c,1,1

        return weight

class PSA_ECA(nn.Module):

    def __init__(self, channel, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSA_ECA, self).__init__()
        self.conv_1 = conv(channel, channel//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(channel, channel//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(channel, channel//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(channel, channel//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = ECAWeight(kernel_size=3)
        self.split_channel = channel // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


# 2. PSA＋SGE
class SEWeightModule(nn.Module):

    def __init__(self, channels, reduction=16):
        super(SEWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels//reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels//reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)

        return weight

class PSA_SGE(nn.Module):

    def __init__(self, channel, groups=4):
        super(PSA_SGE, self).__init__()
        self.se = SEWeightModule(channel // 4)
        self.split_channel = channel // 4
        self.softmax = nn.Softmax(dim=1)

        # SGE
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()

        self.cahnnel = channel//self.groups

    def forward(self, x):
        batch_size, c, h,w=x.shape
        x=x.view(batch_size*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(batch_size*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(batch_size,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(batch_size*self.groups,1,h,w) #bs*g,1,h,w
        x=x*self.sig(t)
        x =x.view(batch_size,c,h,w)
        feats = x.view(batch_size, 4, self.split_channel, x.shape[2], x.shape[3])

        x1 = x[:,:self.cahnnel,:,:]
        x2 = x[:,self.cahnnel:self.cahnnel*2,:,:]
        x3 = x[:,self.cahnnel*2:self.cahnnel*3,:,:]
        x4 = x[:,self.cahnnel*3:,:,:]

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)


        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


# 3. PSA+SGE+ECA
class PSA_SGE_ECA(nn.Module):

    def __init__(self, channel, groups=4):
        super(PSA_SGE_ECA, self).__init__()
        self.split_channel = channel // 4
        self.softmax = nn.Softmax(dim=1)

        self.se = ECAWeight(kernel_size=3)

        # SGE
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()

        self.cahnnel = channel//self.groups

    def forward(self, x):
        batch_size, c, h,w=x.shape
        x=x.view(batch_size*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(batch_size*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(batch_size,self.groups,h,w) #bs,g,h*w
        
        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(batch_size*self.groups,1,h,w) #bs*g,1,h,w
        x=x*self.sig(t)
        x =x.view(batch_size,c,h,w)
        feats = x.view(batch_size, 4, self.split_channel, x.shape[2], x.shape[3])

        x1 = x[:,:self.cahnnel,:,:]
        x2 = x[:,self.cahnnel:self.cahnnel*2,:,:]
        x3 = x[:,self.cahnnel*2:self.cahnnel*3,:,:]
        x4 = x[:,self.cahnnel*3:,:,:]

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)


        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


# 4. 3×3Conv -> EPSA

# 5. Suffle Attentionをそのまま利用する

input=torch.randn(1,256,160,160)
psa = PSA_ECA(channel=256)
output=psa(input)
print(output.shape)