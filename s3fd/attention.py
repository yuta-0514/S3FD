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


class MultiAttentionNetwork(nn.Module):
    def __init__(self, channel, num_masks=4):
        super(MultiAttentionNetwork, self).__init__()
        
        self.channel =channel
        self.attn_conv = nn.Conv2d(self.channel, num_masks, 1, bias=False)

        nn.init.xavier_uniform_(self.attn_conv.weight)

        self.mask_ = None
        self.num_masks = num_masks

    def forward(self, x):
        b, n, h, w = x.shape
        attn = torch.sigmoid(self.attn_conv(x))  # [B, M, H, W]
        B, _, H, W = attn.shape
        self.mask_ = attn
        
        x = x.reshape(B, 1, self.channel, H, W)
        attn = attn.reshape(B, self.num_masks, 1, H, W)
        
        x = x * attn  # [B, M, 512, H, W]

        x = x.reshape(B * self.num_masks, -1, H, W)  # [BM, 512, H, W]
        
        y = torch.zeros(b, n, h, w )
        for i in range(B):
            y[i] = torch.sum(x[i*4+1:(i+1)*4], dim =0)
        
        return y
    
    def divergence_loss(self):
        mask = self.mask_  # [B, M, H, W]
        B, M, H, W = mask.shape
        device = mask.device
        
        flatten_mask = mask.reshape(B, M, -1)
        diag = 1 - torch.eye(M).unsqueeze(0).to(device)  # [1, M, M]
        
        max_val, _ = flatten_mask.max(dim=2, keepdim=True)
        flatten_mask = flatten_mask / (max_val + 1e-2)
        
        div_loss = torch.bmm(flatten_mask, flatten_mask.transpose(1, 2)) * diag  # [B, M, M] x [1, M, M]
        return (div_loss.view(-1) ** 2).mean()


class SKAttention(nn.Module):
    def __init__(self, channel=512,kernels=[1,3,5,7],reduction=16,group=1,L=32):
        super(SKAttention, self).__init__()
        self.d=max(L,channel//reduction)
        self.convs=nn.ModuleList([])
        for k in kernels:
            self.convs.append(
                nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(channel,channel,kernel_size=k,padding=k//2,groups=group)),
                    ('bn',nn.BatchNorm2d(channel)),
                    ('relu',nn.ReLU())
                ]))
            )
        self.fc=nn.Linear(channel,self.d)
        self.fcs=nn.ModuleList([])
        for i in range(len(kernels)):
            self.fcs.append(nn.Linear(self.d,channel))
        self.softmax=nn.Softmax(dim=0)

    def forward(self, x):
        bs, c, _, _ = x.size()
        conv_outs=[]

        ### split
        for conv in self.convs:
            print(conv)
            conv_outs.append(conv(x))
        feats=torch.stack(conv_outs,0)#k,bs,channel,h,w

        ### fuse
        U=sum(conv_outs) #bs,c,h,w

        ### reduction channel
        S=U.mean(-1).mean(-1) #bs,c
        Z=self.fc(S) #bs,d

        ### calculate attention weight
        weights=[]
        for fc in self.fcs:
            weight=fc(Z)
            weights.append(weight.view(bs,c,1,1)) #bs,channel
        attention_weughts=torch.stack(weights,0)#k,bs,channel,1,1
        attention_weughts=self.softmax(attention_weughts)#k,bs,channel,1,1

        ### fuse
        V=(attention_weughts*feats).sum(0)
        return V


x = torch.rand(1,256,160,160)
Attention = SKAttention(channel=256)
y = Attention(x)
print(y.shape)