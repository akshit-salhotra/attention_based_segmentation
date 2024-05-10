import torch.nn as nn
import torch.nn.functional as F
import torch

def attention(x,in_ch=1,mid_dim=16,out_ch=1):
    attendconv1=nn.Conv2d(in_ch,mid_dim,1)
    ReLU=nn.ReLU(inplace=True)
    attendconv2=nn.Conv2d(mid_dim,out_ch,1)
    at=F.sigmoid(attendconv2(ReLU(attendconv1(x))))
    return at


def attention_local(x,in_ch=1,mid_dim=4,out_ch=1):
    attendconv1=nn.Conv2d(in_ch,mid_dim,3,padding=1)
    ReLU=nn.ReLU(inplace=True)
    attendconv2=nn.Conv2d(mid_dim,out_ch,1)
    at=F.sigmoid(attendconv2(ReLU(attendconv1(x))))
    return at
