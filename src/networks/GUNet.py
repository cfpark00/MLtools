import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

convs = {
    2:nn.Conv2d,
    3:nn.Conv3d
}
upconvs = {
    2:nn.ConvTranspose2d,
    3:nn.ConvTranspose3d
}
avgpools = {
    2:nn.AvgPool2d,
    3:nn.AvgPool3d
}
maxpools = {
    2:nn.MaxPool2d,
    3:nn.MaxPool3d
}


class CNA(nn.Module):
    def __init__(self, in_channels, out_channels,n_groups=16,act="GELU",dim=2):
        super().__init__()
        assert out_channels%n_groups==0, "out_channels must be divisible by n_groups"
        assert act in ["GELU","SiLU","ReLU","LeakyReLU"], "act must be GELU, SiLU, ReLU or LeakyReLU"
        self.conv=convs[dim](in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.gnorm=nn.GroupNorm(n_groups, out_channels)
        if act=="GELU":
            self.act=nn.GELU()
        elif act=="SiLU":
            self.act=nn.SiLU()
        elif act=="ReLU":
            self.act=nn.ReLU()
        elif act=="LeakyReLU":
            self.act=nn.LeakyReLU()
        else:
            raise ValueError("act must be GELU, ReLU or LeakyReLU")

    def forward(self, x):
        return self.act(self.gnorm(self.conv(x)))

class NCNA(nn.Module):
    def __init__(self, in_channels, out_channels,N,skip=False,dim=2):
        """
        in_channels: number of input channels
        out_channels: number of output channels(= middle channels)
        N: number of CNA layers
        skip: if True, add first output to output
        """
        super().__init__()
        assert N>1
        self.skip=skip
        channels=[]
        channels.append(in_channels)
        for i in range(N):
            channels.append(out_channels)#len(channels) ==  N+1
        
        self.layers=nn.ModuleList()
        for i in range(N):
            self.layers.append(CNA(channels[i],channels[i+1],dim=dim))
            
    def forward(self, x):
        for i,layer in enumerate(self.layers):
            if i==0:
                x=layer(x)
                x1=x
            else:
                x=layer(x)
        if self.skip:
            x=x+x1
        return x

class DownNCNA(nn.Module):
    """Downscaling with pool then NCNA"""
    def __init__(self, in_channels, out_channels,N,pool="avg",skip=False,dim=2):
        super().__init__()
        assert pool in ["avg","max"], "pool must be avg or max"
        if pool=="avg":
            self.pool=avgpools[dim](2)
        elif pool=="max":
            self.pool=maxpools[dim](2)
        else:
            raise ValueError("pool must be avg or max")
        self.ncna=NCNA(in_channels, out_channels,N=N,skip=skip,dim=dim)

    def forward(self, x):
        return self.ncna(self.pool(x))

class UpNCNA(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels,N,skip=False,dim=2):
        super().__init__()
        self.dim=dim
        self.upconv = upconvs[dim](in_channels, in_channels//2, kernel_size=2, stride=2)
        self.ncna = NCNA(in_channels, out_channels,N=N,skip=skip,dim=dim)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        if self.dim==2:
            # input is CHW
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])
        else:
            # input is CHWD
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            diffZ = x2.size()[4] - x1.size()[4]

            x1 = F.pad(x1, [diffZ // 2, diffZ - diffZ // 2,
                            diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.ncna(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size=1,dim=2):
        super(OutConv, self).__init__()
        padding=kernel_size//2
        self.conv=convs[dim](in_channels, out_channels, kernel_size=kernel_size,padding=padding)

    def forward(self, x):
        return self.conv(x)

class GUNet(nn.Module):
    def __init__(self, n_channels, n_classes, N=2,width=64,skip=False,catorig=False,dim=2):
        super(GUNet, self).__init__()
        assert dim in [2,3], "dim must be 2 or 3"
        self.dim=dim
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.catorig=catorig

        self.inc = NCNA(self.n_channels, width,N=N,skip=skip,dim=dim)
        self.down1 = DownNCNA(width, 2*width,N=N,skip=skip,dim=dim)
        self.down2 = DownNCNA(2*width, 4*width,N=N,skip=skip,dim=dim)
        self.down3 = DownNCNA(4*width, 8*width,N=N,skip=skip,dim=dim)
        self.down4 = DownNCNA(8*width, 16*width,N=N,skip=skip,dim=dim)
        self.up1 = UpNCNA(16*width, 8*width,N=N,skip=skip,dim=dim)
        self.up2 = UpNCNA(8*width, 4*width,N=N,skip=skip,dim=dim)
        self.up3 = UpNCNA(4*width, 2*width,N=N,skip=skip,dim=dim)
        self.up4 = UpNCNA(2*width, width,N=N,skip=skip,dim=dim)
        if self.catorig:
            self.outc = OutConv(width+self.n_channels, self.n_classes,kernel_size=3,dim=dim)
        else:
            self.outc = OutConv(width, self.n_classes,kernel_size=3,dim=dim)

    def forward(self, x):
        orig=x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.catorig:
            logits=self.outc(torch.cat([x,orig],axis=1))
        else:
            logits=self.outc(x)
        return logits

class FCVAE(nn.Module):
    """
    Fully Convolutional VAE
    """