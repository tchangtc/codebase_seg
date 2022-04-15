import torch
import torch.nn as nn
from .trans_layers import LayerNorm

class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation function
    normalization includes BN and IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, 
        groups=1, dilation=1, bias=False, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super(ConvNormAct, self).__init__()

        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact
    
    def forward(self ,x):
        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))
        
        return out

class SingleConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, padding=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=False):
        super(SingleConv, self).__init__()
        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i // 2 for i in kernel_size]

        self.conv = ConvNormAct(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

    def forward(self, x):
        return self.conv(x)

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super(BasicBlock, self).__init__()

        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SELU, True, False]

        pad_size = [i // 2 for i in kernel_size]

        self.conv1 = ConvNormAct(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(in_ch=out_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(x)
        out += self.shortcut(residual)
        return out

class Bottleneck(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=[3,3,3], stride=1, groups=1, dialation=1, norm=nn.BatchNorm3d, act=nn.ReLU, preact=True):
        super(Bottleneck, self).__init__()

        assert norm in [nn.BatchNorm3d, nn.InstanceNorm3d, LayerNorm, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        pad_size = [i // 2 for i in kernel_size]

        self.expansion = 2
        self.conv1 = ConvNormAct(in_ch, out_ch//self.expansion, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch//self.expansion, out_ch//self.expansion, kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, groups=groups, dialation=dialation, preact=preact)
        self.conv3 = ConvNormAct(out_ch//self.expansion, out_ch, 1, stride=1, padding=0, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=pad_size, norm=norm, act=act, preact=preact)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += self.shortcut(residual)
        return out

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, bias=False):
        super(DepthwiseSeparableConv, self).__init__()

        if isinstance(kernel_size, list):
            padding = [i // 2 for i in kernel_size]
        else:
            padding = kernel_size // 2
        
        self.depthwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch,
            bias=bias
        )

        self.pointwise = nn.Conv3d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=bias
        )

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out