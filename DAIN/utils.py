import math
import torch
import torch.nn as nn
import torch.nn.init as weight_init


def conv3x3(in_planes, out_planes, dilation=1, stride=1):
    "3x3 convolution with padding"

    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=int(dilation * (3 - 1) / 2),
        dilation=dilation,
        bias=False,
    )


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):

        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, dilation, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, dilation=1, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

        self.conv2 = conv3x3(planes, planes, dilation, stride)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.1),
    )


def predict_flow(in_planes):
    return nn.Conv2d(in_planes, 2, kernel_size=3, stride=1, padding=1, bias=True)


def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(
        in_planes, out_planes, kernel_size, stride, padding, bias=True
    )


import time


def block_1(
    in_planes,
    out_planes,
    kernel_size=3,
    stride1=2,
    stride2=1,
    stride3=1,
    padding=1,
    dilation=1,
):

    return nn.Sequential(
        conv(in_planes, out_planes, kernel_size, stride1, padding, dilation),
        conv(out_planes, out_planes, kernel_size, stride2, padding, dilation),
        conv(out_planes, out_planes, kernel_size, stride3, padding, dilation),
    )


class block_2(nn.Module):
    def __init__(self, od, dd):

        super().__init__()

        self.conv0 = conv(od, 128, kernel_size=3, stride=1)
        self.conv1 = conv(od + dd[0], 128, kernel_size=3, stride=1)
        self.conv2 = conv(od + dd[1], 96, kernel_size=3, stride=1)
        self.conv3 = conv(od + dd[2], 64, kernel_size=3, stride=1)
        self.conv4 = conv(od + dd[3], 32, kernel_size=3, stride=1)
        self.predict_flow = predict_flow(od + dd[4])

    def forward(self, x):

        x = torch.cat((self.conv0(x), x), 1)
        x = torch.cat((self.conv1(x), x), 1)
        x = torch.cat((self.conv2(x), x), 1)
        x = torch.cat((self.conv3(x), x), 1)
        x = torch.cat((self.conv4(x), x), 1)

        return x

