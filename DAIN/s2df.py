import math

import torch
import torch.nn as nn

from .utils import BasicBlock


class S2DF(nn.Module):
    def __init__(self, block=BasicBlock, dense=True):
        super(S2DF, self).__init__()
        self.inplanes = 64
        self.dense = dense
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True),
        )

        self.block2 = block(self.inplanes, 64, dilation=4)
        self.block3 = block(self.inplanes, 64, dilation=8)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        y = []

        y.append(x)  # raw feature
        x = self.block1(x)
        y.append(x)

        x = self.block2(x)
        y.append(x)

        x = self.block3(x)
        y.append(x)
        
        return torch.cat(y, dim=1)


if __name__ == "__main__":

    x = torch.randn(2, 3, 224, 448)

    model = S2DF(BasicBlock, False)
    print(model)
    print(model(x).shape)
