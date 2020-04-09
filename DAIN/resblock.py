import math

import torch
import torch.nn as nn
import torch.nn.init as weight_init

from utils import BasicBlock


class MultipleBasicBlock(nn.Module):
    def __init__(self, input_feature, block=BasicBlock, intermediate_feature=64):
        super(MultipleBasicBlock, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(
                input_feature,
                intermediate_feature,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=True,
            ),
            nn.ReLU(inplace=True),
        )

        self.block2 = block(intermediate_feature, intermediate_feature, dilation=1)
        self.block3 = block(intermediate_feature, intermediate_feature, dilation=1)
        self.block4 = block(intermediate_feature, intermediate_feature, dilation=1)

        self.block5 = nn.Conv2d(intermediate_feature, 3, (3, 3), 1, (1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


if __name__ == "__main__":

    model = MultipleBasicBlock(200, BasicBlock)
    print(model)
