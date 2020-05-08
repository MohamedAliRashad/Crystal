import torch
import torch.nn as nn
from torch.autograd import Variable
from Correlation import CorrelationModule
import numpy as np
from utils import conv, predict_flow, deconv, block_1, block_2

    
class PWCDCNet(nn.Module):
    def __init__(self, md=4):

        """
        input: md --- maximum displacement (for correlation. default: 4), after warpping
        """

        super(PWCDCNet, self).__init__()
        self.c1 = block_1(3, 16, kernel_size=3)

        self.c2 = block_1(16, 32, kernel_size=3)
        self.c3 = block_1(32, 64, kernel_size=3)
        self.c4 = block_1(64, 96, kernel_size=3)
        self.c5 = block_1(96, 128, kernel_size=3)
        self.c6 = block_1(128, 196, kernel_size=3)

        self.corr = CorrelationModule(
            pad_size=md,
            kernel_size=1,
            max_displacement=md,
            stride1=1,
            stride2=1,
            corr_multiply=1,
        )
        self.leakyRELU = nn.LeakyReLU(0.1)

        self.deconv = deconv(2, 2, kernel_size=4, stride=2, padding=1)

        nd = (2 * md + 1) ** 2  # md =4 -> nd =81
        dd = np.cumsum([128, 128, 96, 64, 32], dtype=np.int32).astype(np.int)
        dd = [int(d) for d in dd]

        od = nd
        self.conv6 = block_2(od, dd)
        self.upfeat6 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        self.predict_flow6 = predict_flow(od + dd[4])

        od = nd + 128 + 4
        self.conv5 = block_2(od, dd)
        self.upfeat5 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        self.predict_flow5 = predict_flow(od + dd[4])

        od = nd + 96 + 4
        self.conv4 = block_2(od, dd)
        self.upfeat4 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        self.predict_flow4 = predict_flow(od + dd[4])

        od = nd + 64 + 4
        self.conv3 = block_2(od, dd)
        self.upfeat3 = deconv(od + dd[4], 2, kernel_size=4, stride=2, padding=1)
        self.predict_flow3 = predict_flow(od + dd[4])
        od = nd + 32 + 4
        self.conv2 = block_2(od, dd)
        self.predict_flow2 = predict_flow(od + dd[4])

        self.dc_conv1 = conv(
            od + dd[4], 128, kernel_size=3, stride=1, padding=1, dilation=1
        )
        self.dc_conv2 = conv(128, 128, kernel_size=3, stride=1, padding=2, dilation=2)
        self.dc_conv3 = conv(128, 128, kernel_size=3, stride=1, padding=4, dilation=4)
        self.dc_conv4 = conv(128, 96, kernel_size=3, stride=1, padding=8, dilation=8)
        self.dc_conv5 = conv(96, 64, kernel_size=3, stride=1, padding=16, dilation=16)
        self.dc_conv6 = conv(64, 32, kernel_size=3, stride=1, padding=1, dilation=1)
        self.dc_conv7 = predict_flow(32)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in")
                if m.bias is not None:
                    m.bias.data.zero_()

        W_MAX = 2048
        H_MAX = 1024
        B_MAX = 3
        xx = torch.arange(0, W_MAX).view(1, -1).cuda().repeat(H_MAX, 1)
        yy = torch.arange(0, H_MAX).view(-1, 1).cuda().repeat(1, W_MAX)
        xx = xx.view(1, 1, H_MAX, W_MAX).repeat(B_MAX, 1, 1, 1)
        yy = yy.view(1, 1, H_MAX, W_MAX).repeat(B_MAX, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        ## for saving time on allocating a grid in forward
        self.W_MAX = W_MAX
        self.H_MAX = H_MAX
        self.B_MAX = B_MAX
        self.grid = Variable(grid, requires_grad=False)

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        assert B <= self.B_MAX and H <= self.H_MAX and W <= self.W_MAX
        vgrid = self.grid[:B, :, :H, :W] + flo

        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        output = nn.functional.grid_sample(x, vgrid)
        mask = torch.autograd.Variable(
            torch.cuda.FloatTensor().resize_(x.size()).zero_() + 1, requires_grad=False
        )
        mask = nn.functional.grid_sample(mask, vgrid)
        mask[mask < 0.9999] = 0
        mask[mask > 0] = 1

        return output * mask

    def forward(self, x, output_more=False):
        im1 = x[:, :3, :, :]
        im2 = x[:, 3:, :, :]
        # image 1
        c11 = self.c1(im1)
        c12 = self.c2(c11)
        c13 = self.c3(c12)
        c14 = self.c4(c13)
        c15 = self.c5(c14)
        c16 = self.c6(c15)
        # image 2
        c21 = self.c1(im2)
        c22 = self.c2(c21)
        c23 = self.c3(c22)
        c24 = self.c4(c23)
        c25 = self.c5(c24)
        c26 = self.c6(c25)

        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)

        x = self.conv6(corr6)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)

        x = self.conv5(x)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)

        x = self.conv4(x)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)
        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = self.conv3(x)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = self.conv2(x)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))
        if not output_more:
            return flow2
        else:
            return [flow2, flow3, flow4, flow5, flow6]


def pwc_dc_net(path=None):

    model = PWCDCNet()
    if path is not None:
        data = torch.load(path)
        if "state_dict" in data.keys():
            model.load_state_dict(data["state_dict"])
        else:
            model.load_state_dict(data)
    return model

if __name__ == "__main__":
    p = PWCDCNet()
    # print(p)
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url("http://vllab1.ucmerced.edu/~wenbobao/DAIN/pwc_net.pth.tar")
    p.load_state_dict(state_dict)