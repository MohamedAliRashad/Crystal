from torch.nn import Module
import torch
from torch.autograd import Variable
from torch.autograd import gradcheck

import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

lib = load(
    name="filterinterpolation_cuda",
    sources=["helper/FilterInterpolation/filterinterpolation_cuda.cc", "helper/FilterInterpolation/filterinterpolation_cuda_kernel.cu"],
    verbose=True,
)


class FilterInterpolationLayer(Function):
    def __init__(self):
        super(FilterInterpolationLayer, self).__init__()

    @staticmethod
    def forward(ctx, input1, input2, input3):

        assert input1.is_contiguous()
        assert input2.is_contiguous()
        assert input3.is_contiguous()

        if input1.is_cuda:
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            lib.FilterInterpolationLayer_gpu_forward(input1, input2, input3, output)
        else:
            output = torch.FloatTensor(input1.data.size())
            lib.FilterInterpolationLayer_cpu_forward(input1, input2, input3, output)

        ctx.save_for_backward(input1, input2, input3)
        return output

    @staticmethod
    def backward(ctx, gradoutput):

        input1, input2, input3 = ctx.saved_tensors

        gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
        gradinput2 = torch.cuda.FloatTensor().resize_(input2.size()).zero_()
        gradinput3 = torch.cuda.FloatTensor().resize_(input3.size()).zero_()
        if input1.is_cuda:
            err = lib.FilterInterpolationLayer_gpu_backward(
                input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3
            )
            if err != 0:
                print(err)

        else:
            err = lib.FilterInterpolationLayer_cpu_backward(
                input1, input2, input3, gradoutput, gradinput1, gradinput2, gradinput3
            )
            if err != 0:
                print(err)

        return gradinput1, gradinput2, gradinput3


# calculate the weights of flow
class WeightLayer(Function):
    def __init__(self, lambda_e=10.0 / 255.0, lambda_v=1.0, Nw=3):
        # lambda_e = 10.0 , lambda_v = 1.0,  Nw = 3,
        super(WeightLayer, self).__init__()
        self.lambda_e = lambda_e
        self.lambda_v = lambda_v
        self.Nw = Nw

    # flow1_grad
    def forward(self, input1, input2, input3):

        self.input1 = input1.contiguous()  # ref1 image
        self.input2 = input2.contiguous()  # ref2 image
        self.input3 = input3.contiguous()

        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output = torch.zeros(input1.size(0), 1, input1.size(2), input1.size(3))

        if input1.is_cuda:
            output = output.cuda()
            err = lib.WeightLayer_gpu_forward(
                input1,
                input2,
                input3,
                # flow1_grad,
                output,
                self.lambda_e,
                self.lambda_v,
                self.Nw,
            )
            if err != 0:
                print(err)
        else:
            err = lib.WeightLayer_cpu_forward(
                input1, input2, input3, output, self.lambda_e, self.lambda_v, self.Nw
            )
            if err != 0:
                print(err)

        self.output = output  # save this for fast back propagation
        #  the function returns the output to its caller
        return output

    # TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        gradinput1 = torch.zeros(self.input1.size())
        gradinput2 = torch.zeros(self.input2.size())
        gradinput3 = torch.zeros(self.input3.size())
        if self.input1.is_cuda:
            gradinput1 = gradinput1.cuda(self.device)
            gradinput2 = gradinput2.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)

            err = lib.WeightLayer_gpu_backward(
                self.input1,
                self.input2,
                self.input3,
                self.output,
                gradoutput,
                gradinput1,
                gradinput2,
                gradinput3,
                self.lambda_e,
                self.lambda_v,
                self.Nw,
            )
            if err != 0:
                print(err)

        else:
            err = lib.WeightLayer_cpu_backward(
                self.input1,
                self.input2,
                self.input3,
                self.output,
                gradoutput,
                gradinput1,
                gradinput2,
                gradinput3,
                self.lambda_e,
                self.lambda_v,
                self.Nw,
            )
            if err != 0:
                print(err)

        return gradinput1, gradinput2, gradinput3


class PixelValueLayer(Function):
    def __init__(self, sigma_d=3, tao_r=0.05, Prowindow=2):
        super(PixelValueLayer, self).__init__()

        self.sigma_d = sigma_d
        self.tao_r = tao_r  # maybe not useable
        self.Prowindow = Prowindow

    def forward(self, input1, input3, flow_weights):

        self.input1 = input1.contiguous()  # ref1 image
        self.input3 = input3.contiguous()  # ref1 flow
        self.flow_weights = flow_weights.contiguous()  # ref1 flow weights

        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output = torch.zeros(input1.size())

        if input1.is_cuda:
            output = output.cuda()
            err = lib.PixelValueLayer_gpu_forward(
                input1,
                input3,
                flow_weights,
                output,
                self.sigma_d,
                self.tao_r,
                self.Prowindow,
            )
            if err != 0:
                print(err)
        else:
            err = lib.PixelValueLayer_cpu_forward(
                input1,
                input3,
                flow_weights,
                output,
                self.sigma_d,
                self.tao_r,
                self.Prowindow,
            )
            if err != 0:
                print(err)

        # the function returns the output to its caller
        return output

    # TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        gradinput1 = torch.zeros(self.input1.size())
        gradinput3 = torch.zeros(self.input3.size())
        gradflow_weights = torch.zeros(self.flow_weights.size())

        if self.input1.is_cuda:
            gradinput1 = gradinput1.cuda(self.device)
            gradinput3 = gradinput3.cuda(self.device)
            gradflow_weights = gradflow_weights.cuda(self.device)

            err = lib.PixelValueLayer_gpu_backward(
                self.input1,
                self.input3,
                self.flow_weights,
                gradoutput,
                gradinput1,
                gradinput3,
                gradflow_weights,
                self.sigma_d,
                self.tao_r,
                self.Prowindow,
            )
            if err != 0:
                print(err)

        else:
            err = lib.PixelValueLayer_cpu_backward(
                self.input1,
                self.input3,
                self.flow_weights,
                gradoutput,
                gradinput1,
                gradinput3,
                gradflow_weights,
                self.sigma_d,
                self.tao_r,
                self.Prowindow,
            )
            if err != 0:
                print(err)
        return gradinput1, gradinput3, gradflow_weights


class PixelWeightLayer(Function):
    def __init__(self, threshhold, sigma_d=3, tao_r=0.05, Prowindow=2):
        super(PixelWeightLayer, self).__init__()
        self.threshhold = threshhold
        self.sigma_d = sigma_d
        self.tao_r = tao_r  # maybe not useable
        self.Prowindow = Prowindow

    def forward(self, input3, flow_weights):

        self.input3 = input3.contiguous()  # ref1 flow
        self.flow_weights = flow_weights.contiguous()  # ref1 flow weights

        if input3.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output = torch.zeros([input3.size(0), 1, input3.size(2), input3.size(3)])

        if input3.is_cuda:
            output = output.cuda()
            err = lib.PixelWeightLayer_gpu_forward(
                input3, flow_weights, output, self.sigma_d, self.tao_r, self.Prowindow
            )
            if err != 0:
                print(err)
        else:
            err = lib.PixelWeightLayer_cpu_forward(
                input3, flow_weights, output, self.sigma_d, self.tao_r, self.Prowindow
            )
            if err != 0:
                print(err)

        self.output = output
        # the function returns the output to its caller
        return output

    # TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        gradinput3 = torch.zeros(self.input3.size())
        gradflow_weights = torch.zeros(self.flow_weights.size())

        if self.input3.is_cuda:
            gradinput3 = gradinput3.cuda(self.device)
            gradflow_weights = gradflow_weights.cuda(self.device)

            err = lib.PixelWeightLayer_gpu_backward(
                self.input3,
                self.flow_weights,
                self.output,
                gradoutput,
                gradinput3,
                gradflow_weights,
                self.threshhold,
                self.sigma_d,
                self.tao_r,
                self.Prowindow,
            )
            if err != 0:
                print(err)

        else:
            err = lib.PixelWeightLayer_cpu_backward(
                self.input3,
                self.flow_weights,
                self.output,
                gradoutput,
                gradinput3,
                gradflow_weights,
                self.threshhold,
                self.sigma_d,
                self.tao_r,
                self.Prowindow,
            )
            if err != 0:
                print(err)

        return gradinput3, gradflow_weights


class ReliableWeightLayer(Function):
    def __init__(self, threshhold, sigma_d=3, tao_r=0.05, Prowindow=2):
        super(ReliableWeightLayer, self).__init__()

        self.threshhold = threshhold
        self.sigma_d = sigma_d
        self.tao_r = tao_r  # maybe not useable
        self.Prowindow = Prowindow

    def forward(self, input3):

        self.input3 = input3.contiguous()  # ref1 flow

        if input3.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output = torch.zeros([input3.size(0), 1, input3.size(2), input3.size(3)])

        if input3.is_cuda:
            output = output.cuda()
            err = lib.ReliableWeightLayer_gpu_forward(
                input3, output, self.sigma_d, self.tao_r, self.Prowindow
            )
            if err != 0:
                print(err)
        else:
            err = lib.ReliableWeightLayer_cpu_forward(
                input3, output, self.sigma_d, self.tao_r, self.Prowindow
            )
            if err != 0:
                print(err)
        self.output = output  # used for inihibiting some unreliable gradients.
        # the function returns the output to its caller
        return output

    # TODO: if there are multiple outputs of this function, then the order should be well considered?
    def backward(self, gradoutput):
        gradinput3 = torch.zeros(self.input3.size())

        if self.input3.is_cuda:
            gradinput3 = gradinput3.cuda(self.device)

            err = lib.ReliableWeightLayer_gpu_backward(
                self.input3,
                self.output,
                gradoutput,
                gradinput3,
                self.threshhold,
                self.sigma_d,
                self.tao_r,
                self.Prowindow,
            )
            if err != 0:
                print(err)

        else:
            err = lib.ReliableWeightLayer_cpu_backward(
                self.input3,
                self.output,
                gradoutput,
                gradinput3,
                self.threshhold,
                self.sigma_d,
                self.tao_r,
                self.Prowindow,
            )
            # print(err)
            if err != 0:
                print(err)

        return gradinput3


class FilterInterpolationModule(Module):
    def __init__(self):
        super(FilterInterpolationModule, self).__init__()

    def forward(self, input1, input2, input3):
        return FilterInterpolationLayer.apply(input1, input2, input3)

    # we actually dont need to write the backward code for a module, since we have


class AdaptiveWeightInterpolationModule(Module):
    def __init__(
        self,
        training=False,
        threshhold=1e-6,
        lambda_e=30.0 / 255.0,
        lambda_v=1.0,
        Nw=3.0,
        sigma_d=1.5,
        tao_r=0.05,
        Prowindow=2,
    ):
        super(AdaptiveWeightInterpolationModule, self).__init__()

        self.calc_weight1 = WeightLayer(lambda_e, lambda_v, Nw)
        self.padder1 = torch.nn.ReplicationPad2d([0, 1, 0, 1])
        self.interpolate1 = PixelValueLayer(sigma_d, tao_r, Prowindow)
        self.interpolate1_1 = PixelWeightLayer(
            101 * threshhold, sigma_d, tao_r, Prowindow
        )
        self.interpolate_R1_1 = ReliableWeightLayer(
            101 * threshhold, sigma_d, tao_r, Prowindow
        )

        self.calc_weight2 = WeightLayer(lambda_e, lambda_v, Nw)
        self.padder2 = torch.nn.ReplicationPad2d([0, 1, 0, 1])
        self.interpolate2 = PixelValueLayer(sigma_d, tao_r, Prowindow)
        self.interpolate2_1 = PixelWeightLayer(
            101 * threshhold, sigma_d, tao_r, Prowindow
        )
        self.interpolate_R2_1 = ReliableWeightLayer(
            101 * threshhold, sigma_d, tao_r, Prowindow
        )

        self.training = training
        self.threshold = threshhold
        return

    # input1 ==> ref1 image
    # input2 ==> ref2 image
    # input3 ==> ref1 flow
    # input4 ==> ref2 flow
    def forward(self, input1, input2, input3, input4):
        epsilon = 1e-6

        flow_weight1 = self.calc_weight1(input1, input2, input3)
        p1 = self.interpolate1(input1, input3, flow_weight1)
        p1_r, p1_g, p1_b = torch.split(p1, 1, dim=1)
        pw1 = self.interpolate1_1(input3, flow_weight1)
        i1_r, i1_g, i1_b = (
            (p1_r) / (pw1 + self.threshold),
            (p1_g) / (pw1 + self.threshold),
            (p1_b) / (pw1 + self.threshold),
        )
        r1 = pw1
        rw1 = self.interpolate_R1_1(input3)
        w1 = (r1) / (rw1 + self.threshold)

        flow_weight2 = self.calc_weight2(input2, input1, input4)
        p2 = self.interpolate2(input2, input4, flow_weight2)
        p2_r, p2_g, p2_b = torch.split(p2, 1, dim=1)
        pw2 = self.interpolate2_1(input4, flow_weight2)
        i2_r, i2_g, i2_b = (
            (p2_r) / (pw2 + self.threshold),
            (p2_g) / (pw2 + self.threshold),
            (p2_b) / (pw2 + self.threshold),
        )
        r2 = pw2
        rw2 = self.interpolate_R2_1(input4)
        w2 = (r2) / (rw2 + self.threshold)

        w = w1 + w2
        i_r = (i1_r * w1 + i2_r * w2) / (w + self.threshold)  # (w1 + w2)
        i_g = (i1_g * w1 + i2_g * w2) / (w + self.threshold)  # (w1 + w2)
        i_b = (i1_b * w1 + i2_b * w2) / (w + self.threshold)  # (w1 + w2)
        if not self.training:
            i_r[w <= 10 * self.threshold], i_g[w <= 10 * self.threshold], i_b[
                w <= 10 * self.threshold
            ] = (0, 0, 0)
            w[w <= 10 * self.threshold] = 0
        i = torch.cat((i_r, i_g, i_b), dim=1)
        return i
