# modules/FlowProjectionModule.py
from torch.nn import Module

# this is for wrapping the customized layer
import torch
from torch.autograd import Function
from torch.utils.cpp_extension import load

lib = load(
    name="flowprojection_cuda",
    sources=["helper/FlowProjection/flowprojection_cuda.cc", "helper/FlowProjection/flowprojection_cuda_kernel.cu"],
    verbose=True,
)


class FlowProjectionLayer(Function):
    def __init__(self, requires_grad):
        super(FlowProjectionLayer, self).__init__()
        self.requires_grad = requires_grad

    @staticmethod
    def forward(ctx, input1, requires_grad):
        assert input1.is_contiguous()

        fillhole = 1 if requires_grad == False else 0

        if input1.is_cuda:
            count = (
                torch.cuda.FloatTensor()
                .resize_(input1.size(0), 1, input1.size(2), input1.size(3))
                .zero_()
            )
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            err = lib.FlowProjectionLayer_gpu_forward(input1, count, output, fillhole)
        else:
            output = torch.cuda.FloatTensor(input1.data.size())
            err = lib.FlowProjectionLayer_cpu_forward(input1, count, output, fillhole)
        if err != 0:
            print(err)

        ctx.save_for_backward(input1, count)
        ctx.fillhole = fillhole

        return output

    @staticmethod
    def backward(ctx, gradoutput):

        input1, count, output = ctx.saved_tensors

        if input1.is_cuda:
            gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            err = lib.FlowProjectionLayer_gpu_backward(
                input1, count, gradoutput, gradinput1
            )
            if err != 0:
                print(err)

        else:
            gradinput1 = torch.FloatTensor().resize_(input1.size()).zero_()
            err = lib.FlowProjectionLayer_cpu_backward(
                input1, count, gradoutput, gradinput1
            )
            if err != 0:
                print(err)

        return gradinput1, None


class FlowFillholelayer(Function):
    def __init__(self):
        super(FlowFillholelayer, self).__init__()

    def forward(self, input1):
        self.input1 = (
            input1.contiguous()
        )  # need to use in the backward process, so we need to cache it

        if input1.is_cuda:
            self.device = torch.cuda.current_device()
        else:
            self.device = -1

        output = torch.zeros(input1.size())

        if input1.is_cuda:
            output = output.cuda()
            err = lib.FlowFillholelayer_gpu_forward(input1, output)
        else:
            err = lib.FlowFillholelayer_cpu_forward(input1, output)
        if err != 0:
            print(err)

        # the function returns the output to its caller
        return output


class FlowProjectionModule(Module):
    def __init__(self, requires_grad=True):
        super(FlowProjectionModule, self).__init__()

        self.f = FlowProjectionLayer(requires_grad)

    def forward(self, input1):
        return self.f(input1)
