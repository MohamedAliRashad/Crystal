import torch
from torch.autograd import Function
from torch.nn import Module
from torch.utils.cpp_extension import load

lib = load(
    name="depthflowprojection_cuda",
    sources=["helper/DepthFlowProjection/depthflowprojection_cuda.cc", "helper/DepthFlowProjection/depthflowprojection_cuda_kernel.cu"],
    verbose=True,
)


class DepthFlowProjectionLayer(Function):
    def __init__(self, requires_grad):
        super(DepthFlowProjectionLayer, self).__init__()

    @staticmethod
    def forward(ctx, input1, input2, requires_grad):
        assert input1.is_contiguous()
        assert input2.is_contiguous()
        fillhole = 1 if requires_grad == False else 0

        if input1.is_cuda:
            count = (
                torch.cuda.FloatTensor()
                .resize_(input1.size(0), 1, input1.size(2), input1.size(3))
                .zero_()
            )
            output = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            err = lib.DepthFlowProjectionLayer_gpu_forward(
                input1, input2, count, output, fillhole
            )
        else:
            count = (
                torch.FloatTensor()
                .resize_(input1.size(0), 1, input1.size(2), input1.size(3))
                .zero_()
            )
            output = torch.FloatTensor().resize_(input1.size()).zero_()
            err = lib.DepthFlowProjectionLayer_cpu_forward(
                input1, input2, count, output, fillhole
            )
        if err != 0:
            print(err)

        ctx.save_for_backward(input1, input2, count, output)
        ctx.fillhole = fillhole

        return output

    @staticmethod
    def backward(ctx, gradoutput):

        input1, input2, count, output = ctx.saved_tensors

        if input1.is_cuda:
            gradinput1 = torch.cuda.FloatTensor().resize_(input1.size()).zero_()
            gradinput2 = torch.cuda.FloatTensor().resize_(input2.size()).zero_()

            err = lib.DepthFlowProjectionLayer_gpu_backward(
                input1, input2, count, output, gradoutput, gradinput1, gradinput2
            )
            if err != 0:
                print(err)

        else:
            gradinput1 = torch.FloatTensor().resize_(input1.size()).zero_()
            gradinput2 = torch.FloatTensor().resize_(input2.size()).zero_()
            err = lib.DepthFlowProjectionLayer_cpu_backward(
                input1, input2, count, output, gradoutput, gradinput1, gradinput2
            )
            if err != 0:
                print(err)

        return gradinput1, gradinput2, None


class DepthFlowProjectionModule(Module):
    def __init__(self, requires_grad=True):
        super(DepthFlowProjectionModule, self).__init__()
        self.requires_grad = requires_grad

    def forward(self, input1, input2):
        return DepthFlowProjectionLayer.apply(input1, input2, self.requires_grad)
