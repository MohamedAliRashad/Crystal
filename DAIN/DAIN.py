# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

from .DepthFlowProjection import DepthFlowProjectionModule
from .FilterInterpolation import FilterInterpolationModule
from .FlowProjection import FlowProjectionModule
from .MegaDepth import HourGlass
from .pwcnet import PWCDCNet
from .resblock import MultipleBasicBlock
from .s2df import S2DF
from .stack import Stack


class DAIN(torch.nn.Module):
    def __init__(self, channel=3, filter_size=4, timestep=0.5):

        # base class initialization
        super(DAIN, self).__init__()

        self.filter_size = filter_size
        self.timestep = timestep
        assert (
            timestep == 0.5
        )  # TODO: or else the WeigtedFlowProjection should also be revised... Really Tedious work.
        self.numFrames = int(1.0 / timestep) - 1

        i = 0
        self.initScaleNets_filter, self.initScaleNets_filter1, self.initScaleNets_filter2 = self.get_MonoNet5(
            channel if i == 0 else channel + filter_size * filter_size,
            filter_size * filter_size,
            "filter",
        )

        self.ctxNet = S2DF()
        self.ctx_ch = 3 * 64 + 3

        self.rectifyNet = MultipleBasicBlock(3 + 3 + 3 + 2 * 1 + 2 * 2 + 16 * 2 + 2 * self.ctx_ch, intermediate_feature=128)

        self._initialize_weights()

        self.flownets = PWCDCNet()
        self.div_flow = 20.0

        # extract depth information
        self.depthNet = HourGlass()

        return

    def _initialize_weights(self):
        count = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                count += 1
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            # else:
            #     print(m)

    def forward(self, input):

        """
        Parameters
        ----------
        input: shape (3, batch, 3, width, height)
        -----------
        """
        losses = []
        offsets = []
        filters = []
        occlusions = []

        device = torch.cuda.current_device()
        s1 = torch.cuda.current_stream()
        s2 = torch.cuda.current_stream()

        """
            STEP 1: sequeeze the input 
        """
        assert input.size(0) == 2
        input_0, input_2 = torch.squeeze(input, dim=0)

        # prepare the input data of current scale
        cur_input_0 = input_0
        cur_input_2 = input_2

        """
            STEP 3.2: concatenating the inputs.
        """
        cur_offset_input = torch.cat((cur_input_0, cur_input_2), dim=1)
        cur_filter_input = (
            cur_offset_input
        )

        """
            STEP 3.3: perform the estimation by the Three subpath Network 
        """
        time_offsets = [kk * self.timestep for kk in range(1, 1 + self.numFrames, 1)]

        with torch.cuda.stream(s1):
            temp = self.depthNet(
                torch.cat(
                    (cur_filter_input[:, :3, ...], cur_filter_input[:, 3:, ...]), dim=0
                )
            )
            log_depth = [
                temp[: cur_filter_input.size(0)],
                temp[cur_filter_input.size(0) :],
            ]

            cur_ctx_output = [
                torch.cat(
                    (self.ctxNet(cur_filter_input[:, :3, ...]), log_depth[0].detach()),
                    dim=1,
                ),
                torch.cat(
                    (self.ctxNet(cur_filter_input[:, 3:, ...]), log_depth[1].detach()),
                    dim=1,
                ),
            ]
            temp = self.forward_singlePath(
                self.initScaleNets_filter, cur_filter_input, "filter"
            )
            cur_filter_output = [
                self.forward_singlePath(self.initScaleNets_filter1, temp, name=None),
                self.forward_singlePath(self.initScaleNets_filter2, temp, name=None),
            ]

            depth_inv = [1e-6 + 1 / torch.exp(d) for d in log_depth]

        with torch.cuda.stream(s2):
            for _ in range(1):
                cur_offset_outputs = [
                    self.forward_flownets(
                        self.flownets, cur_offset_input, time_offsets=time_offsets
                    ),
                    self.forward_flownets(
                        self.flownets,
                        torch.cat(
                            (
                                cur_offset_input[:, 3:, ...],
                                cur_offset_input[:, 0:3, ...],
                            ),
                            dim=1,
                        ),
                        time_offsets=time_offsets[::-1],
                    ),
                ]

        torch.cuda.synchronize()  # synchronize s1 and s2

        cur_offset_outputs = [
            self.FlowProject(cur_offset_outputs[0], depth_inv[0]),
            self.FlowProject(cur_offset_outputs[1], depth_inv[1]),
        ]

        """
            STEP 3.4: perform the frame interpolation process 
        """
        cur_offset_output = [cur_offset_outputs[0][0], cur_offset_outputs[1][0]]
        ctx0, ctx2 = self.FilterInterpolate_ctx(
            cur_ctx_output[0], cur_ctx_output[1], cur_offset_output, cur_filter_output
        )

        cur_output, ref0, ref2 = self.FilterInterpolate(
            cur_input_0,
            cur_input_2,
            cur_offset_output,
            cur_filter_output,
            self.filter_size ** 2,
        )

        rectify_input = torch.cat(
            (
                cur_output,
                ref0,
                ref2,
                cur_offset_output[0],
                cur_offset_output[1],
                cur_filter_output[0],
                cur_filter_output[1],
                ctx0,
                ctx2,
            ),
            dim=1,
        )
        cur_output_rectified = self.rectifyNet(rectify_input) + cur_output

        """
            STEP 4: return the results
        """
        cur_outputs = [cur_output, cur_output_rectified]
        return cur_outputs, cur_offset_output, cur_filter_output

    def forward_flownets(self, model, input, time_offsets=None):

        if time_offsets == None:
            time_offsets = [0.5]
        elif type(time_offsets) == float:
            time_offsets = [time_offsets]
        elif type(time_offsets) == list:
            pass
        temp = model(
            input
        )  # this is a single direction motion results, but not a bidirectional one

        temps = [
            self.div_flow * temp * time_offset for time_offset in time_offsets
        ]  # single direction to bidirection should haven it.
        temps = [
            nn.Upsample(scale_factor=4, mode="bilinear")(temp) for temp in temps
        ]  # nearest interpolation won't be better i think
        return temps

    """keep this function"""

    def forward_singlePath(self, modulelist, input, name):
        stack = Stack()

        k = 0
        temp = []
        for layers in modulelist:  # self.initScaleNets_offset:

            if k == 0:
                temp = layers(input)
            else:
                # met a pooling layer, take its input
                if isinstance(layers, nn.AvgPool2d) or isinstance(layers, nn.MaxPool2d):
                    stack.push(temp)

                temp = layers(temp)

                # met a unpooling layer, take its output
                if isinstance(layers, nn.Upsample):
                    if name == "offset":
                        temp = torch.cat(
                            (temp, stack.pop()), dim=1
                        )  # short cut here, but optical flow should concat instead of add
                    else:
                        temp += (
                            stack.pop()
                        )  # short cut here, but optical flow should concat instead of add
            k += 1
        return temp

    """keep this funtion"""

    def get_MonoNet5(self, channel_in, channel_out, name):

        """
        Generally, the MonoNet is aimed to provide a basic module for generating either offset, or filter, or occlusion.

        :param channel_in: number of channels that composed of multiple useful information like reference frame, previous coarser-scale result
        :param channel_out: number of output the offset or filter or occlusion
        :param name: to distinguish between offset, filter and occlusion, since they should use different activations in the last network layer

        :return: output the network model
        """
        model = []

        # block1
        model += self.conv_relu(channel_in * 2, 16, (3, 3), (1, 1))
        model += self.conv_relu_maxpool(
            16, 32, (3, 3), (1, 1), (2, 2)
        )  # THE OUTPUT No.5
        # block2
        model += self.conv_relu_maxpool(
            32, 64, (3, 3), (1, 1), (2, 2)
        )  # THE OUTPUT No.4
        # block3
        model += self.conv_relu_maxpool(
            64, 128, (3, 3), (1, 1), (2, 2)
        )  # THE OUTPUT No.3
        # block4
        model += self.conv_relu_maxpool(
            128, 256, (3, 3), (1, 1), (2, 2)
        )  # THE OUTPUT No.2
        # block5
        model += self.conv_relu_maxpool(256, 512, (3, 3), (1, 1), (2, 2))

        # intermediate block5_5
        model += self.conv_relu(512, 512, (3, 3), (1, 1))

        # block 6
        model += self.conv_relu_unpool(
            512, 256, (3, 3), (1, 1), 2
        )  # THE OUTPUT No.1 UP
        # block 7
        model += self.conv_relu_unpool(
            256, 128, (3, 3), (1, 1), 2
        )  # THE OUTPUT No.2 UP
        # block 8
        model += self.conv_relu_unpool(128, 64, (3, 3), (1, 1), 2)  # THE OUTPUT No.3 UP

        # block 9
        model += self.conv_relu_unpool(64, 32, (3, 3), (1, 1), 2)  # THE OUTPUT No.4 UP

        # block 10
        model += self.conv_relu_unpool(32, 16, (3, 3), (1, 1), 2)  # THE OUTPUT No.5 UP

        # output our final purpose
        branch1 = []
        branch2 = []
        branch1 += self.conv_relu_conv(16, channel_out, (3, 3), (1, 1))
        branch2 += self.conv_relu_conv(16, channel_out, (3, 3), (1, 1))

        return (nn.ModuleList(model), nn.ModuleList(branch1), nn.ModuleList(branch2))

    """keep this function"""

    @staticmethod
    def FlowProject(inputs, depth=None):
        if depth is not None:
            outputs = [
                DepthFlowProjectionModule(input.requires_grad)(input, depth)
                for input in inputs
            ]
        else:
            outputs = [
                FlowProjectionModule(input.requires_grad)(input) for input in inputs
            ]
        return outputs

    """keep this function"""

    @staticmethod
    def FilterInterpolate_ctx(ctx0, ctx2, offset, filter):
        ##TODO: which way should I choose

        ctx0_offset = FilterInterpolationModule()(
            ctx0, offset[0].detach(), filter[0].detach()
        )
        ctx2_offset = FilterInterpolationModule()(
            ctx2, offset[1].detach(), filter[1].detach()
        )

        return ctx0_offset, ctx2_offset

    """Keep this function"""

    @staticmethod
    def FilterInterpolate(ref0, ref2, offset, filter, filter_size2):
        ref0_offset = FilterInterpolationModule()(ref0, offset[0], filter[0])
        ref2_offset = FilterInterpolationModule()(ref2, offset[1], filter[1])
        return ref0_offset / 2.0 + ref2_offset / 2.0, ref0_offset, ref2_offset

    """keep this function"""

    @staticmethod
    def conv_relu_conv(input_filter, output_filter, kernel_size, padding):

        # we actually don't need to use so much layer in the last stages.
        layers = nn.Sequential(
            nn.Conv2d(input_filter, input_filter, kernel_size, 1, padding),
            nn.ReLU(inplace=False),
            nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
        )
        return layers

    """keep this fucntion"""

    @staticmethod
    def conv_relu(input_filter, output_filter, kernel_size, padding):
        layers = nn.Sequential(
            *[
                nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
                nn.ReLU(inplace=False),
            ]
        )
        return layers

    """keep this function"""

    @staticmethod
    def conv_relu_maxpool(
        input_filter, output_filter, kernel_size, padding, kernel_size_pooling
    ):

        layers = nn.Sequential(
            *[
                nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
                nn.ReLU(inplace=False),
                nn.MaxPool2d(kernel_size_pooling),
            ]
        )
        return layers

    """klkeep this function"""

    @staticmethod
    def conv_relu_unpool(
        input_filter, output_filter, kernel_size, padding, unpooling_factor
    ):

        layers = nn.Sequential(
            *[
                nn.Upsample(scale_factor=unpooling_factor, mode="bilinear"),
                nn.Conv2d(input_filter, output_filter, kernel_size, 1, padding),
                nn.ReLU(inplace=False)
            ]
        )
        return layers

if __name__ == "__main__":
    d = DAIN()

    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url("http://vllab1.ucmerced.edu/~wenbobao/DAIN/best.pth")
    d.load_state_dict(state_dict)
