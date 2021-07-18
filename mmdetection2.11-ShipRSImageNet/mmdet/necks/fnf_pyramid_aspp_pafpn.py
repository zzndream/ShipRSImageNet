import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from mmcv.cnn import ConvModule, xavier_init, kaiming_init, constant_init
from mmcv.runner import auto_fp16

from ..builder import NECKS
from .fpn import FPN
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _NewEmptyTensorOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


class free_dilation_Conv2d(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            bias=False,
            norm=None,
            activation=None,
    ):
        super(free_dilation_Conv2d, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.with_bias = bias
        self.padding = padding
        self.dilation = dilation
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(
            torch.Tensor(out_channel, in_channel, *self.kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channel))
        else:
            self.bias = None

        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)

    def _updateweight_2d5(self):
        # clone 和 detach有讲究 https://blog.csdn.net/winycg/article/details/100813519
        # self.freeweight = self.weight.clone().detach()
        # self.freeweight = self.freeweight.expand(self.weight.shape[0],self.weight.shape[1],self.weight.shape[2] + 2, self.weight.shape[3] +2)

        DEVICE = self.weight.get_device()
        self.freeweight_2d5 = torch.zeros(
            [self.weight.shape[0], self.weight.shape[1], self.weight.shape[2] + 2, self.weight.shape[3] + 2])
        self.freeweight_2d5 = self.freeweight_2d5.to(DEVICE)

        biases = [[[0, 0], [1, 1], [0, 2]], \
                  [[1, 1], [1, 1], [1, 1]], \
                  [[2, 0], [1, 1], [2, 2]]]
        for k in range(self.weight.shape[2]):
            for l in range(self.weight.shape[3]):
                self.freeweight_2d5[:, :, k + biases[k][l][0], l + biases[k][l][1]] = self.weight[:, :, k, l]

    def _updateweight_2d7(self):
        # clone 和 detach有讲究 https://blog.csdn.net/winycg/article/details/100813519
        # self.freeweight = self.weight.clone().detach()
        # self.freeweight = self.freeweight.expand(self.weight.shape[0],self.weight.shape[1],self.weight.shape[2] + 2, self.weight.shape[3] +2)

        DEVICE = self.weight.get_device()
        self.freeweight_2d7 = torch.zeros(
            [self.weight.shape[0], self.weight.shape[1], self.weight.shape[2] + 4, self.weight.shape[3] + 4])
        self.freeweight_2d7 = self.freeweight_2d7.to(DEVICE)

        biases = [[[1, 0], [4, 0], [4, 2]], \
                  [[0, 0], [2, 2], [4, 2]], \
                  [[0, 2], [4, 0], [4, 4]]]
        for k in range(self.weight.shape[2]):
            for l in range(self.weight.shape[3]):
                self.freeweight_2d7[:, :, k + biases[k][l][0], l + biases[k][l][1]] = self.weight[:, :, k, l]

    def forward(self, x):
        if self.dilation == 1.5:
            self._updateweight_2d5()
            output = F.conv2d(x, self.freeweight_2d5, self.bias, self.stride, 2, 1)
            # output2 = F.conv2d(x, self.weight, self.bias, self.stride, 2, 2)
            # output = (output1 + output2) / 2
        elif self.dilation == 2.5:
            self._updateweight_2d7()
            # output1 = F.conv2d(x, self.weight, self.bias, self.stride, 2, 2)
            output = F.conv2d(x, self.freeweight_2d7, self.bias, self.stride, 3, 1)
            # output = (output1 + output2) / 2
        else:
            output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation)


        if self.norm is not None:
            output = self.norm(output)
        if self.activation is not None:
            output = self.activation(output)
        return output

    def extra_repr(self):
        tmpstr = "in_channel=" + str(self.in_channel)
        tmpstr += ", out_channel=" + str(self.out_channel)
        tmpstr += ", kernel_size=" + str(self.kernel_size)
        tmpstr += ", stride=" + str(self.stride)
        tmpstr += ", padding=" + str(self.padding)
        tmpstr += ", dilation=" + str(self.dilation)
        tmpstr += ", bias=" + str(self.with_bias)
        return tmpstr

class FNF_ASPP(nn.Module):
    """ASPP (Atrous Spatial Pyramid Pooling)

    This is an implementation of the ASPP module used in DetectoRS
    (https://arxiv.org/pdf/2006.02334.pdf)

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of channels produced by this module
        dilations (tuple[int]): Dilations of the four branches.
            Default: (1, 3, 6, 1)
    """

    def __init__(self, in_channels, out_channels, dilations=(1, 3, 6, 1)):
        super().__init__()
        assert dilations[-1] == 1
        self.aspp = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
        self.gap = nn.AdaptiveAvgPool2d(1)
        concat_ch = out_channels * (len(dilations))
        self.conv_out = nn.Conv2d(
            concat_ch,
            out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            padding=0,
            bias=True)

        # Weight
        self.fusion_weight = nn.Parameter(torch.ones(len(dilations)+1, dtype=torch.float32), requires_grad=True)
        self.fusion_weight_relu = nn.ReLU()
        self.epsilon = 1e-4,

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)

    def forward(self, x):
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp)):
            inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
            out.append(F.relu_(self.aspp[aspp_idx](inp)))
        out[-1] = out[-1].expand_as(out[-2])


        # Weights for fusion
        fusion_weight = self.fusion_weight_relu(fusion_weight)
        weight = fusion_weight / (torch.sum(fusion_weight, dim=0) + self.epsilon)
        # fusion out
        output = []
        for aspp_idx in range(len(self.aspp)+1):
            output = output + weight[i] * out[i]

        # out = torch.cat(out, dim=1)
        # out = self.conv_out(out)
        return output
# class ASPP(nn.Module):
#     """ASPP (Atrous Spatial Pyramid Pooling)
#
#     This is an implementation of the ASPP module used in DetectoRS
#     (https://arxiv.org/pdf/2006.02334.pdf)
#
#     Args:
#         in_channels (int): Number of input channels.
#         out_channels (int): Number of channels produced by this module
#         dilations (tuple[int]): Dilations of the four branches.
#             Default: (1, 3, 6)
#     """
#
#     def __init__(self, in_channels, out_channels, dilations=(1, 3, 6)):
#         super().__init__()
#         #z最后一个扩张率实际上没有用到，会被替换为全局池化
#         assert dilations[-1] == 1
#         self.aspp = nn.ModuleList()
#         for dilation in dilations:
#             kernel_size = 3 if dilation > 1 else 1
#             padding = dilation if dilation > 1 else 0
#             conv = nn.Conv2d(
#                 in_channels,
#                 out_channels,
#                 kernel_size=kernel_size,
#                 stride=1,
#                 dilation=dilation,
#                 padding=padding,
#                 bias=True)
#             self.aspp.append(conv)
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         concat_ch = out_channels * (len(dilations))
#         self.conv_out = nn.Conv2d(
#             concat_ch,
#             out_channels,
#             kernel_size=1,
#             stride=1,
#             dilation=1,
#             padding=0,
#             bias=True)
#
#         self.init_weights()
#
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 kaiming_init(m)
#
#     def forward(self, x):
#         avg_x = self.gap(x)
#         out = []
#         for aspp_idx in range(len(self.aspp)):
#             # 最后一个扩张率被地换位全局池化
#             inp = avg_x if (aspp_idx == len(self.aspp) - 1) else x
#             out.append(F.relu_(self.aspp[aspp_idx](inp)))
#         out[-1] = out[-1].expand_as(out[-2])
#         out = torch.cat(out, dim=1)
#         out = self.conv_out(out)
#         return out


@NECKS.register_module()
class FNF_Pyramid_ASPP_PAFPN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAFPN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool): Whether to add conv layers on top of the
            original feature maps. Default: False.
        extra_convs_on_inputs (bool): Whether to apply extra conv on
            the original feature from the backbone. Default: False.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 aspp_dilations=(1, 3, 6),
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None):
        super(FNF_Pyramid_ASPP_PAFPN,
              self).__init__(in_channels, out_channels, num_outs, start_level,
                             end_level, add_extra_convs, extra_convs_on_inputs,
                             relu_before_extra_convs, no_norm_on_lateral,
                             conv_cfg, norm_cfg, act_cfg)
        # add extra bottom up pathway
        self.aspp_dilations = aspp_dilations
        self.aspp_convs = nn.ModuleList()
        self.downsample_convs = nn.ModuleList()
        self.pafpn_convs = nn.ModuleList()

        for i in range(self.start_level + 1, self.backbone_end_level):

            if i == 1:
                adaptive_aspp_dilations = (1, 3, 6, 12, 1)
            elif i == 2:
                adaptive_aspp_dilations = (1, 3, 6, 1)
            elif i == 3:
                adaptive_aspp_dilations = (1, 3, 1)
            elif i == 4:
                adaptive_aspp_dilations = (1, 2, 1)
            else:
                adaptive_aspp_dilations = (1, 2, 1)

            aspp_conv = FNF_ASPP(out_channels, out_channels,
                             adaptive_aspp_dilations)

            d_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                stride=2,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            pafpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)
            self.aspp_convs.append(aspp_conv)
            self.downsample_convs.append(d_conv)
            self.pafpn_convs.append(pafpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)
        # 横向连接
        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # 构建自顶向下路径
        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')
        # 构建输出
        # build outputs
        #
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i] = self.aspp_convs[i](inter_outs[i])  # zzn
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
