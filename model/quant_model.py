### https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from imagenet_model.quant_conv import QConv, QLinear, QConv3x3, QConv1x1, SwitchableBatchNorm2d
from .shuffle_utils import channel_shuffle

__all__ = ['resnet18_quant', 'resnet34_quant', 'resnet50_quant']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}


class QBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, oneBit_outchannel=-1, oneBit_inchannel=-1, mid_oneBit=-1,
                 last_conv=False, first_conv=False, shuffle=False): # shuffle not works, mid_oneBit not works
        super(QBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = SwitchableBatchNorm2d
        # if groups != 1 or base_width != 64:
        #     raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if first_conv:
            self.conv1 = QConv3x3(args, inplanes, planes, stride, groups=groups,
                                  oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                                  first_conv=first_conv)
        else:
            self.conv1 = QConv3x3(args, inplanes, planes, stride, groups=groups,
                                  oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel)
        self.bn1 = norm_layer(planes, groups=groups, oneBit_outchannel=oneBit_outchannel)
        self.relu = nn.ReLU(inplace=True)

        if last_conv:
            self.conv2 = QConv3x3(args, planes, planes, groups=groups,
                                  oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_outchannel,
                                  last_conv=last_conv)
            self.bn2 = norm_layer(planes, groups=groups, oneBit_outchannel=oneBit_outchannel, last_conv=last_conv)

        else:
            self.conv2 = QConv3x3(args, planes, planes, groups=groups,
                              oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_outchannel)
            self.bn2 = norm_layer(planes, groups=groups, oneBit_outchannel=oneBit_outchannel)
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        self.weight_bit = 2
        self.act_bit = 8
        self.oneBit_outchannel = oneBit_outchannel
        self.oneBit_inchannel = oneBit_inchannel
        self.last_conv = last_conv
        self.first_conv = first_conv
        self.inplanes = inplanes

    def select(self, identity):
        if (self.weight_bit & 1) == 0:
            index = self.inplanes // self.groups * (self.weight_bit // 2)
            return identity[:, :index, :, :]
        else:
            index = self.inplanes // self.groups * (self.weight_bit // 2) + self.oneBit_outchannel
            return identity[:, :index, :, :]

    def addZeroS(self, identity):
        if (self.weight_bit & 1) == 0:
            return identity
        else:
            index = self.inplanes // self.groups - self.oneBit_outchannel
            zeros = torch.zeros(identity.shape[0], index, identity.shape[2], identity.shape[3]).cuda()
            out = torch.cat((identity, zeros), dim=1)
            return out

    def forward(self, x):
        identity = x

        if self.first_conv:
            identity = self.select(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.last_conv:
            identity = self.addZeroS(identity)

        if self.downsample is not None:
            out = channel_shuffle(out, self.weight_bit//2, self.weight_bit)
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        if self.last_conv:
            # if self.weight_bit == 2 or self.weight_bit == 4 or self.weight_bit == 6 or self.weight_bit == 8:
            #     out = out.chunk(self.weight_bit // 2, dim=1)
            # if self.weight_bit == 3 or self.weight_bit == 5 or self.weight_bit == 7:
            #     out = out.chunk((self.weight_bit+1) // 2, dim=1)
            out = out.chunk((self.weight_bit + 1) // 2, dim=1)

            # out = torch.sum(torch.stack(out, dim=0), dim=0)
            out = torch.mean(torch.stack(out, dim=0), dim=0)

        return out


class QBottleneck(nn.Module):
    expansion = 4

    def __init__(self, args, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, oneBit_outchannel=-1, oneBit_inchannel=-1, mid_oneBit=-1,
                 last_conv=False, first_conv=False, shuffle=False): # shuffle not works
        super(QBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = SwitchableBatchNorm2d
        width = int(planes * (base_width / 64.))

        if first_conv:
            self.conv1 = QConv1x1(args, inplanes, width, groups=groups,
                                  oneBit_outchannel=mid_oneBit, oneBit_inchannel=oneBit_inchannel,
                                  first_conv=first_conv)
        else:
            self.conv1 = QConv1x1(args, inplanes, width, groups=groups,
                                  oneBit_outchannel=mid_oneBit, oneBit_inchannel=oneBit_inchannel)
        self.bn1 = norm_layer(width, groups=groups, oneBit_outchannel=mid_oneBit)


        self.conv2 = QConv3x3(args, width, width, stride, groups=groups,
                              oneBit_outchannel=mid_oneBit, oneBit_inchannel=mid_oneBit)
        self.bn2 = norm_layer(width, groups=groups, oneBit_outchannel=mid_oneBit)


        if last_conv:
            self.conv3 = QConv1x1(args, width, planes * self.expansion, groups=groups,
                                  oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=mid_oneBit,
                                  last_conv=last_conv)
            self.bn3 = norm_layer(planes * self.expansion, groups=groups, oneBit_outchannel=oneBit_outchannel,
                                  last_conv=last_conv)
        else:
            self.conv3 = QConv1x1(args, width, planes * self.expansion, groups=groups,
                              oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=mid_oneBit)
            self.bn3 = norm_layer(planes * self.expansion, groups=groups, oneBit_outchannel=oneBit_outchannel)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        self.weight_bit = 2
        self.act_bit = 8
        self.oneBit_outchannel = oneBit_outchannel
        self.oneBit_inchannel = oneBit_inchannel
        self.last_conv = last_conv
        self.first_conv = first_conv
        self.inplanes = inplanes

    def select(self, identity):
        if (self.weight_bit & 1) == 0:
            index = self.inplanes // self.groups * (self.weight_bit // 2)
            return identity[:, :index, :, :]
        else:
            index = self.inplanes // self.groups * ((self.weight_bit+1) // 2)
            return identity[:, :index, :, :]

    def addZeroS(self, identity):
        if (self.weight_bit & 1) == 0:
            return identity
        else:
            index = self.inplanes // self.groups - self.oneBit_outchannel
            zeros = torch.zeros(identity.shape[0], index, identity.shape[2], identity.shape[3]).cuda()
            out = torch.cat((identity, zeros), dim=1)
            return out

    def forward(self, x):
        identity = x

        if self.first_conv:
            identity = self.select(identity)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.last_conv:
            identity = self.addZeroS(identity)

        if self.downsample is not None:
            # out = channel_shuffle(out, self.weight_bit // 2, self.weight_bit)
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        if self.last_conv:
            out = out.chunk((self.weight_bit + 1) // 2, dim=1)
            out = torch.mean(torch.stack(out, dim=0), dim=0)

        return out


class QResNet(nn.Module):

    def __init__(self, args, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(QResNet, self).__init__()
        if norm_layer is None:
            norm_layer = SwitchableBatchNorm2d
        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.inplanes = 64 * self.groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes // self.groups, kernel_size=7, stride=2, padding=3,  # the first layer uses fp weights
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes // self.groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if args.arch == 'resnet18_quant':
            self.layer1 = self._make_layer(args, block, 64 * self.groups, layers[0],
                                        oneBit_outchannel=40, oneBit_inchannel=64, first_conv=True)
            self.layer2 = self._make_layer(args, block, 128 * self.groups, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0],
                                        oneBit_outchannel=76, oneBit_inchannel=40)
            self.layer3 = self._make_layer(args, block, 256 * self.groups, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1],
                                        oneBit_outchannel=184, oneBit_inchannel=76)
            self.layer4 = self._make_layer(args, block, 512 * self.groups, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2],
                                        oneBit_outchannel=396, oneBit_inchannel=184, last_conv=True)
            
        elif args.arch == 'resnet34_quant':
            self.layer1 = self._make_layer(args, block, 64 * self.groups, layers[0],
                                        oneBit_outchannel=39, oneBit_inchannel=64, first_conv=True)
            self.layer2 = self._make_layer(args, block, 128 * self.groups, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0],
                                        oneBit_outchannel=78, oneBit_inchannel=39)
            self.layer3 = self._make_layer(args, block, 256 * self.groups, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1],
                                        oneBit_outchannel=188, oneBit_inchannel=78)
            self.layer4 = self._make_layer(args, block, 512 * self.groups, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2],
                                        oneBit_outchannel=400, oneBit_inchannel=188, last_conv=True)
            
        elif args.arch == 'resnet50_quant':
            self.layer1 = self._make_layer(args, block, 64 * self.groups, layers[0],
                                        oneBit_outchannel=208, oneBit_inchannel=64, mid_oneBit=40, first_conv=True)
            self.layer2 = self._make_layer(args, block, 128 * self.groups, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0],
                                        oneBit_outchannel=416, oneBit_inchannel=208, mid_oneBit=80)
            self.layer3 = self._make_layer(args, block, 256 * self.groups, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1],
                                        oneBit_outchannel=832, oneBit_inchannel=416, mid_oneBit=160)
            self.layer4 = self._make_layer(args, block, 512 * self.groups, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2],
                                        oneBit_outchannel=2048, oneBit_inchannel=832,
                                        mid_oneBit=320, last_conv=True)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # the last layer uses fp weights

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, QConv):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, QBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, args, block, planes, blocks, stride=1, dilate=False,
                    oneBit_outchannel=-1, oneBit_inchannel=-1, mid_oneBit=-1, last_conv=False, first_conv=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv1x1(args, self.inplanes, planes * block.expansion, stride, groups=self.groups,
                         oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel),
                norm_layer(planes * block.expansion, groups=self.groups, oneBit_outchannel=oneBit_outchannel),
            )

        layers = []
        layers.append(block(args, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer,
                            oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                            mid_oneBit=mid_oneBit, first_conv=first_conv))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(args, self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer,
                                    oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_outchannel,
                                    mid_oneBit=mid_oneBit, last_conv=last_conv))
            else:
                layers.append(block(args, self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer,
                                    oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_outchannel,
                                    mid_oneBit=mid_oneBit))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.tile(1, self.groups, 1, 1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = x.chunk(self.groups, dim=1)
        # x = torch.sum(torch.stack(x), dim=0)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet_quant(args, arch, block, layers, pretrained, progress, **kwargs):
    model = QResNet(args, block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        print("load pretrained full-precision weights")
        print(model.load_state_dict(state_dict, strict=False))
    return model


def resnet18_quant(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_quant(args, 'resnet18', QBasicBlock, [2, 2, 2, 2], pretrained, progress,
                         **kwargs)


def resnet34_quant(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_quant(args, 'resnet34', QBasicBlock, [3, 4, 6, 3], pretrained, progress,
                         **kwargs)


def resnet50_quant(args, pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet_quant(args, 'resnet50', QBottleneck, [3, 4, 6, 3], pretrained, progress,
                         **kwargs)



import torch
import torch.nn as nn
import math
import numpy as np
from torch.hub import load_state_dict_from_url
from imagenet_model.quant_conv import QConv, QConv_Tra_Mulit, SwitchableBatchNorm2d, SwitchableBatchNorm2d_Tra_Mulit
from .shuffle_utils import channel_shuffle_mv1



class QDepthwiseSeparableConv(nn.Module):
    def __init__(self, args, inp=0, outp=0, stride=0,
                 oneBit_outchannel=0, oneBit_inchannel=0,
                 last_conv=False, first_conv=False, channel_shuffle=False):
        super(QDepthwiseSeparableConv, self).__init__()
        assert stride in [1, 2]
        self.weight_bit = 2
        self.act_bit = 8
        self.last_conv = last_conv
        self.first_conv = first_conv
        self.basewidth = inp // args.groups
        self.channel_shuffle = channel_shuffle

        layers = [
            QConv_Tra_Mulit(inp, inp, kernel_size=3, stride=stride,
                            padding=1, groups=inp, bias=False, dilation=1, args=args,
                            oneBit_outchannel=oneBit_inchannel,
                            basewidth=self.basewidth,
                            first_conv=first_conv),
            SwitchableBatchNorm2d_Tra_Mulit(inp, basewidth=self.basewidth, oneBit_outchannel=oneBit_inchannel),
            nn.ReLU(inplace=True),

            QConv(inp, outp, kernel_size=1, stride=1,
                     padding=0, groups=args.groups, bias=False, args=args,
                     oneBit_outchannel=oneBit_outchannel, oneBit_inchannel=oneBit_inchannel,
                     last_conv=last_conv),
            SwitchableBatchNorm2d(outp, groups=args.groups, oneBit_outchannel=oneBit_outchannel),
            nn.ReLU(inplace=True),
        ]
        self.body = nn.Sequential(*layers)

    def forward(self, x):
        out = self.body(x)
        # if self.channel_shuffle:
        #     out = channel_shuffle_mv1(out, self.weight_bit // 2, self.weight_bit)
        if self.last_conv:
            out = out.chunk((self.weight_bit + 1) // 2, dim=1)
            out = torch.mean(torch.stack(out, dim=0), dim=0)
        return out


class Model(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(Model, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

        oneBit = [52, 40, 101, 148, 223, 373, 404, 236, 376, 419, 233, 892, 1024]
        # oneBit = [56, 43, 108, 156, 240, 404, 435, 250, 404, 450, 250, 962, 1024]
        j = 0
        # head
        channels = 32
        first_stride = 2
        self.head = nn.Sequential(
            nn.Conv2d(
                3, channels, kernel_size=3,
                stride=first_stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.groups = args.groups
        channels = channels * self.groups
        # body
        for idx, [c, n, s] in enumerate(self.block_setting):
            outp = c * self.groups
            if idx == len(self.block_setting) - 1:
                for i in range(n):
                    if i == 0:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s,
                                                        oneBit_inchannel=oneBit[j - 1], oneBit_outchannel=oneBit[j],
                                                        channel_shuffle=(i == n//2-1)))
                    elif i == n - 1:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1, last_conv=True,
                                                        oneBit_inchannel=oneBit[j - 1], oneBit_outchannel=oneBit[j],
                                                        channel_shuffle=(i == n//2-1)))
                    else:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
                                                        oneBit_inchannel=oneBit[j - 1], oneBit_outchannel=oneBit[j],
                                                        channel_shuffle=(i == n//2-1)))
                    channels = outp
                    j += 1
            elif idx == 0:
                for i in range(n):
                    assert n == 1
                    setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s,
                                                    oneBit_inchannel=32, oneBit_outchannel=oneBit[j], first_conv=True))
                    channels = outp
                    j += 1
            else:
                for i in range(n):
                    if i == 0:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s,
                                                    oneBit_inchannel=oneBit[j-1], oneBit_outchannel=oneBit[j],
                                                    channel_shuffle=(i == n//2-1)))
                    elif i == n - 1:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
                                                    oneBit_inchannel=oneBit[j-1], oneBit_outchannel=oneBit[j],
                                                    channel_shuffle=(i == n//2-1)))
                    else:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
                                                    oneBit_inchannel=oneBit[j-1], oneBit_outchannel=oneBit[j],
                                                    channel_shuffle=(i == n//2-1)))
                    channels = outp
                    j += 1

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # classifier
        self.classifier = nn.Linear(1024, num_classes)  # the last layer uses fp weights

        self.reset_parameters()
        self.i = 0
        self.args = args

    def forward(self, x):
        # if self.i % 50 == 0:
        #     print(self.i, x.shape)
        # self.i += 1
        # print(x[-1][0][0])
        x = self.head(x)
        # print(self.head[1].running_mean, self.head[1].running_var,
        #       self.head[1].track_running_stats, self.head[1].weight, self.head[1].bias)
        # import IPython
        # IPython.embed()
        # print('1', x[-1][0][0])
        # import sys
        # sys.exit()
        x = x.tile(1, self.groups, 1, 1)
        for idx, [_, n, _] in enumerate(self.block_setting):
            for i in range(n):
                x = getattr(self, 'stage_{}_layer_{}'.format(idx, i))(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def reset_parameters(self):
        for n, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mv1_quant(args, **kwargs):
    return Model(args)
