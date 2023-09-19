### https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .quant_modules import QConv, QLinear, SwitchableBatchNorm2d, QBasicBlock, QBottleneck, QDepthwiseSeparableConv
from .quant_ops import QConv1x1, QConv3x3
from .shuffle_utils import channel_shuffle

__all__ = ['resnet18_quant', 'resnet34_quant', 'resnet50_quant', 'mv1_quant', 'MobileNetV1']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
}

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
                                         first_conv=True)
            self.layer2 = self._make_layer(args, block, 128 * self.groups, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0],
                                        )
            self.layer3 = self._make_layer(args, block, 256 * self.groups, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(args, block, 512 * self.groups, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2], last_conv=True)
            
        elif args.arch == 'resnet34_quant':
            self.layer1 = self._make_layer(args, block, 64 * self.groups, layers[0], first_conv=True)
            self.layer2 = self._make_layer(args, block, 128 * self.groups, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0]
                                        )
            self.layer3 = self._make_layer(args, block, 256 * self.groups, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(args, block, 512 * self.groups, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2], last_conv=True)
            
        elif args.arch == 'resnet50_quant':
            self.layer1 = self._make_layer(args, block, 64 * self.groups, layers[0], first_conv=True)
            self.layer2 = self._make_layer(args, block, 128 * self.groups, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(args, block, 256 * self.groups, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(args, block, 512 * self.groups, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2],
                                        last_conv=True)

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
                    last_conv=False, first_conv=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QConv1x1(args, self.inplanes, planes * block.expansion, stride, groups=self.groups),
                norm_layer(planes * block.expansion, groups=self.groups),
            )

        layers = []
        layers.append(block(args, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, first_conv=first_conv))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            if i == blocks - 1:
                layers.append(block(args, self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer, last_conv=last_conv))
            else:
                layers.append(block(args, self.inplanes, planes, groups=self.groups,
                                    base_width=self.base_width, dilation=self.dilation,
                                    norm_layer=norm_layer))

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
    groups = args.groups
    model = QResNet(args, block, layers, groups=groups, **kwargs)
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

class MobileNetV1(nn.Module):
    def __init__(self, args, num_classes=1000):
        super(MobileNetV1, self).__init__()

        # setting of inverted residual blocks
        self.block_setting = [
            # c, n, s
            [64, 1, 1],
            [128, 2, 2],
            [256, 2, 2],
            [512, 6, 2],
            [1024, 2, 2],
        ]

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
                                                        channel_shuffle=(i == n//2-1)))
                    elif i == n - 1:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1, last_conv=True,
                                                        channel_shuffle=(i == n//2-1)))
                    else:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                                QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
                                                        channel_shuffle=(i == n//2-1)))
                    channels = outp
                    j += 1
            elif idx == 0:
                for i in range(n):
                    assert n == 1
                    setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s,
                                                    first_conv=True))
                    channels = outp
                    j += 1
            else:
                for i in range(n):
                    if i == 0:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=s,
                                                    channel_shuffle=(i == n//2-1)))
                    elif i == n - 1:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
                                                    channel_shuffle=(i == n//2-1)))
                    else:
                        setattr(self, 'stage_{}_layer_{}'.format(idx, i),
                            QDepthwiseSeparableConv(args, inp=channels, outp=outp, stride=1,
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
    return MobileNetV1(args)
