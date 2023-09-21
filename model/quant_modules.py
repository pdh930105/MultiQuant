import sys
from .quant_ops import QConv, QConv1x1, QConv3x3, QConv_Tra_Mulit, QLinear, SwitchableBatchNorm2d, SwitchableBatchNorm2d_Tra_Mulit
from .shuffle_utils import channel_shuffle
import torch
import torch.nn as nn


class QBasicBlock4Cifar(nn.Module):
    def __init__(self, args, inplanes, planes, stride=1, groups=1, dilation=1,
                 norm_layer=None, last_conv=False, first_conv=False, r8=False):
        super(QBasicBlock4Cifar, self).__init__()

        self.stride = stride
        self.groups = groups
        self.weight_bit = 2
        self.act_bit = 8
        self.last_conv = last_conv
        self.first_conv = first_conv
        self.inplanes = inplanes
        self.r8 = r8
        self.planes = planes
        assert not self.last_conv or not self.first_conv

        if norm_layer is None:
            norm_layer = SwitchableBatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        if first_conv:
            self.conv1 = QConv3x3(args, inplanes, planes, stride, groups=groups,
                                  first_conv=first_conv)
        else:
            self.conv1 = QConv3x3(args, inplanes, planes, stride, groups=groups)

        self.bn1 = norm_layer(planes, groups=groups)
        self.relu = nn.ReLU(inplace=True)

        if last_conv:
            self.conv2 = QConv3x3(args, planes, planes, groups=groups, last_conv=last_conv)
            self.bn2 = norm_layer(planes, groups=groups, last_conv=last_conv)

        else:
            self.conv2 = QConv3x3(args, planes, planes, groups=groups)
            self.bn2 = norm_layer(planes, groups=groups)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if r8:
                self.shortcut = nn.Sequential(
                QConv1x1(args, inplanes, planes, stride, groups=self.groups, last_conv=last_conv),
                norm_layer(planes, groups=groups, last_conv=last_conv),
            )
            else:
                self.shortcut = nn.Sequential(
                        QConv1x1(args, inplanes, planes, stride, groups=self.groups),
                        norm_layer(planes, groups=groups)
                        )

    def select(self, identity):
        assert (self.weight_bit & 1) == 0, "only support 2^k module"
        index = self.inplanes // self.groups * (self.weight_bit // 2)
        return identity[:, :index, :, :]

    def addZeroS(self, identity):
        if (self.weight_bit & 1) == 0:
            return identity

    def forward(self, x):
        identity = x
        print("before shape : ", x.shape)
        if self.first_conv:
            identity = self.select(identity)
        print("after shape : ", identity.shape)

        out = self.conv1(x)
        print("out shape : ", out.shape)
        out = self.bn1(out)

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.last_conv:
            identity = self.addZeroS(identity)
        # out = channel_shuffle(out, self.weight_bit // 2, self.weight_bit)
        out += self.shortcut(identity)
        out = self.relu(out)

        if self.last_conv:
            out = out.chunk((self.weight_bit + 1) // 2, dim=1)
            out = torch.mean(torch.stack(out, dim=0), dim=0)

        return out


class QBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, args, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
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
                                first_conv=first_conv)
        else:
            self.conv1 = QConv3x3(args, inplanes, planes, stride, groups=groups,
                                    )
        self.bn1 = norm_layer(planes, groups=groups)
        self.relu = nn.ReLU(inplace=True)

        if last_conv:
            self.conv2 = QConv3x3(args, planes, planes, groups=groups,
                                  last_conv=last_conv)
            self.bn2 = norm_layer(planes, groups=groups,  last_conv=last_conv)

        else:
            self.conv2 = QConv3x3(args, planes, planes, groups=groups)
            self.bn2 = norm_layer(planes, groups=groups)
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        self.weight_bit = 2
        self.act_bit = 8
        self.last_conv = last_conv
        self.first_conv = first_conv
        self.inplanes = inplanes

    def select(self, identity):
        if (self.weight_bit & 1) == 0:
            index = self.inplanes // self.groups * (self.weight_bit // 2)
            return identity[:, :index, :, :]

    def addZeroS(self, identity):
        if (self.weight_bit & 1) == 0:
            return identity
        else:
            index = self.inplanes // self.groups 
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
                 base_width=64, dilation=1, norm_layer=None,
                 last_conv=False, first_conv=False, shuffle=False): # shuffle not works
        super(QBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = SwitchableBatchNorm2d
        width = int(planes * (base_width / 64.))

        if first_conv:
            self.conv1 = QConv1x1(args, inplanes, width, groups=groups,
                                  first_conv=first_conv)
        else:
            self.conv1 = QConv1x1(args, inplanes, width, groups=groups)
        self.bn1 = norm_layer(width, groups=groups)


        self.conv2 = QConv3x3(args, width, width, stride, groups=groups,
)
        self.bn2 = norm_layer(width, groups=groups)


        if last_conv:
            self.conv3 = QConv1x1(args, width, planes * self.expansion, groups=groups,
                                   last_conv=last_conv)
            self.bn3 = norm_layer(planes * self.expansion, groups=groups, 
                                  last_conv=last_conv)
        else:
            self.conv3 = QConv1x1(args, width, planes * self.expansion, groups=groups,
                               )
            self.bn3 = norm_layer(planes * self.expansion, groups=groups)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.groups = groups
        self.weight_bit = 2
        self.act_bit = 8
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
            index = self.inplanes // self.groups
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


# for mobilenet
class QDepthwiseSeparableConv(nn.Module):
    def __init__(self, args, inp=0, outp=0, stride=0,
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
                            basewidth=self.basewidth,
                            first_conv=first_conv),
            SwitchableBatchNorm2d_Tra_Mulit(inp, basewidth=self.basewidth),
            nn.ReLU(inplace=True),

            QConv(inp, outp, kernel_size=1, stride=1,
                     padding=0, groups=args.groups, bias=False, args=args,
                      last_conv=last_conv),
            SwitchableBatchNorm2d(outp, groups=args.groups),
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
