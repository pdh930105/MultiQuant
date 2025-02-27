import torch
import torch.nn as nn 
from .quant_ops import QConv1x1

def cyclic_permute(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    index_range=list(range(1,groups))+[0]
    index=torch.tensor(index_range)
    x = x[:,index,:,:, :]
    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


def channel_shuffle(x, groups, weight_bit=-1):
    batchsize, num_channels, height, width = x.size()
    if weight_bit == 7:
        if num_channels % 4 == 0:
            groups = 4
        else:
            groups = 3

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


def channel_shuffle_mv1(x, groups, weight_bit=-1):
    batchsize, num_channels, height, width = x.size()
    if weight_bit == 5 or weight_bit == 7:
        if num_channels % 4 == 0:
            groups = 4
        else:
            groups = 3

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class CyclicShuffle(nn.Module):
    def __init__(self, args, inplanes,  groups=1):
        super(CyclicShuffle, self).__init__()
       
        self.inplanes=inplanes
        self.groups=groups
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1)
        self.conv1=nn.Sequential(
                QConv1x1(args, self.inplanes, self.inplanes, args.QWeightFlag, args.QActFlag, args.weight_levels, 
                        args.act_levels, stride=1, groups=self.groups),
                nn.BatchNorm2d(self.inplanes),
                nn.ReLU(),
            )
    
    def forward(self, x):
        identity = x
        out = cyclic_permute(x,self.groups)
        out=self.conv1(out)
        out += identity
        return out