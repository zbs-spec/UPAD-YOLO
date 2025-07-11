import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class UDMConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=None, dilation=(1, 2, 3), groups=1, padding_mode='zeros', bias=False):
        super(UDMConv, self).__init__()

        if padding is None:
            if isinstance(kernel_size, int):
                padding = kernel_size // 2
            elif isinstance(kernel_size, tuple):
                padding = tuple(k // 2 for k in kernel_size)
            else:
                raise ValueError("kernel_size must be an int or a tuple of ints")

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation[0],
                               groups=in_channels, bias=False)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation[1],
                               groups=in_channels, bias=False)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation[2],
                               groups=in_channels, bias=False)

        self.alpha1 = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.alpha2 = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.alpha3 = nn.Parameter(torch.ones(1, in_channels, 1, 1))

        init.xavier_uniform_(self.alpha1)
        init.xavier_uniform_(self.alpha2)
        init.xavier_uniform_(self.alpha3)

        self.conv_point = nn.Conv2d(in_channels * 3, out_channels, 1, stride=1, padding=0, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x) * self.alpha1
        x2 = self.conv2(x) * self.alpha2
        x3 = self.conv3(x) * self.alpha3

        x2 = F.interpolate(x2, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x3 = F.interpolate(x3, size=x1.shape[2:], mode='bilinear', align_corners=False)

        x = torch.cat([x1, x2, x3], dim=1)

        x = self.conv_point(x)

        x = self.bn(x)
        x = self.act(x)

        return x
