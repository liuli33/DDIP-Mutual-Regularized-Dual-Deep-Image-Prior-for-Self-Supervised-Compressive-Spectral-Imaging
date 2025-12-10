import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
from .basicblock import RCABlock
import torch
import torch.nn as nn
import torch.nn.functional as F
# import tensorly as tl
# from tensorly.decomposition import tucker
from thop import profile


class Model_Down(nn.Module):
    """
    Convolutional (Downsampling) Blocks.

    nd = Number of Filters
    kd = Kernel size

    """

    def __init__(self, in_channels, nd=128, kd=3, padding=1, stride=2):
        super(Model_Down, self).__init__()
        self.padder = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nd, kernel_size=kd, stride=stride, bias=True)
        self.bn1 = nn.BatchNorm2d(nd)
        self.conv2 = nn.Conv2d(in_channels=nd, out_channels=nd, kernel_size=1, stride=1, bias=True)
        self.bn2 = nn.BatchNorm2d(nd)
        self.relu = nn.ELU()
        self.bn3 = nn.BatchNorm2d(nd)
        self.down_rb = RCABlock(nd, nd)
        self.down_rb1 = RCABlock(nd, nd)
        self.down_rb2 = RCABlock(nd, nd)
        self.down_rb3 = RCABlock(nd, nd)
        self.relu2 = nn.ELU()
        self.relu3 = nn.ELU()
        self.relu4 = nn.ELU()
        self.bn4 = nn.BatchNorm2d(nd)
    def forward(self, x):
        # x = self.padder(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.down_rb(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.down_rb1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.down_rb2(x)
        x = self.down_rb3(x)
        return x


class Model_Skip(nn.Module):
    """

    Skip Connections

    ns = Number of filters
    ks = Kernel size

    """

    def __init__(self, in_channels=128, ns=4, ks=1, padding=0, stride=1):
        super(Model_Skip, self).__init__()
        self.padder = nn.ReflectionPad2d(padding)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=ns, kernel_size=ks, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(ns)
        self.relu = nn.ELU()
        self.sk_rb = RCABlock(ns, ns)
        self.bn3 = nn.BatchNorm2d(ns)
        self.relu2 = nn.ELU()


    def forward(self, x):
        # x = self.padder(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = self.sk_rb(x)
        # x = self.bn3(x)
        # x = self.relu2(x)
        return x


class Model_Up(nn.Module):
    """
    Convolutional (Downsampling) Blocks.

    nd = Number of Filters
    kd = Kernel size

    """

    def __init__(self, in_channels=132, nu=128, ku=3, padding=1):
        super(Model_Up, self).__init__()
        self.bn1 = nn.BatchNorm2d(nu)
        self.padder = nn.ReflectionPad2d(padding)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=nu, kernel_size=ku, stride=1, padding=0, bias=True)
        self.bn2 = nn.BatchNorm2d(nu)

        self.conv2 = nn.Conv2d(in_channels=nu, out_channels=nu, kernel_size=1, stride=1,
                               padding=0, bias=True)  # According to supmat.pdf ku = 1 for second layer
        self.bn3 = nn.BatchNorm2d(nu)
        self.bn4 = nn.BatchNorm2d(nu)
        self.up_rb = RCABlock(nu, nu)
        self.up_rb1 = RCABlock(nu, nu)
        self.relu = nn.ELU()
        self.relu2 = nn.ELU()
        self.relu3 = nn.ELU()
        self.relu4 = nn.ELU()
        self.up_rb1 = RCABlock(nu, nu)

    def forward(self, x):
        # x = self.padder(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.up_rb(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.up_rb1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        return x


class Model(nn.Module):
    def __init__(self, length=4, in_channels=28, nu=[64, 64, 64, 64], nd=
    [64, 64, 64, 64], ns=[16, 16, 16, 16], ku=[1, 1, 1, 1], kd=[1, 1, 1, 1], ks=[1, 1, 1, 1]):
        super(Model, self).__init__()
        assert length == len(nu), 'Hyperparameters do not match network depth.'

        self.length = length

        self.downs = nn.ModuleList([Model_Down(in_channels=nd[i - 1], nd=nd[i], kd=kd[i]) if i != 0 else
                                    Model_Down(in_channels=in_channels, nd=nd[i], kd=kd[i]) for i in
                                    range(self.length)])

        self.skips = nn.ModuleList([Model_Skip(in_channels=nd[i], ns=ns[i], ks=ks[i]) if i != 0 else
                                    Model_Skip(in_channels=in_channels, ns=ns[i], ks=ks[i]) for i in
                                    range(self.length)])

        self.ups = nn.ModuleList(
            [Model_Up(in_channels=ns[i] + nu[i], nu=nu[i], ku=ku[i])  for i in
             range(self.length - 1, -1, -1)])  # Elements ordered backwards
        self.up_sample = [nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Upsample(scale_factor=2, mode='bilinear')
                          ]


        self.conv_out = nn.Conv2d(nu[0], in_channels, 1, padding=0)

        def initialize(self):  # 初始化模型参数
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data)



    def forward(self, x):
        s = []  # Skip Activations
        # Downpass
        for i in range(self.length):
            s.append(self.skips[i].forward(x))
            x = self.downs[i].forward(x)

        # Uppass
        for i in range(self.length):
            x = self.up_sample[i](x)
            x = self.ups[i].forward(torch.cat([x, s[self.length - i-1]], axis=1))
        x = self.conv_out(x)  # Squash to
        return x


class Modelreal(nn.Module):
    def __init__(self, length=3, in_channels=28, nu=[128, 128, 128], nd=
    [128, 128, 128], ns=[128, 128, 128], ku=[1, 1, 1], kd=[1, 1, 1], ks=[1, 1, 1]):
        super(Modelreal, self).__init__()
        assert length == len(nu), 'Hyperparameters do not match network depth.'

        self.length = length

        self.downs = nn.ModuleList([Model_Down(in_channels=nd[i - 1], nd=nd[i], kd=kd[i]) if i != 0 else
                                    Model_Down(in_channels=in_channels, nd=nd[i], kd=kd[i]) for i in
                                    range(self.length)])

        self.skips = nn.ModuleList([Model_Skip(in_channels=nd[i], ns=ns[i], ks=ks[i]) if i != 0 else
                                    Model_Skip(in_channels=in_channels, ns=ns[i], ks=ks[i]) for i in
                                    range(self.length)])

        self.ups = nn.ModuleList(
            [Model_Up(in_channels=ns[i] + nu[i], nu=nu[i], ku=ku[i])  for i in
             range(self.length - 1, -1, -1)])  # Elements ordered backwards
        self.up_sample = [nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Upsample(scale_factor=2, mode='bilinear'),
                                       nn.Upsample(scale_factor=2, mode='bilinear')
                          ]


        self.conv_out = nn.Conv2d(nu[0], in_channels, 1, padding=0)

        def initialize(self):  # 初始化模型参数
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight.data)



    def forward(self, x):
        s = []  # Skip Activations
        # Downpass
        for i in range(self.length):
            s.append(self.skips[i].forward(x))
            x = self.downs[i].forward(x)

        # Uppass
        for i in range(self.length):
            x = self.up_sample[i](x)
            x = self.ups[i].forward(torch.cat([x, s[self.length - i-1]], axis=1))
        x = self.conv_out(x)  # Squash to
        return x
if __name__ == '__main__':
    model = Model(in_channels=31)
    input1 = torch.randn([5, 31, 256, 256])
    flops, params = profile(model, (input1,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')