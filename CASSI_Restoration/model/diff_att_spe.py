import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numbers
import math
import warnings
# from .basicblock import RCABlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile
from einops import rearrange
from torch import einsum
from torch.nn.init import _calculate_fan_in_and_fan_out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class SPE_NOLCAL(nn.Module):
    def __init__(self, dim, num_heads=8, num_hashes=64, bucket_size=128, attention_dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = 4
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        if self.dim % self.window_size != 0:
            pad_size = self.window_size - (dim % self.window_size)
            self.pos_emb = nn.Parameter(
            torch.Tensor(1, num_heads, int((self.dim+pad_size) / self.window_size), int((self.dim+pad_size) / self.window_size)))
        else:
            self.pos_emb = nn.Parameter(
		    torch.Tensor(1, num_heads, int(self.dim / self.window_size), int(self.dim / self.window_size)))

        trunc_normal_(self.pos_emb)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        # c1 7 b0 4 7组， 四个波段一组
        if c % self.window_size != 0:
            pad_size = self.window_size - (c % self.window_size)
            q, k, v = map(lambda t:torch.cat([t, torch.zeros(b, pad_size, h, w, device=x.device)], dim=1),(q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b (c1 b0) h w -> b c1 (b0 h w)',
                                          b0=self.window_size), (q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        head_dim = h * w * self.window_size / self.num_heads
        q = q * (head_dim ** -0.5)
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, 'b (c1) (b0 h w) -> b (c1 b0) h w', h=h, w=w, b0=self.window_size)
        if c != self.dim:
            out = out[:, :self.dim, :, :]
        out = self.project_out(out)


        return out


class SPE_LOCAL(nn.Module):
    def __init__(self, dim, num_heads=8, num_hashes=64, bucket_size=128, attention_dropout=0.1):
        super().__init__()
        self.dim = dim
        self.window_size = 4
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        if self.dim % self.window_size != 0:
            pad_size = self.window_size - (dim % self.window_size)
            self.pos_emb = nn.Parameter(
            torch.Tensor(1, num_heads, int((self.dim+pad_size) / self.window_size), int((self.dim+pad_size) / self.window_size)))
        else:
            self.pos_emb = nn.Parameter(
		    torch.Tensor(1, num_heads, int(self.dim / self.window_size), int(self.dim / self.window_size)))

        trunc_normal_(self.pos_emb)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)
        # c1 7 b0 4 7组， 四个波段一组
        if c % self.window_size != 0:
            pad_size = self.window_size - (c % self.window_size)
            q, k, v = map(lambda t:torch.cat([t, torch.zeros(b, pad_size, h, w, device=x.device)], dim=1),(q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b (c1 b0) h w -> (b b0) c1 ( h w)',
                                          b0=self.window_size), (q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))
        head_dim = h * w * self.window_size / self.num_heads
        q = q * (head_dim ** -0.5)
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, '(b b0) c1 (h w) -> b (c1 b0) h w', h=h, w=w, b0=self.window_size)
        if c != self.dim:
            out = out[:, :self.dim, :, :]
        out = self.project_out(out)


        return out

class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor=2.66
                 ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=True)

        self.act_fn = nn.GELU()

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=False)

    def forward(self, x):
        """
        x: [b, c, h, w]
        return out: [b, c, h, w]
        """
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = self.act_fn(x1) * x2
        x = self.project_out(x)
        return x

class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b, h, w, c]
        return out: [b, h, w, c]
        """
        out = self.net(x)
        return out

def FFN_FN(

        ffn_name,
        dim
):
    if ffn_name == "Gated_Dconv_FeedForward":
        return Gated_Dconv_FeedForward(
            dim,
            ffn_expansion_factor=2
        )
    elif ffn_name == "FeedForward":
        return FeedForward(dim=dim)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (b, c, h, w)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class PreNorm(nn.Module):
    def __init__(self, dim, fn, layernorm_type='WithBias'):
        super().__init__()
        self.fn = fn
        self.layernorm_type = layernorm_type
        if layernorm_type == 'BiasFree' or layernorm_type == 'WithBias':
            self.norm = LayerNorm(dim, layernorm_type)
        else:
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        if self.layernorm_type == 'BiasFree' or self.layernorm_type == 'WithBias':
            x = self.norm(x)
        else:
            h, w = x.shape[-2:]
            x = to_4d(self.norm(to_3d(x)), h, w)
        return self.fn(x, *args, **kwargs)

class SpeBlock(nn.Module):
    """
    The Local and Non-Local Transformer Block (LNLB) is the most important component. Each LNLB consists of three layer-normalizations (LNs), a Local MSA, a Non-Local MSA, and a GDFN (Zamir et al. 2022).
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size: tuple,
                 window_num: tuple,
                 layernorm_type,
                 num_blocks,
                 ):
        super().__init__()

        self.window_size = window_size
        self.window_num = window_num
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                SPE_LOCAL(
                    dim=dim,
                    num_heads=num_heads,

                ),
                SPE_NOLCAL(
                    dim=dim,
                    num_heads=num_heads,

                ),
               FFN_FN(
                    ffn_name="Gated_Dconv_FeedForward",
                    dim=dim
                ),
                FFN_FN(
                    ffn_name="Gated_Dconv_FeedForward",
                    dim=dim
                )
            ]))

    def forward(self, x):
        for (Spe_b1, Spe_b2, ffn1, ffn2) in self.blocks:
            x = x + Spe_b1(x)
            x = x + ffn1(x)
            x = x + Spe_b2(x)
            x = x + ffn2(x)

        return x

# class SpeBlock(nn.Module):
#     def __init__(self, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True, mode='CRC',
#                  reduction=16, negative_slope=0.2):
#         super(RCABlock, self).__init__()
#         assert in_channels == out_channels, 'Only support in_channels==out_channels.'
#         if mode[0] in ['R', 'L']:
#             mode = mode[0].lower() + mode[1:]
#
#         self.res = nn.Sequential(nn.Conv2d(in_channels))
#         self.sape_l = SPE_LOCAL(out_channels, num_heads=1)
#         self.sape_nl = SPE_NOLCAL(out_channels, num_heads=1)
#
#     def forward(self, x):
#         res = self.res(x)
#         spa_x = self.sape_l(res)
#         spe_x = self.sape_nl(res)
#
#         return x + spe_x + spa_x

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
        self.down_rb = SpeBlock(nd,1, (8,8),(8,8),"Gated_Dconv_FeedForward",1,)
        self.down_rb1 = SpeBlock(nd,1, (8,8),(8,8),"Gated_Dconv_FeedForward",1,)
        self.down_rb2 = SpeBlock(nd,1, (8,8),(8,8),"Gated_Dconv_FeedForward",1,)
        self.down_rb3 = SpeBlock(nd,1, (8,8),(8,8),"Gated_Dconv_FeedForward",1,)
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

        # x = self.down_rb1(x)
        # x = self.bn4(x)
        # x = self.relu4(x)
        # x = self.down_rb2(x)
        # x = self.down_rb3(x)
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
        self.sk_rb = SpeBlock(ns,1, (8,8),(8,8),"Gated_Dconv_FeedForward",1,)
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
        self.up_rb = SpeBlock(nu,1, (8,8),(8,8),"Gated_Dconv_FeedForward",1,)
        self.up_rb1 = SpeBlock(nu,1, (8,8),(8,8),"Gated_Dconv_FeedForward",1,)
        self.relu = nn.ELU()
        self.relu2 = nn.ELU()
        self.relu3 = nn.ELU()
        self.relu4 = nn.ELU()
        self.up_rb1 = SpeBlock(nu,1, (8,8),(8,8),"Gated_Dconv_FeedForward",1,)

    def forward(self, x):
        # x = self.padder(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.up_rb(x)
        # x = self.bn2(x)
        # x = self.relu2(x)
        x = self.up_rb1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        return x


class Spe_Model(nn.Module):
    def __init__(self, length=4, in_channels=28, nu=[64, 64, 64, 64], nd=
    [64, 64, 64, 64], ns=[16, 16, 16, 16], ku=[1, 1, 1, 1], kd=[1, 1, 1, 1], ks=[1, 1, 1, 1]):
        super(Spe_Model, self).__init__()
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
    model = Spe_Model(in_channels=31)
    input1 = torch.randn([5, 31, 256, 256])
    flops, params = profile(model, (input1,))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')