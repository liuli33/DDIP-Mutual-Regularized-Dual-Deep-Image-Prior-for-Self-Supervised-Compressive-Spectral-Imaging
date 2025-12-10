import torch
from torch import nn, optim

dtype = torch.cuda.FloatTensor
import numpy as np
# from skimage.restoration import denoise _nl_means
import matplotlib.pyplot as plt
import scipy.io
import math
from einops import rearrange
from torch import einsum
from skimage.metrics import peak_signal_noise_ratio
import warnings
from torch.nn import functional as F
import numbers
# data_all =["data/om1"]
# c_all = ["case2"]
#
# ###################
# # Here are the hyperparameters.
# max_iter = 5001
# w_decay = 0.1
# lr_real = 0.0001
# phi = 5*10e-6
# mu = 1.2
# gamma = 0.1
# down = 4
# omega = 2
# ###################
from .skip2 import Model
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

#
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

class LocalMSA(nn.Module):
    """
    The Local MSA partitions the input into non-overlapping windows of size M × M, treating each pixel within the window as a token, and computes self-attention within the window.
    """

    def __init__(self,
                 dim,
                 num_heads,
                 window_size,
                 ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

        self.pos_emb = nn.Parameter(
            torch.Tensor(1, num_heads, window_size * window_size, window_size * window_size))
        trunc_normal_(self.pos_emb)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        b, c, h, w = x.shape

        q, k, v = self.qkv_dwconv(self.qkv(x)).chunk(3, dim=1)

        # q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1)-> b (h w) (b0 b1 c)',
        #                                   h=self.window_num[0], w=self.window_num[1]), (q, k, v))
        #
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b c (h b0) (w b1) -> (b h w) (b0 b1) c',
                                          b0=self.window_size, b1=self.window_size), (q, k, v))

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), (q, k, v))

        q *= self.scale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out = rearrange(out, '(b h w) (b0 b1) c -> b c (h b0) (w b1)', h=h // self.window_size,
                        w=w // self.window_size,
                        b0=self.window_size)
        out = self.project_out(out)

        return out

class soft(nn.Module):
    def __init__(self):
        super(soft, self).__init__()

    def forward(self, x, lam):
        x_abs = x.abs() - lam
        zeros = x_abs - x_abs
        n_sub = torch.max(x_abs, zeros)
        x_out = torch.mul(torch.sign(x), n_sub)
        return x_out


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=0.5):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear_omega = nn.Linear(in_features, 1, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(5 / self.in_features) / self.omega_0,
                                            np.sqrt(5 / self.in_features) / self.omega_0)

    def forward(self, input):
        self.omega_1 = self.linear(input)
        return torch.sin(self.omega_1 * self.linear(input))
        # 23.65

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

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
        return x / torch.sqrt(sigma+1e-5) * self.weight
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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
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


class Liner_att(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=0.5):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear_om = nn.Linear(in_features, in_features, bias=bias)
        self.linear_att = nn.Linear(in_features, out_features, bias=bias)
        # self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
                self.linear_att.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
                self.linear_om.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(5 / self.in_features) / self.omega_0,
                                            np.sqrt(5 / self.in_features) / self.omega_0)
                self.linear_att.weight.uniform_(-np.sqrt(5 / self.in_features) / self.omega_0,
                                            np.sqrt(5 / self.in_features) / self.omega_0)
                self.linear_om.weight.uniform_(-np.sqrt(5 / self.in_features) / self.omega_0,
                                            np.sqrt(5 / self.in_features) / self.omega_0)
    def forward(self, input):
        self.omega_1 = self.linear(input)
        return torch.sin(self.omega_1*self.linear_att(input))
        # 23.65
    # def forward(self, input):
    #     self.omega_1 = self.linear_om(input)
    #     self.wei = self.linear_att(input)
    #     self.att_out = self.linear(input)
    #     return torch.sin(self.omega_0 *self.linear(input))
class Gated_Dconv_FeedForward(nn.Module):
    def __init__(self,
                 dim,
                 ffn_expansion_factor = 2.66
    ):
        super(Gated_Dconv_FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=True)

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
                ffn_expansion_factor=2.66,
            )
    elif ffn_name == "FeedForward":
        return FeedForward(dim = dim)

class Network(nn.Module):
    def __init__(self, r_1, r_2, r_3, mid_channel):
        super(Network, self).__init__()

        self.U_net = nn.Sequential(Liner_att(28, mid_channel, is_first=True),
                                   Liner_att(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_1))

        self.V_net = nn.Sequential(Liner_att(28, mid_channel, is_first=True),
                                   Liner_att(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_2))

        self.W_net = nn.Sequential(Liner_att(28, mid_channel, is_first=True),
                                   Liner_att(mid_channel, mid_channel, is_first=True),
                                   nn.Linear(mid_channel, r_3))
        self.center_model = Model(in_channels=12)
        self.ouer_lineer =nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.ouer_lineer1 = nn.Conv2d(28, 28, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList([])
        num_blocks = 2
        dim = 28
        window_size = 8
        num_heads = 4
        layernorm_type = "WithBias"
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, LocalMSA(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                ),
                        layernorm_type=layernorm_type) ,
                PreNorm(dim, LocalMSA(
                    dim=dim,
                    window_size=window_size,
                    num_heads=num_heads,
                ),
                        layernorm_type=layernorm_type),
                PreNorm(dim, FFN_FN(
                    ffn_name="Gated_Dconv_FeedForward",
                    dim=dim
                ),
                        layernorm_type=layernorm_type)
            ]))
        # self.ouer_lineer2 = nn.Conv2d(28, 28, kernel_size=3, padding=1)

    def forward(self, centre, U_input, V_input, W_input):
        U = self.U_net(U_input)
        V = self.V_net(V_input)
        W = self.W_net(W_input)
        centre = self.ouer_lineer(centre.unsqueeze(0)).squeeze(0)
        centre = centre.permute(1, 2, 0)
        centre = centre @ U.t()
        centre = centre.permute(2, 1, 0)
        centre = centre @ V.t()
        centre = centre.permute(0, 2, 1)
        centre = centre @ W.t()
        out = centre.unsqueeze(0)
        # for (local_msa, nonlocal_msa, ffn) in self.blocks:
        #     x = out + local_msa(out)
        #     x = x + nonlocal_msa(x)
        #     x = x + ffn(x)
        return out.squeeze(0)

