import torch
import torch.nn as nn
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat
import torch.nn.functional as F
import torch.fft as fft
from PIL import Image
import numpy as np
import random
import math
from einops import repeat
import os,sys
sys.path.insert(0,os.getcwd())
from nets.BaMamba.block_util import FeedForward

class EdgeGradient(nn.Module):
    def __init__(self):
        super(EdgeGradient, self).__init__()
        in_planes = 1
        # TODO? 边界算子
        self.register_buffer('sobel_x',
                             torch.tensor([ [ [ [ -1, 0, 1 ], [ -2, 0, 2 ], [ -1, 0, 1 ] ] ] ]).float() * 0.1)
        self.register_buffer('sobel_y',
                             torch.tensor([ [ [ [ -1, -2, -1 ], [ 0, 0, 0 ], [ 1, 2, 1 ] ] ] ]).float() * 0.1)

        self.register_buffer('laplacian',
                             torch.tensor([ [ [ [ 0, -2, 0 ], [ -1, 4, -1 ], [ 0, -1, 0 ] ] ] ]).float() * 0.1)

        self.conv = nn.Conv2d(in_channels = 3, out_channels = 1, kernel_size = 3, padding = 1)

        self.learned = nn.Parameter(torch.ones(in_planes, ), requires_grad = True)
        self.scale = nn.Parameter(torch.ones(in_planes, ), requires_grad = True)
        self.shift = nn.Parameter(torch.zeros(in_planes, ), requires_grad = True)

    def forward(self, feat):
        inc = feat.shape[ 1 ]
        if inc > 1:
            # TODO?
            feat = feat.mean(dim = 1, keepdim = True)

        edge_x = F.conv2d(feat, self.sobel_x, padding = 1)
        edge_y = F.conv2d(feat, self.sobel_y, padding = 1)
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2)

        laplacian = F.conv2d(feat, self.laplacian, padding = 1)

        delta = torch.abs(laplacian - edge) / self.learned[ None, :, None, None ]
        predict = delta * self.scale[ None, :, None, None ] + self.shift[ None, :, None, None ]

        edgefeat = F.sigmoid(self.conv(torch.cat((edge, laplacian, predict), dim = 1)))
        # TODO?
        return edgefeat


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state = 16,
            d_conv = 3,
            expand = 2.,
            dt_rank = "auto",
            dt_min = 0.001,
            dt_max = 0.1,
            dt_init = "random",
            dt_scale = 1.0,
            dt_init_floor = 1e-4,
            dropout = 0.1,
            conv_bias = True,
            bias = False,
            device = None,
            dtype = None,
            **kwargs,
    ):
        factory_kwargs = {"device": torch.device('cuda'), "dtype": torch.float32}
        super().__init__()
        self.d_model = d_model  #
        self.d_state = d_state

        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias = bias, **factory_kwargs)

        self.conv2d = nn.Conv2d(
            in_channels = self.d_inner,
            out_channels = self.d_inner,
            groups = self.d_inner,
            bias = conv_bias,
            kernel_size = d_conv,
            padding = (d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias = False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias = False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias = False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias = False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([ t.weight for t in self.x_proj ], dim = 0))  # (K=4, N, inner)
        del self.x_proj

        self.K = 4

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([ t.weight for t in self.dt_projs ], dim = 0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([ t.bias for t in self.dt_projs ], dim = 0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies = self.K, merge = True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies = self.K, merge = True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)

        self.edge = EdgeGradient() #TODO?
        self.edge_conv = nn.Linear(1,self.K,bias = False)
        self.scale = nn.Parameter(torch.ones((1,1,self.d_state,1)),requires_grad = True)
        self.shift = nn.Parameter(torch.zeros((1,1,self.d_state,1)),requires_grad = True)

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias = bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale = 1.0, dt_init = "random", dt_min = 0.001, dt_max = 0.1,
                dt_init_floor = 1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias = True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min = dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies = 1, device = None, merge = True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype = torch.float32, device = device),
            "n -> d n",
            d = d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r = copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies = 1, device = None, merge = True):
        # D "skip" parameter
        D = torch.ones(d_inner, device = device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r = copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):

        edge = self.edge(x)

        B, C, H, W = x.shape
        L = H * W
        K = 4
        # TODO? 梯度感知
        x_hwwh = torch.stack([ x.view(B, -1, L), torch.transpose(x, dim0 = 2, dim1 = 3).contiguous().view(B, -1, L) ],
                             dim = 1).view(B, 2, -1, L)

        x1 = torch.cat([ x_hwwh, torch.flip(x_hwwh, dims = [ -1 ]) ],
                       dim = 1)  #

        xs = x1  # TODO?

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [ self.dt_rank, self.d_state, self.d_state ], dim = 2)  #

        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l) #TODO? 边界感知
        *_, hidden, _ = Cs.shape


        weight = edge.view(B, -1, L)
        weight = self.edge_conv(weight.contiguous().permute((0,2,1))).permute((0,2,1))
        weight = weight.float().view(B, K, 1, L).expand((-1,-1,hidden,-1))
        weight = weight * self.scale + self.shift

        Cs = Cs + weight #TODO?


        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)

        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z = None,  # TODO?
            delta_bias = dt_projs_bias,
            delta_softplus = True,
            return_last_state = False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float


        inv_y = torch.flip(out_y[ :, 2:4 ], dims = [ -1 ]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[ :, 1 ].view(B, -1, W, H), dim0 = 2, dim1 = 3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[ :, 1 ].view(B, -1, W, H), dim0 = 2, dim1 = 3).contiguous().view(B, -1, L)

        return out_y[ :, 0 ], inv_y[ :, 0 ], wh_y, invwh_y


    def forward(self, x: torch.Tensor):

        x = x.contiguous().permute((0, 2, 3, 1))
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim = -1)
        x = x.permute(0, 3, 1, 2).contiguous()

        # TODO?
        x = self.act(self.conv2d(x))

        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4  ## Local enhancement
        y = torch.transpose(y, dim0 = 1, dim1 = 2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out.contiguous().permute((0, 3, 1, 2))


class BoundaryMamaba(nn.Module):
    def __init__(self,in_planes = None,num_blocks = 3):
        super(BoundaryMamaba, self).__init__()

        self.scan = nn.ModuleList([])

        for i in range(num_blocks):
            self.scan.append(
                nn.ModuleList(
                    [
                        SS2D(d_model = in_planes),
                        FeedForward(dim = in_planes)
                    ]
                )
            )

    def forward(self,x):
        out = x
        for gradmamna,ffn in self.scan:
            out = gradmamna(out) + out
            out = ffn(out) + out

        return out + x


if __name__ == '__main__':
    x = torch.randn((4, 3, 720, 960)).cuda()


    model = BoundaryMamaba(in_planes = 3).cuda()
    z = model(x)
    print(z.shape)
