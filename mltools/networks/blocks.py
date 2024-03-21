import torch
import torch.nn as nn

from mltools.networks.network_tools import get_conv, zero_init


class AttnBlock(nn.Module):
    def __init__(self, in_channels, n_heads=4, dim=2, **kwargs):
        super().__init__()
        self.in_channels = in_channels
        assert (
            self.in_channels % n_heads == 0
        ), "in_channels must be divisible by n_heads"
        self.n_heads = n_heads
        self.dim = dim
        assert self.dim == 2 or self.dim == 3, "dim must be 2 or 3"

        norm_params = kwargs.get("norm_params", {})

        self.norm = nn.GroupNorm(num_channels=in_channels, **norm_params)

        self.q = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.k = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.v = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = get_conv(
            in_channels, in_channels, dim=self.dim, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        if self.dim == 2:
            b, c, h, w = q.shape
            c_ = c // self.n_heads
            q = q.reshape(b, c_, self.n_heads, h * w)
            k = k.reshape(b, c_, self.n_heads, h * w)
            w_ = torch.einsum("bcnq,bcnk->bqkn", q, k)
            w_ = w_ * (int(c_) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b, c_, self.n_heads, h * w)
            h_ = torch.einsum("bcnd,bqdn->bcnq", v, w_)
            h_ = h_.reshape(b, c, h, w)
            h_ = self.proj_out(h_)
        elif self.dim == 3:
            b, c, d, h, w = q.shape
            c_ = c // self.n_heads
            q = q.reshape(b, c_, self.n_heads, d * h * w)
            k = k.reshape(b, c_, self.n_heads, d * h * w)
            w_ = torch.einsum("bcnq,bcnk->bqkn", q, k)
            w_ = w_ * (int(c_) ** (-0.5))
            w_ = torch.nn.functional.softmax(w_, dim=2)
            v = v.reshape(b, c_, self.n_heads, d * h * w)
            h_ = torch.einsum("bcnd,bqdn->bcnq", v, w_)
            h_ = h_.reshape(b, c, d, h, w)
            h_ = self.proj_out(h_)
        return x + h_


class ResNetBlock(nn.Module):
    def __init__(
        self,
        ch_in,
        ch_out,
        dim=2,
        conditioning_dims=None,
        dropout_prob=0.0,
        nca_params={},
    ):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.dim = dim
        assert self.dim in [2, 3], "dim must be 2 or 3"
        self.conditioning_dims = conditioning_dims

        self.nca_params = nca_params
        norm_params = self.nca_params.get("norm_params", {})
        get_act = self.nca_params.get("get_act", lambda: nn.GELU())
        conv_params = self.nca_params.get("conv_params", {})

        self.net1 = nn.Sequential(
            nn.GroupNorm(num_channels=ch_in, **norm_params),
            get_act(),
            get_conv(ch_in, ch_out, dim=self.dim, **conv_params),
        )
        if conditioning_dims is not None:
            self.cond_projs = nn.ModuleList()
            for condition_dim in self.conditioning_dims:
                self.cond_projs.append(zero_init(nn.Linear(condition_dim, ch_out)))
        self.net2 = nn.Sequential(
            nn.GroupNorm(num_channels=ch_out, **norm_params),
            get_act(),
            *([nn.Dropout(dropout_prob)] * (dropout_prob > 0.0)),
            get_conv(ch_out, ch_out, dim=self.dim, init=zero_init, **conv_params),
        )
        if ch_in != ch_out:
            self.skip_conv = get_conv(
                ch_in, ch_out, dim=self.dim, kernel_size=1, padding=0
            )

    def forward(self, x, conditionings=None):
        h = self.net1(x)
        if conditionings is not None:
            assert len(conditionings) == len(self.conditioning_dims)
            assert all(
                [
                    conditionings[i].shape
                    == (x.shape[0], self.conditioning_dims[i])
                    for i in range(len(conditionings))
                ]
            )
            for i, conditioning in enumerate(conditionings):
                conditioning_ = self.cond_projs[i](conditioning)
                if self.dim == 2:
                    h = h + conditioning_[:, :, None, None]
                elif self.dim == 3:
                    h = h + conditioning_[:, :, None, None, None]
        h = self.net2(h)
        if x.shape[1] != self.ch_out:
            x = self.skip_conv(x)
        return x + h


class ResNetDown(nn.Module):
    def __init__(self, resnet_blocks, attention_blocks=None):
        super().__init__()
        self.resnet_blocks = resnet_blocks
        self.attention_blocks = attention_blocks
        self.dim = self.resnet_blocks[-1].dim
        self.down = get_conv(
            self.resnet_blocks[-1].ch_out,
            self.resnet_blocks[-1].ch_out,
            dim=self.dim,
            kernel_size=2,
            stride=2,
            padding=0,
        )

    def forward(self, x, conditionings, no_down=False):
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, conditionings)
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
        if no_down:
            return x, None
        x_skip = x
        x = self.down(x)
        return x, x_skip


class ResNetUp(nn.Module):
    def __init__(
        self, resnet_blocks, attention_blocks=None, ch_out=None, conv_params={}
    ):
        super().__init__()
        self.resnet_blocks = resnet_blocks
        self.ch_out = ch_out if ch_out is not None else self.resnet_blocks[-1].ch_out
        self.attention_blocks = attention_blocks
        self.dim = self.resnet_blocks[-1].dim
        self.up = get_conv(
            self.resnet_blocks[-1].ch_out,
            self.ch_out,
            dim=self.dim,
            kernel_size=2,
            stride=2,
            padding=0,
            transposed=True,
        )

    def forward(self, x, x_skip=None, conditionings=None, no_up=False):
        for i, resnet_block in enumerate(self.resnet_blocks):
            x = resnet_block(x, conditionings)
            if self.attention_blocks is not None:
                x = self.attention_blocks[i](x)
        if not no_up:
            x = self.up(x)
        if x_skip is not None:
            x = torch.cat([x, x_skip], dim=1)
        return x
