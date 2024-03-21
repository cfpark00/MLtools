import torch
from torch import nn
import warnings

from mltools.networks.network_tools import zero_init, get_conv, get_timestep_embedding
from mltools.networks.blocks import AttnBlock, ResNetBlock, ResNetDown, ResNetUp

class CUNet(nn.Module):
    def __init__(
        self,
        shape=(1, 256, 256),
        chs=[48, 96, 192, 384],
        s_conditioning_channels: int = 0,
        v_conditioning_dims: list = [],
        v_embedding_dim: int = 64,
        t_conditioning=False,
        t_embedding_dim=64,
        norm_groups: int = 8,
        n_blocks: int = 4,
        mid_attn=True,
        n_attention_heads: int = 4,
        dropout_prob: float = 0.1,
        conv_padding_mode: str = "zeros",
        verbose: int = 0,
    ):
        super().__init__()
        self.shape=shape
        self.chs=chs
        self.dim=len(self.shape)-1
        self.in_channels = self.shape[0]
        self.s_conditioning_channels = s_conditioning_channels
        self.v_conditioning_dims = v_conditioning_dims
        self.v_embedding_dim = v_embedding_dim
        self.t_conditioning = t_conditioning
        self.t_embedding_dim = t_embedding_dim
        self.norm_groups = norm_groups
        self.mid_attn = mid_attn
        if self.mid_attn and self.dim ==3:
            raise ValueError("3D attention very highly discouraged.")
        self.n_attention_heads = n_attention_heads
        self.dropout_prob = dropout_prob
        self.verbose = verbose

        if self.t_conditioning:
            self.t_conditioning_dim = int(4 * self.t_embedding_dim)
            self.embed_t_conditioning = nn.Sequential(
                nn.Linear(self.t_embedding_dim, self.t_conditioning_dim),
                nn.GELU(),
                nn.Linear(self.t_conditioning_dim,self.t_conditioning_dim),
                nn.GELU(),
                )
        if len(self.v_conditioning_dims)>0:
            self.embeds_v_conditionings =nn.ModuleList()
            for v_conditioning_dim in self.v_conditioning_dims:
                self.embeds_v_conditionings.append(nn.Sequential(
                    nn.Linear(v_conditioning_dim, self.v_embedding_dim),
                    nn.GELU(),
                    nn.Linear(self.v_embedding_dim, self.v_embedding_dim),
                    nn.GELU(),
                    ))
        conditioning_dims=[]
        if self.t_conditioning:
            conditioning_dims.append(self.t_conditioning_dim)
        for _ in self.v_conditioning_dims:
            conditioning_dims.append(self.v_embedding_dim)
        if len(conditioning_dims)==0:
            conditioning_dims=None
        self.conditioning_dims=conditioning_dims
    

        self.conv_kernel_size=3
        self.norm_eps=1e-6
        self.norm_affine=True
        self.act="gelu"
        self.num_res_blocks = 1
        assert self.conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        norm_params = dict(num_groups=self.norm_groups, eps=self.norm_eps, affine=self.norm_affine)
        assert self.act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"
        def get_act():
            if self.act == "gelu":
                return nn.GELU()
            elif self.act == "relu":
                return nn.ReLU()
            elif self.act == "silu":
                return nn.SiLU()
        padding = self.conv_kernel_size // 2
        conv_params = dict(
            kernel_size=self.conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=self.conditioning_dims,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
        )

        self.n_sizes = len(self.chs)
        self.conv_in = get_conv(
            self.in_channels+self.s_conditioning_channels, self.chs[0], dim=self.dim, **conv_params
        )

        #down
        self.downs = nn.ModuleList()
        for i_level in range(self.n_sizes):
            ch_in = chs[0] if i_level == 0 else chs[i_level - 1]
            ch_out = chs[i_level]
            resnets = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_out, **resnet_params))
                ch_in = ch_out
            down = ResNetDown(resnets)
            self.downs.append(down)

        # middle
        self.mid1 = ResNetBlock(ch_out, ch_out, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_out,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        # when no pad 1x262x262
        self.mid2 = ResNetBlock(ch_out, ch_out, **resnet_params)

        # upsampling
        self.ups = nn.ModuleList()
        ch_skip=0
        for i_level in reversed(range(self.n_sizes)):
            ch_in = self.chs[i_level]
            ch_out = self.chs[0] if i_level == 0 else self.chs[i_level - 1]  # for up
            resnets = nn.ModuleList()
            for i_resnet in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in+(ch_skip if i_resnet==0 else 0), ch_in, **resnet_params))
            up = ResNetUp(resnet_blocks=resnets,ch_out=ch_out)
            ch_skip=ch_out
            self.ups.append(up)

        self.norm_out = nn.GroupNorm(num_channels=ch_out, **norm_params)
        self.act_out = get_act()
        self.conv_out = get_conv(
            in_channels=ch_out,
            out_channels=self.in_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )


    def forward(self, x, t=None, s_conditioning=None, v_conditionings=None):
        if s_conditioning is not None:
            if self.s_conditioning_channels != s_conditioning.shape[1]:
                raise ValueError(
                    f"Expected s_conditioning to have {self.s_conditioning_channels} channels, but got {s_conditioning.shape[1]}"
                )
            x_concat = torch.concat(
                (x, s_conditioning),
                axis=1,
            )
        else:
            x_concat = x

        conditionings = []
        if t is not None:
            if not self.t_conditioning:
                raise ValueError("t is not None, but t_conditioning is False")
            t=t.expand(x_concat.shape[0]).clone()#this clone has to be done for the t_embedding step
            assert t.shape == (x_concat.shape[0],)

            t_embedding = get_timestep_embedding(
                t, self.t_embedding_dim
            )
        
            t_cond = self.embed_t_conditioning(t_embedding)
            conditionings.append(t_cond)
        else:
            assert not self.t_conditioning, "t is None, but t_conditioning is True"

        if v_conditionings is not None:
            if len(v_conditionings) != len(self.v_conditioning_dims):
                raise ValueError(
                    f"Expected {len(self.v_conditioning_dims)} v_conditionings, but got {len(v_conditionings)}"
                )
            for i, v_conditioning in enumerate(v_conditionings):
                if v_conditioning.shape[1] != self.v_conditioning_dims[i]:
                    raise ValueError(
                        f"Expected v_conditioning to have {self.v_conditioning_dims[i]} channels, but got {v_conditioning.shape[1]}"
                    )
                v_cond = self.embeds_v_conditionings[i](v_conditioning)
                conditionings.append(v_cond)
        
        if len(conditionings) == 0:
            conditionings = None

        h = x_concat  # (B, C, H, W, D)

        h = self.conv_in(x_concat)
        #print(h.shape)
        skips=[]
        for i, down in enumerate(self.downs):
            h, h_skip = down(
                h, conditionings=conditionings, no_down=(i == (len(self.downs) - 1))
            )
            #print(i,h.shape)
            if h_skip is not None:
                skips.append(h_skip)
        #print("total skips:",len(skips),[skip.shape for skip in skips])

        # middle
        h = self.mid1(h, conditionings=conditionings)
        #print("m1",h.shape)
        if self.mid_attn:
            h = self.mid_attn1(h)
            #print("ma1",h.shape)
        h = self.mid2(h, conditionings=conditionings)
        #print("m2",h.shape)

        # upsampling
        for i, up in enumerate(self.ups):
            x_skip = skips.pop() if len(skips) > 0 else None
            h = up(h, x_skip=x_skip, conditionings=conditionings, no_up=(i == self.n_sizes - 1))
            #print(i,h.shape)
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h+x


class Encoder(nn.Module):
    def __init__(
        self,
        shape,
        chs=[48, 96, 192],
        attn_sizes=[],
        mid_attn=False,
        num_res_blocks=1,
        dropout_prob=0.0,
        z_channels=4,
        double_z=True,
        n_attention_heads=1,
        norm_groups=8,
        norm_eps=1e-6,
        norm_affine=True,
        act="gelu",
        conv_kernel_size=3,
        conv_padding_mode="zeros",
    ):
        super().__init__()
        self.shape=shape
        self.in_channels = self.shape[0]
        assert self.shape[1] == self.shape[2], "input must be square"
        self.input_size = self.shape[1]
        self.chs = chs
        self.dim = len(self.shape) - 1
        self.attn_sizes = attn_sizes
        self.mid_attn = mid_attn
        if (len(self.attn_sizes)>0 or self.mid_attn) and self.dim ==3:
            raise ValueError("3D attention very highly discouraged.")
        self.num_res_blocks = num_res_blocks
        self.dropout_prob = dropout_prob
        self.z_channels = z_channels
        self.double_z = double_z
        self.n_attention_heads = n_attention_heads

        assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        norm_params = dict(num_groups=norm_groups, eps=norm_eps, affine=norm_affine)
        assert act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"

        def get_act():
            if act == "gelu":
                return nn.GELU()
            elif act == "relu":
                return nn.ReLU()
            elif act == "silu":
                return nn.SiLU()

        padding = conv_kernel_size // 2
        conv_params = dict(
            kernel_size=conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=None,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
        )

        self.n_sizes = len(self.chs)
        self.conv_in = get_conv(
            self.in_channels, self.chs[0], dim=self.dim, **conv_params
        )

        curr_size = self.input_size
        self.downs = nn.ModuleList()
        for i_level in range(self.n_sizes):
            ch_in = chs[0] if i_level == 0 else chs[i_level - 1]
            ch_out = chs[i_level]

            resnets = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_out, **resnet_params))
                if curr_size in self.attn_sizes:
                    print("add attention")
                    attentions.append(
                        AttnBlock(
                            ch_out,
                            n_heads=self.n_attention_heads,
                            dim=self.dim,
                            norm_params=norm_params,
                        )
                    )
                ch_in = ch_out
            if len(attentions) == 0:
                attentions = None
            down = ResNetDown(resnets, attentions)
            curr_size = curr_size // 2
            self.downs.append(down)

        # middle

        # when no pad 1x266x266
        self.mid1 = ResNetBlock(ch_in, ch_in, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_in,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        # when no pad 1x262x262
        self.mid2 = ResNetBlock(ch_in, ch_in, **resnet_params)

        # end
        self.norm_out = nn.GroupNorm(num_channels=ch_in, **norm_params)
        self.act_out = get_act()
        # when no pad 1x258x258
        self.conv_out = get_conv(
            in_channels=ch_in,
            out_channels=2 * z_channels if double_z else z_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )
        # when no pad 1x256x256

    def forward(self, x):
        # timestep embedding
        conditionings = None

        # downsampling
        h = self.conv_in(x)
        # print(h.shape)
        for i, down in enumerate(self.downs):
            h, _ = down(
                h, conditionings=conditionings, no_down=(i == (len(self.downs) - 1))
            )

        # middle
        h = self.mid1(h, conditionings=conditionings)
        if self.mid_attn:
            h = self.mid_attn1(h)
        h = self.mid2(h, conditionings=conditionings)

        # end
        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        shape,
        chs=[48, 96, 192],
        attn_sizes=[],
        mid_attn=False,
        num_res_blocks=1,
        dropout_prob=0.0,
        z_channels=4,
        double_z=True,
        n_attention_heads=1,
        norm_groups=8,
        norm_eps=1e-6,
        norm_affine=True,
        act="gelu",
        conv_kernel_size=3,
        conv_padding_mode="zeros",
    ):
        super().__init__()
        self.shape=shape
        assert self.shape[1] == self.shape[2], "input must be square"
        self.in_channels = self.shape[0]
        self.input_size = self.shape[1]
        self.chs = chs
        self.dim = len(self.shape) - 1
        self.attn_sizes = attn_sizes
        self.mid_attn = mid_attn
        if (len(self.attn_sizes)>0 or self.mid_attn) and self.dim ==3:
            raise ValueError("3D attention very highly discouraged.")
        self.num_res_blocks = num_res_blocks
        self.dropout_prob = dropout_prob
        self.z_channels = z_channels
        self.double_z = double_z
        self.n_attention_heads = n_attention_heads

        assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd"
        norm_params = dict(num_groups=norm_groups, eps=norm_eps, affine=norm_affine)
        assert act in ["gelu", "relu", "silu"], "act must be gelu or relu or silu"

        def get_act():
            if act == "gelu":
                return nn.GELU()
            elif act == "relu":
                return nn.ReLU()
            elif act == "silu":
                return nn.SiLU()

        padding = conv_kernel_size // 2
        conv_params = dict(
            kernel_size=conv_kernel_size,
            padding=padding,
            padding_mode=conv_padding_mode,
        )
        nca_params = dict(
            norm_params=norm_params, get_act=get_act, conv_params=conv_params
        )
        resnet_params = dict(
            dim=self.dim,
            conditioning_dims=None,
            dropout_prob=self.dropout_prob,
            nca_params=nca_params,
        )

        self.n_sizes = len(self.chs)

        ch_in = self.chs[-1]
        self.conv_in = get_conv(self.z_channels, ch_in, dim=self.dim, **conv_params)

        self.mid1 = ResNetBlock(ch_in, ch_in, **resnet_params)
        if self.mid_attn:
            self.mid_attn1 = AttnBlock(
                ch_in,
                n_heads=self.n_attention_heads,
                dim=self.dim,
                norm_params=norm_params,
            )
        self.mid2 = ResNetBlock(ch_in, ch_in, **resnet_params)

        # upsampling
        curr_size = self.input_size // 2 ** (self.n_sizes - 1)
        self.ups = nn.ModuleList()
        for i_level in reversed(range(self.n_sizes)):
            ch_in = self.chs[i_level]

            resnets = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                resnets.append(ResNetBlock(ch_in, ch_in, **resnet_params))
                if curr_size in self.attn_sizes:
                    attentions.append(
                        AttnBlock(
                            ch_in,
                            n_heads=self.n_attention_heads,
                            dim=self.dim,
                            norm_params=norm_params,
                        )
                    )
            if len(attentions) == 0:
                attentions = None
            ch_out = self.chs[0] if i_level == 0 else self.chs[i_level - 1]  # for up
            up = ResNetUp(
                ch_out=ch_out, resnet_blocks=resnets, attention_blocks=attentions
            )
            curr_size = curr_size // 2
            self.ups.append(up)

        self.norm_out = nn.GroupNorm(num_channels=ch_out, **norm_params)
        self.act_out = get_act()
        self.conv_out = get_conv(
            in_channels=ch_out,
            out_channels=self.in_channels,
            dim=self.dim,
            init=zero_init,
            **conv_params,
        )

    def forward(self, z):
        self.last_z_shape = z.shape

        # timestep embedding
        conditionings = None

        # z to block_in
        h = self.conv_in(z)
        # print("after in",h.shape)

        # middle
        h = self.mid1(h, conditionings=conditionings)
        if self.mid_attn:
            h = self.mid_attn1(h)
        h = self.mid2(h, conditionings=conditionings)
        # print("after mid1,mid2",h.shape)

        # upsampling
        for i, up in enumerate(self.ups):
            h = up(h, conditionings=conditionings, no_up=(i == self.n_sizes - 1))
            # print(i,h.shape)

        h = self.norm_out(h)
        h = self.act_out(h)
        h = self.conv_out(h)
        return h
