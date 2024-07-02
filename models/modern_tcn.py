import torch
from torch import nn

from layers.decompositions import TrendDecomp
from layers.flatten import FlattenHeads
from layers.rev_in import RevIN


class LayerNorm(nn.Module):

    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.Layernorm(channels)

    def forward(self, x):

        B, M, D, N = x.shape
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, N, D)
        x = self.norm(x)
        x = x.reshape(B, M, N, D)
        x = x.permute(0, 1, 3, 2)
        return x


def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
    )
    result.add_module("bn", get_bn(out_channels))
    return result


def fuse_bn(conv, bn):

    kernel = conv.weight
    running_mean = bn.running_mean
    running_var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    eps = bn.eps
    std = (running_var + eps).sqrt()
    t = (gamma / std).reshape(-1, 1, 1)
    return kernel * t, beta - running_mean * gamma / std


class ReparamLargeKernelConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, small_kernel, small_kernel_merged=False, nvars=7):
        super(ReparamLargeKernelConv, self).__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        # We assume the conv does not change the feature map size, so padding = k//2. Otherwise, you may configure padding as you wish,
        # and change the padding of small_conv accordingly.
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=False,
            )
            if small_kernel is not None:
                assert small_kernel <= kernel_size, "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bias=False,
                )

    def forward(self, inputs):

        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        return out

    def PaddingTwoEdge1d(self, x, pad_length_left, pad_length_right, pad_values=0):

        D_out, D_in, ks = x.shape
        if pad_values == 0:
            pad_left = torch.zeros(D_out, D_in, pad_length_left)
            pad_right = torch.zeros(D_out, D_in, pad_length_right)
        else:
            pad_left = torch.ones(D_out, D_in, pad_length_left) * pad_values
            pad_right = torch.ones(D_out, D_in, pad_length_right) * pad_values
        x = torch.cat([pad_left, x], dims=-1)
        x = torch.cat([x, pad_right], dims=-1)
        return x

    def get_equivalent_kernel_bias(self):
        eq_k, eq_b = fuse_bn(self.lkb_origin.conv, self.lkb_origin.bn)
        if hasattr(self, "small_conv"):
            small_k, small_b = fuse_bn(self.small_conv.conv, self.small_conv.bn)
            eq_b += small_b
            eq_k += self.PaddingTwoEdge1d(
                small_k, (self.kernel_size - self.small_kernel) // 2, (self.kernel_size - self.small_kernel) // 2, 0
            )
        return eq_k, eq_b

    def merge_kernel(self):
        eq_k, eq_b = self.get_equivalent_kernel_bias()
        self.lkb_reparam = nn.Conv1d(
            in_channels=self.lkb_origin.conv.in_channels,
            out_channels=self.lkb_origin.conv.out_channels,
            kernel_size=self.lkb_origin.conv.kernel_size,
            stride=self.lkb_origin.conv.stride,
            padding=self.lkb_origin.conv.padding,
            dilation=self.lkb_origin.conv.dilation,
            groups=self.lkb_origin.conv.groups,
            bias=True,
        )
        self.lkb_reparam.weight.data = eq_k
        self.lkb_reparam.bias.data = eq_b
        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):

        super(Block, self).__init__()
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
            nvars=nvars,
        )
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(
            in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars
        )
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(
            in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars
        )
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(
            in_channels=nvars * dmodel, out_channels=nvars * dff, kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel
        )
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(
            in_channels=nvars * dff, out_channels=nvars * dmodel, kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel
        )
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

    def forward(self, x):

        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(self, ffn_ratio, num_blocks, large_size, small_size, dmodel, dw_model, nvars, small_kernel_merged=False, drop=0.1):

        super(Stage, self).__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for i in range(num_blocks):
            blk = Block(
                large_size=large_size,
                small_size=small_size,
                dmodel=dmodel,
                dff=d_ffn,
                nvars=nvars,
                small_kernel_merged=small_kernel_merged,
                drop=drop,
            )
            blks.append(blk)

        self.blocks = nn.ModuleList(blks)

    def forward(self, x):

        for blk in self.blocks:
            x = blk(x)

        return x


class ModernTCNLayer(nn.Module):
    def __init__(
        self,
        patch_size,
        patch_stride,
        stem_ratio,
        downsample_ratio,
        ffn_ratio,
        num_blocks,
        large_size,
        small_size,
        dims,
        dw_dims,
        nvars,
        small_kernel_merged=False,
        backbone_dropout=0.1,
        head_dropout=0.1,
        use_multi_scale=True,
        revin=True,
        affine=True,
        subtract_last=False,
        freq=None,
        seq_len=512,
        c_in=7,
        individual=False,
        target_window=96,
    ):
        super(ModernTCNLayer, self).__init__()

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride), nn.BatchNorm1d(dims[0]))
        self.downsample_layers.append(stem)

        self.num_stage = len(num_blocks)
        if self.num_stage > 1:
            for i in range(self.num_stage - 1):
                downsample_layer = nn.Sequential(
                    nn.BatchNorm1d(dims[i]),
                    nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
                )
            self.downsample_layers.append(downsample_layer)

        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # backbone
        self.num_stage = len(num_blocks)
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(
                ffn_ratio,
                num_blocks[stage_idx],
                large_size[stage_idx],
                small_size[stage_idx],
                dmodel=dims[stage_idx],
                dw_model=dw_dims[stage_idx],
                nvars=nvars,
                small_kernel_merged=small_kernel_merged,
                drop=backbone_dropout,
            )
            self.stages.append(layer)

        # head
        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[self.num_stage - 1]

        if use_multi_scale:
            self.head_nf = d_model * patch_num
            self.head = FlattenHeads(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)
        else:

            if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)
            self.head = FlattenHeads(self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout)

    def forward_feature(self, x, te=None):

        B, M, L = x.shape
        x = x.unsqueeze(-2)

        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(self, x, te=None):

        # instance norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "norm")
            x = x.permute(0, 2, 1)
        x = self.forward_feature(x, te)
        x = self.head(x)
        # de-instance norm
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "denorm")
            x = x.permute(0, 2, 1)
        return x

    def structural_reparam(self):
        for m in self.modules():
            if hasattr(m, "merge_kernel"):
                m.merge_kernel()


class ModernTCN(nn.Module):
    def __init__(self, configs):
        super(ModernTCN, self).__init__()
        # hyper param
        self.stem_ratio = configs.stem_ratio
        self.downsample_ratio = configs.downsample_ratio
        self.ffn_ratio = configs.ffn_ratio
        self.num_blocks = configs.num_blocks
        self.large_size = configs.large_size
        self.small_size = configs.small_size
        self.dims = configs.dims
        self.dw_dims = configs.dw_dims

        self.nvars = configs.enc_in
        self.small_kernel_merged = configs.small_kernel_merged
        self.drop_backbone = configs.dropout
        self.drop_head = configs.head_dropout
        self.use_multi_scale = configs.use_multi_scale
        self.revin = configs.revin
        self.affine = configs.affine
        self.subtract_last = configs.subtract_last

        self.freq = configs.freq
        self.seq_len = configs.seq_len
        self.c_in = (self.nvars,)
        self.individual = configs.individual
        self.target_window = configs.pred_len

        self.kernel_size = configs.kernel_size
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride

        # decomp
        self.decomposition = configs.decomposition
        if self.decomposition:
            self.decomp_module = TrendDecomp(self.kernel_size)
            self.model_res = ModernTCNLayer(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=self.freq,
                seq_len=self.seq_len,
                c_in=self.c_in,
                individual=self.individual,
                target_window=self.target_window,
            )
            self.model_trend = ModernTCNLayer(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=self.freq,
                seq_len=self.seq_len,
                c_in=self.c_in,
                individual=self.individual,
                target_window=self.target_window,
            )
        else:
            self.model = ModernTCNLayer(
                patch_size=self.patch_size,
                patch_stride=self.patch_stride,
                stem_ratio=self.stem_ratio,
                downsample_ratio=self.downsample_ratio,
                ffn_ratio=self.ffn_ratio,
                num_blocks=self.num_blocks,
                large_size=self.large_size,
                small_size=self.small_size,
                dims=self.dims,
                dw_dims=self.dw_dims,
                nvars=self.nvars,
                small_kernel_merged=self.small_kernel_merged,
                backbone_dropout=self.drop_backbone,
                head_dropout=self.drop_head,
                use_multi_scale=self.use_multi_scale,
                revin=self.revin,
                affine=self.affine,
                subtract_last=self.subtract_last,
                freq=self.freq,
                seq_len=self.seq_len,
                c_in=self.c_in,
                individual=self.individual,
                target_window=self.target_window,
            )

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        te = None
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            res = self.model_res(res_init, te)
            trend = self.model_trend(trend_init, te)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            if te is not None:
                te = te.permute(0, 2, 1)
            x = self.model(x, te)
            x = x.permute(0, 2, 1)
        return x
