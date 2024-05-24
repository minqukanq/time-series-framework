import torch.nn as nn
from einops import rearrange
from torch.nn.utils import weight_norm

from layers.embed import DataEmbedding


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.network = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.network(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCN(nn.Module):
    def __init__(self, args):
        super(TCN, self).__init__()

        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.c_in = args.c_in
        self.c_out = args.c_out
        self.d_model = args.d_model
        self.n_layers = args.e_layers
        self.dropout = args.dropout

        kernel_size = 2

        self.num_channels = [args.d_model] * self.n_layers
        layers = []
        num_levels = len(self.num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = self.d_model if i == 0 else self.num_channels[i - 1]
            out_channels = self.num_channels[i]

            layers += [
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=self.dropout,
                    dilation=dilation_size,
                )
            ]

        self.enc_embedding = DataEmbedding(self.c_in, args.d_model, args.embed, args.freq, self.dropout)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)

        self.network = nn.Sequential(*layers)
        self.out_proj = nn.Linear(args.d_model, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # embedding
        x_enc = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,D]
        x_enc = self.predict_linear(x_enc.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension

        x_enc = rearrange(x_enc, "b w c -> b c w")
        x_enc = self.network(x_enc)
        x_enc = rearrange(x_enc, "b c w -> b w c")
        x_enc = self.out_proj(x_enc)

        return x_enc
