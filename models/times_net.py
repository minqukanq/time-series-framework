import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.conv_blocks import InceptionBlockV1
from layers.embed import DataEmbedding


def fft_for_period(x, k=3):
    # [Batch Size, Time Length, Channel]
    xf = torch.fft.rfft(x, dim=1)  # dim=1 입력의 두 번째 축을 주파수 축으로 사용하겠다.

    # 주파수 스펙트럼의 크기를 구하여, 각 주파수에 대한 중요도를 계산
    # abs() 주파수 스펙트럼의 크기를 계산. xf의 복소수 결과 중 크기만을 남기고 복수수의 각도 정보를 제거 (허수가 사라짐)
    # batch 방향과 channel 방향으로 평균을 구해서 주파수 축 방향으로 총 평균값 계산
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 상수 제거

    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list  # 주기 계산
    return period, abs(xf).mean(-1)[:, top_list]  # 모든 Batch에 대해서 높은 주파수를 갖은 시간축을 선택


class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.k = configs.top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlockV1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels),
            nn.GELU(),
            InceptionBlockV1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels),
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = fft_for_period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            # reshape
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class TimesNet(nn.Module):
    def __init__(self, args):
        super(TimesNet, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.model = nn.ModuleList([TimesBlock(args) for _ in range(args.e_layers)])
        self.enc_embedding = DataEmbedding(args.c_in, args.d_model, args.embed, args.freq, args.dropout)
        self.layer = args.e_layers
        self.layer_norm = nn.LayerNorm(args.d_model)
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(args.d_model, args.c_out, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        """
        Args:
            x_enc: B, T, C / Batch size, Time series, Channels
            x_mark_enc: B, T, C / encoded datetime

        Returns:
            Batch, Pred Length, Channels
        """
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,D]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization
        dec_out = dec_out * stdev
        dec_out = dec_out + means

        return dec_out[:, -self.pred_len :, :]  # [B,L,C]

    def anomaly_detection(self, x_enc):
        """
        Args:
            x_enc: B, T, C / Batch size, Time series, Channels
            x_mark_enc: B, T, C / encoded datetime

        Returns:
            Batch, Pred Length, Channels
        """
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,D]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))

        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization
        dec_out = dec_out * stdev
        dec_out = dec_out + means

        return dec_out  # [B,L,C]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.args.task_name == "forecasting":
            return self.forecast(x_enc, x_mark_enc)
        if self.args.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
