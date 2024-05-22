import torch
import torch.nn as nn
from einops import rearrange


class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        """
        x: [batch_size, window_size, channel_size]
        """

        # w 차원만 padding, 1 이 들어간 차원은 변화X
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)

        x = self.avg(rearrange(x, "b w c -> b c w"))
        x = rearrange(x, "b c w -> b w c")
        return x


class TrendDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(TrendDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        """
        x: [batch_size, window_size, channel_size]
        """
        trend = self.moving_avg(x)
        remainder = x - trend
        return remainder, trend


class TrendDecompMulti(nn.Module):
    """
    Multiple Series decomposition block from FEDformer
    """

    def __init__(self, kernel_size):
        super(TrendDecompMulti, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = [TrendDecomp(kernel) for kernel in kernel_size]

    def forward(self, x):
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class DFTDecomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k=5):
        super(DFTDecomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend
