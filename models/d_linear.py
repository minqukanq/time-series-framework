import torch
import torch.nn as nn
from einops import rearrange

from layers.decompositions import TrendDecomp


class DLinear(nn.Module):
    def __init__(self, args, kernel_size=25, individual=False):
        super(DLinear, self).__init__()
        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.decompsition = TrendDecomp(kernel_size)
        self.channels = args.c_in

        self.individual = individual
        if self.individual:
            self.seasonal_linear = torch.nn.ModuleList()
            self.trend_linear = torch.nn.ModuleList()
            for i in range(self.channels):
                self.trend_linear.append(torch.nn.Linear(self.seq_len, self.pred_len))
                self.seasonal_linear.append(torch.nn.Linear(self.seq_len, self.pred_len))

                self.trend_linear[i].weight = torch.nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.seasonal_linear[i].weight = torch.nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        else:
            self.trend_linear = torch.nn.Linear(self.seq_len, self.pred_len)
            self.trend_linear.weight = torch.nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.seasonal_linear = torch.nn.Linear(self.seq_len, self.pred_len)
            self.seasonal_linear.weight = torch.nn.Parameter((1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

    def forecast(self, x):
        """
        x: [batch_size, window_size, channel_size]
        """
        # Normalization
        means = x.mean(1, keepdim=True).detach()
        x = x - means
        stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x /= stdev

        seasonal_init, trend_init = self.decompsition(x)
        trend_init = rearrange(trend_init, "b w c -> b c w")
        seasonal_init = rearrange(seasonal_init, "b w c -> b c w")

        if self.individual:
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], dtype=trend_init.dtype).to(
                trend_init.device
            )
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], dtype=seasonal_init.dtype).to(
                seasonal_init.device
            )
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.trend_linear[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.seasonal_linear[idx](seasonal_init[:, idx, :])

        else:
            trend_output = self.trend_linear(trend_init)
            seasonal_output = self.seasonal_linear(seasonal_init)

        out = rearrange(seasonal_output + trend_output, "b c w -> b w c")

        # De-Normalization
        out = out * stdev
        out = out + means
        return out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.args.task_name == "forecasting":
            return self.forecast(x_enc)
        if self.args.task_name == "anomaly_detection":
            raise NotImplementedError
