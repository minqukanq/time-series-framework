import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.embed import DataEmbedding


class LSTNet(nn.Module):
    def __init__(self, args):
        super(LSTNet, self).__init__()
        self.P = args.seq_len
        self.m = args.d_model
        self.hidR = args.rnn_hidden
        self.hidC = args.cnn_hidden
        self.hidS = args.skip_hidden
        self.Ck = args.num_kernels
        self.skip = args.skip
        self.pt = int((self.P - self.Ck) / self.skip)
        self.hw = args.highway_window
        self.c_out = args.c_out

        self.enc_embedding = DataEmbedding(args.c_in, args.d_model, args.embed, args.freq, args.dropout)

        self.conv1 = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))
        self.GRU1 = nn.GRU(self.hidC, self.hidR)
        self.dropout = nn.Dropout(p=args.dropout)
        if self.skip > 0:
            self.GRUskip = nn.GRU(self.hidC, self.hidS)
            self.linear1 = nn.Linear(self.hidR + self.skip * self.hidS, self.m)
        else:
            self.linear1 = nn.Linear(self.hidR, self.m)
        if self.hw > 0:
            self.highway = nn.Linear(self.hw, 1)
        self.output = None
        # if args.output_fun == 'sigmoid':
        #     self.output = F.sigmoid
        # if args.output_fun == 'tanh':
        #     self.output = F.tanh

        self.projection = nn.Linear(self.m, self.m * self.c_out)

    def forward(self, x_enc, x_enc_mark, x_dec, x_dec_mark):
        batch_size = x_enc.size(0)

        x_enc = self.enc_embedding(x_enc, x_enc_mark)

        # CNN
        c = x_enc.view(-1, 1, self.P, self.m)
        c = F.relu(self.conv1(c))
        c = self.dropout(c)
        c = torch.squeeze(c, 3)

        # RNN
        r = c.permute(2, 0, 1).contiguous()
        _, r = self.GRU1(r)
        r = self.dropout(torch.squeeze(r, 0))

        # skip-rnn

        if self.skip > 0:
            s = c[:, :, int(-self.pt * self.skip) :].contiguous()
            s = s.view(batch_size, self.hidC, self.pt, self.skip)
            s = s.permute(2, 0, 3, 1).contiguous()
            s = s.view(self.pt, batch_size * self.skip, self.hidC)
            _, s = self.GRUskip(s)
            s = s.view(batch_size, self.skip * self.hidS)
            s = self.dropout(s)
            r = torch.cat((r, s), 1)

        res = self.linear1(r)

        # highway
        if self.hw > 0:
            z = x_enc[:, -self.hw :, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.m)
            res = res + z

        res = self.projection(res)
        res = res.view(batch_size, self.m, self.c_out)

        if self.output:
            res = self.output(res)

        return res
