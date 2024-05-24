import torch.nn as nn

from layers.embed import DataEmbedding


class LSTM(nn.Module):
    def __init__(self, args):
        super(LSTM, self).__init__()

        self.enc_embedding = DataEmbedding(args.c_in, args.d_model, args.embed, args.freq, args.dropout)

        self.lstm = nn.LSTM(args.d_model, args.rnn_hidden, args.e_layers, batch_first=True)
        self.predict = nn.Sequential(nn.Dropout(args.dropout), nn.Linear(args.rnn_hidden, args.pred_len))

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x = self.enc_embedding(x_enc, x_mark_enc)

        out, _ = self.lstm(x)
        out = self.predict(out[:, -1:, :])

        return out
