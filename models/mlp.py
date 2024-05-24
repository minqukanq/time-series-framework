import torch.nn as nn

from layers.embed import DataEmbedding


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()

        self.args = args
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.hidden_size = args.d_ff

        self.enc_embedding = DataEmbedding(args.c_in, args.d_model, args.embed, args.freq, args.dropout)

        self.input_layer = nn.Linear(args.d_model, args.d_ff)
        self.hidden_layer = nn.Linear(args.d_ff, args.d_model)
        self.output_layer = nn.Linear(args.d_model, args.c_out)

        self.relu = nn.ReLU()

    def forecast(self, x_enc, x_mark_enc):
        x = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,D]

        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)

        return x

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.args.task_name == "forecasting":
            return self.forecast(x_enc, x_mark_enc)
