import os

import torch

from models.d_linear import DLinear
from models.i_transformer import ITransformer
from models.lst_net import LSTNet
from models.lstm import LSTM
from models.micn import MICN
from models.mlp import MLP
from models.patch_tst import PatchTST
from models.seg_rnn import SegRNN
from models.tcn import TCN
from models.time_mixer import TimeMixer
from models.times_net import TimesNet
from models.transformer import Transformer


class ExpBasic:
    def __init__(self, args) -> None:
        self.args = args
        self.model_dict = {
            "TimeMixer": TimeMixer,
            "DLinear": DLinear,
            "TimesNet": TimesNet,
            "MICN": MICN,
            "ITransformer": ITransformer,
            "PatchTST": PatchTST,
            "SegRNN": SegRNN,
            "LSTNet": LSTNet,
            "Transformer": Transformer,
            "LSTM": LSTM,
            "MLP": MLP,
            "TCN": TCN,
        }
        self.device = self._acquire_device()
        self.args.device = self.device
        self.model = self._build_model().to(self.device)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device("cuda:{}".format(self.args.gpu))
            print("Use GPU: cuda:{}".format(self.args.gpu))
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self, flag):
        pass

    def train(self, setting):
        raise NotImplementedError

    def vali(self):
        pass

    def test(self):
        pass
