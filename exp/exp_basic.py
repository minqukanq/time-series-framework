import os
import torch


class ExpBasic:
    def __init__(self, args) -> None:
        self.args = args
        self.model_dict = {}
        self.device = self._acquire_device()

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
