import argparse
import random

import numpy as np
import torch

from exp.exp_forecast import ExpForecasting
from utils.tools import make_setting

if __name__ == "__main__":
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description="DeepLearning Experiments")

    # config
    parser.add_argument("--task_name", type=str, default="forecasting", help="task name, options:[forecasting]")
    parser.add_argument("--is_training", type=int, required=True, default=1, help="status")
    parser.add_argument("--id", type=str, required=True, help="model id")
    parser.add_argument("--model", type=str, required=True, help="model name, options: [TimesNet]")
    parser.add_argument("--use_wandb", action="store_true", help="use weight & biases for graph loss", default=False)

    # data loader
    parser.add_argument("--data", type=str, default="CryptocurrencyDataset", help="dataset type")
    parser.add_argument("--root_path", type=str, default="data/", help="root path of the data file")
    parser.add_argument(
        "--data_paths",
        type=str,
        default="hour4.csv",
        help="data file",
    )
    parser.add_argument("--test_start_date", type=str, default="2023-05-16 00:00:00", help="test start date")
    parser.add_argument(
        "--scaler",
        type=str,
        default=None,
        help="data scaler, options:[S, M]; \
            S:StandardScaler, \
            M:MinMaxScaler",
    )

    parser.add_argument(
        "--features",
        type=str,
        default="S",
        help="forecasting task, options:[M, S, MS]; \
            M:multivariate predict multivariate, \
            S:univariate predict univariate, \
            MS:multivariate predict univariate",
    )
    parser.add_argument("--target", type=str, default="close", help="target feature in S or MS task")
    parser.add_argument(
        "--freq",
        type=str,
        default="h",
        help="freq for time features encoding, \
                            options:[m:minutely, h:hourly, d:daily]",
    )
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="location of model checkpoints")

    # optimization
    parser.add_argument("--num_workers", type=int, default=10, help="data loader num workers")
    parser.add_argument("--itr", type=int, default=1, help="experiments times")
    parser.add_argument("--epochs", type=int, default=80, help="train epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size of train input data")
    parser.add_argument("--patience", type=int, default=7, help="early stopping patience")
    parser.add_argument("--lr", type=float, default=1e-4, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="test", help="exp description")
    parser.add_argument("--loss", type=str, default="MSE", help="loss function")

    # GPU
    parser.add_argument("--use_gpu", type=bool, default=True, help="use gpu")
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument("--use_multi_gpu", action="store_true", help="use multiple gpus", default=False)
    parser.add_argument("--devices", type=str, default="0,1", help="device ids of multile gpus")

    # forecasting
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="start token length")
    parser.add_argument("--pred_len", type=int, default=63, help="prediction sequence length")
    parser.add_argument("--inverse", action="store_true", help="inverse output data", default=False)

    # model define
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]")
    parser.add_argument("--num_kernels", type=int, default=6, help="for Inception")
    parser.add_argument("--e_layers", type=int, default=2, help="num of encoder layers")
    parser.add_argument("--d_layers", type=int, default=1, help="num of decoder layers")
    parser.add_argument("--d_model", type=int, default=512, help="dimension of model")
    parser.add_argument("--moving_avg", type=int, default=25, help="window size of moving average")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--factor", type=int, default=1, help="attn factor")
    parser.add_argument("--c_in", type=int, default=1, help="Input size")
    parser.add_argument("--c_out", type=int, default=1, help="output size")
    parser.add_argument("--d_ff", type=int, default=2048, help="dimension of fcn")
    parser.add_argument("--top_k", type=int, default=5, help="for TimesBlock")

    # supplementary config for LSTNet model
    parser.add_argument('--rnn_hidden', type=int, default=100, help='rnn hidden size')
    parser.add_argument('--cnn_hidden', type=int, default=100, help='cnn hidden size')
    parser.add_argument('--skip_hidden', type=int, default=5)
    parser.add_argument('--skip', type=float, default=24)
    parser.add_argument('--highway_window', type=int, default=24, help='The window size of the highway component')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() else False
    print(torch.cuda.is_available())
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        device_ids = args.devices.split(",")
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)
    print("-" * 50)

    if args.task_name == "forecasting":
        Exp = ExpForecasting

    setting_dict, setting_str = make_setting()
    args.setting = setting_dict

    if args.is_training:
        for itr in range(args.itr):
            exp = Exp(args)
            print("########## training : {} ##########".format(setting_str))
            exp.train(setting_str)
            print("########## testing : {} ##########".format(setting_str))
            exp.test(setting_str)

            torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        print("########## testing : {} ##########".format(setting_str))
        exp.test(setting_str, test=1)

        torch.cuda.empty_cache()
