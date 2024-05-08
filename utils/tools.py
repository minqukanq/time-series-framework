import os
import sys
from pickle import dump

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == "type1":
        lr_adjust = {epoch: args.lr * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, dataset, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            scaler = dataset.scaler
            dump(scaler, open(f"{path}/scaler.pkl", "wb"))

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        if isinstance(model, torch.nn.DataParallel):
            torch.save(model.module.state_dict(), path + "/" + "checkpoint.pth")
        else:
            torch.save(model.state_dict(), path + "/" + "checkpoint.pth")
        self.val_loss_min = val_loss


def load_model(model, path):
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(torch.load())
    else:
        model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    return model


def visual(true, preds=None, name="./pic/test.pdf"):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label="GroundTruth", linewidth=2)
    if preds is not None:
        plt.plot(preds, label="Prediction", linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches="tight")


def make_setting():
    commands = sys.argv[1:]
    setting_dict = {}
    for i in range(len(commands) - 1):
        if commands[i] == "--is_training" or commands[i] == "--use_wandb" or commands[i] == "--data_paths":
            continue
        if commands[i].startswith("--") and (commands[i + 1].startswith("--") is False):
            if "." in commands[i + 1] or " " in commands[i + 1] or os.sep in commands[i + 1]:
                commands[i + 1] = commands[i + 1].replace(" ", "_").replace(".", "_").replace(os.sep, "_")
            setting_dict[commands[i].replace("--", "")] = commands[i + 1]
    setting_str = [f"{key[:3]}_{value}" for key, value in setting_dict.items()]
    setting_str = "__".join(setting_str)
    return setting_dict, setting_str
