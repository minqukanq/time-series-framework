import os
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import wandb

from data_provider.data_factory import data_provider
from exp.exp_basic import ExpBasic
from utils.losses import MAPELoss, SMAPELoss
from utils.metrics import metric
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.tools import EarlyStopping, load_model, visual

warnings.filterwarnings("ignore")


class ExpForecasting(ExpBasic):
    def __init__(self, args) -> None:
        super().__init__(args)

        if args.use_wandb:
            wandb.init(
                project=f"paper_{args.model}",
                group=args.des,
                name=args.id,
                notes=args.des,
                tags=args.setting.values(),
                config=args.setting,
            )

            wandb.define_metric("train_loss", summary="min")
            wandb.define_metric("val_loss", summary="min")
            wandb.define_metric("test_loss", summary="min")

            wandb.define_metric("train_mse", summary="min")
            wandb.define_metric("val_mse", summary="min")
            wandb.define_metric("test_mse", summary="min")

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args)

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            print("Use multi gpu")
        return model

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(self.model.parameters(), lr=1e-10)
        return model_optim

    def _select_criterion(self, loss_name="MSE"):
        if loss_name == "MSE":
            return nn.MSELoss()
        elif loss_name == "SMAPE":
            return SMAPELoss()
        elif loss_name == "MAPE":
            return MAPELoss()

    def train(self, setting):
        train_dataset, train_loader = self._get_data(flag="train")
        vali_dataset, vali_loader = self._get_data(flag="val")
        test_dataset, test_loader = self._get_data(flag="test")

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        optimizer = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)
        scheduler = CosineAnnealingWarmUpRestarts(optimizer=optimizer, eta_max=self.args.lr)

        for epoch in range(self.args.epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            iters = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_input = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_input = torch.cat([batch_y[:, : self.args.label_len, :], dec_input], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)

                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.epochs - epoch) * train_steps - i)
                    print("\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                optimizer.step()
                scheduler.step(epoch + i / iters)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_dataset, vali_loader, criterion)
            test_loss = self.vali(test_dataset, test_loader, criterion)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss
                )
            )
            if self.args.use_wandb:
                wandb.log(
                    {"Steps": train_steps, "train_loss": train_loss, "val_loss": vali_loss, "test_loss": test_loss},
                    step=epoch + 1,
                )

            early_stopping(vali_loss, self.model, train_dataset, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_model_path = path + "/" + "checkpoint.pth"
        self.model = load_model(self.model, best_model_path)
        self.model.to(self.device)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_input = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_input = torch.cat([batch_y[:, : self.args.label_len, :], dec_input], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len :, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag="test")
        print(test_data)
        if test:
            print("loading model")
            self.model.load_state_dict(torch.load(os.path.join("./checkpoints/" + setting, "checkpoint.pth")))

        preds = []
        trues = []
        folder_path = "./test_results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_input = torch.zeros_like(batch_y[:, -self.args.pred_len :, :]).float()
                dec_input = torch.cat([batch_y[:, : self.args.label_len, :], dec_input], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_input, batch_y_mark)

                f_dim = -1 if self.args.features == "MS" else 0
                outputs = outputs[:, -self.args.pred_len :, :]
                batch_y = batch_y[:, -self.args.pred_len :, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + ".pdf"))

        preds = np.array(preds)
        trues = np.array(trues)
        print("test shape:", preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print("test shape:", preds.shape, trues.shape)

        # result save
        folder_path = "./results/" + setting + "/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}".format(mse, mae, rmse, mape, mspe))
        f = open(f"{os.path.splitext(os.path.basename(__file__))[0]}_result.txt", "a")
        f.write(setting + "  \n")
        f.write("mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}".format(mse, mae, rmse, mape, mspe))
        f.write("\n")
        f.write("\n")
        f.close()

        if self.args.use_wandb:
            wandb.log(
                {
                    "Test MAE": mae,
                    "Test MSE": mse,
                    "Test RMSE": rmse,
                    "Test MAPE": mape,
                    "Test MSPE": mspe,
                }
            )
