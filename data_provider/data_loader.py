import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

from utils.time_features import time_features


class CryptocurrencyDataset(Dataset):
    def __init__(
        self,
        root_path,
        size,
        flag="train",
        features="S",
        data_path="KRW-BTC_HOUR.csv",
        train_start_date="2017-09-25 09:00:00",
        test_start_date="2023-05-16 00:00:00",
        target="close",
        scaler=None,
        freq="h",
    ):
        self.seq_len = size[0]
        self.pred_len = size[1]
        self.label_len = size[2]

        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.train_start_date = train_start_date
        self.test_start_date = test_start_date

        self.scaler = scaler
        if scaler == "S":
            self.scaler = StandardScaler()
        elif scaler == "M":
            self.scaler = MinMaxScaler()

        self.__read_data__()

    def __read_data__(self):
        print(self.root_path)
        print(self.data_path)
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("datetime")
        df_raw = df_raw[["datetime"] + cols + [self.target]]

        df_raw = df_raw[df_raw["datetime"] >= self.train_start_date].reset_index(drop=True)

        target_index = df_raw.loc[df_raw["datetime"] == self.test_start_date].index[0]
        num_test = len(df_raw[target_index:])
        num_train = int((len(df_raw) - num_test) * 0.9)

        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scaler is not None:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["datetime"]][border1:border2]
        df_stamp["datetime"] = pd.to_datetime(df_stamp["datetime"])

        data_stamp = time_features(pd.to_datetime(df_stamp["datetime"].values), freq=self.freq)
        data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len

        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
