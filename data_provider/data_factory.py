from torch.utils.data import DataLoader

from data_provider.data_loader import CryptocurrencyDataset

dataset_dict = {"CryptocurrencyDataset": CryptocurrencyDataset}


def data_provider(args, shuffle):
    data = dataset_dict[args.data]
    shuffle = False if shuffle == "test" else True
    batch_size = args.batch_size

    if args.task_name == "forecasting":
        dataset = data(
            root_path=args.root_path,
            data_path=args.data_paths,
            flag=shuffle,
            size=[args.seq_len, args.pred_len, args.label_len],
            features=args.features,
            test_start_date=args.test_start_date,
            target=args.target,
            freq=args.freq,
            scaler=args.scaler,
        )
        print(shuffle, len(data))
        drop_last = True
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=drop_last)
        return dataset, data_loader
