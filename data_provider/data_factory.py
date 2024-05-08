from torch.utils.data import DataLoader

from data_provider.data_loader import CryptocurrencyDataset

dataset_dict = {"CryptocurrencyDataset": CryptocurrencyDataset}


def data_provider(args, flag):
    data = dataset_dict[args.data]
    shuffle = False if flag == "test" else True
    batch_size = args.batch_size
    drop_last = True

    if args.task_name == "forecasting":
        dataset = data(
            root_path=args.root_path,
            data_path=args.data_paths,
            flag=flag,
            size=[args.seq_len, args.pred_len, args.label_len],
            features=args.features,
            test_start_date=args.test_start_date,
            target=args.target,
            freq=args.freq,
            scaler=args.scaler,
        )
        print(flag, len(data))
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers, drop_last=drop_last)
        return dataset, data_loader
