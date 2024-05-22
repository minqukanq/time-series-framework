for market in BTC ETH XRP
do

for model_name in DLinear
do

for pred_len in 6 12 24 48 96
do
python run.py \
    --is_training 1 \
    --use_wandb \
    --data_paths upbit/KRW-$market'_MINUTES_30'.csv \
    --model $model_name \
    --id $market'_30m_'$pred_len \
    --data Market \
    --freq h \
    --scaler S \
    --seq_len 96 \
    --pred_len $pred_len \
    --lr 1e-4 \
    --loss 'MSE' \
    --test_start_date '2022-04-01 09:00:00' \
    --des 'DLinear '$market' Min 30'
done

for pred_len in 6 12 24 48 96
do
python run.py \
    --is_training 1 \
    --use_wandb \
    --data_paths upbit/KRW-$market'_HOUR'.csv \
    --model $model_name \
    --id $market'_1h_'$pred_len \
    --data Market \
    --freq h \
    --scaler S \
    --seq_len 96 \
    --pred_len $pred_len \
    --lr 1e-4 \
    --loss 'MSE' \
    --test_start_date '2022-04-01 09:00:00' \
    --des 'DLinear '$market' Hour 1'
done

for pred_len in 6 12 24 48 96
do
python run.py \
    --is_training 1 \
    --use_wandb \
    --data_paths upbit/KRW-$market'_HOUR4'.csv \
    --model $model_name \
    --id $market'_4h_'$pred_len \
    --data Market \
    --freq h \
    --scaler S \
    --seq_len 96 \
    --pred_len $pred_len \
    --lr 1e-4 \
    --loss 'MSE' \
    --test_start_date '2022-04-01 09:00:00' \
    --des 'DLinear '$market' Hour 4'
done

for pred_len in 3 6 12 24 36
do
python run.py \
    --is_training 1 \
    --use_wandb \
    --data_paths upbit/KRW-$market'_DAYS'.csv \
    --model $model_name \
    --id $market'_1d_'$pred_len \
    --data Market \
    --freq h \
    --scaler S \
    --seq_len 36 \
    --label_len 18 \
    --pred_len $pred_len \
    --lr 1e-4 \
    --loss 'MSE' \
    --test_start_date '2022-04-01 09:00:00' \
    --des 'DLinear '$market' Day'
done

done

done