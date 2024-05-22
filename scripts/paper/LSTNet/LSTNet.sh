model_name=LSTNet

for market in BTC ETH XRP
do

for pred_len in 6 12 24 48 96
do
python -u run.py \
    --is_training 1 \
    --use_wandb \
    --exp exp_forecast \
    --data_paths upbit/KRW-$market'_MINUTES_30.csv' \
    --model $model_name \
    --id $market'_30m_pred'$pred_len'_3' \
    --data Market \
    --freq h \
    --scaler S \
    --pred_len $pred_len \
    --dropout 0.2 \
    --d_model 100 \
    --des 'Paper '$market' Min 30'\
    --gpu_id 0
done

for pred_len in 6 12 24 48 96
do
python -u run.py \
    --is_training 1 \
    --use_wandb \
    --exp exp_forecast \
    --data_paths upbit/KRW-$market'_HOUR.csv' \
    --model $model_name \
    --id $market'_1h_pred'$pred_len'_3' \
    --data Market \
    --freq h \
    --scaler S \
    --pred_len $pred_len \
    --dropout 0.2 \
    --d_model 100 \
    --des 'Paper '$market' Hour 1'\
    --gpu_id 0
done

for pred_len in 6 12 24 48 96
do
python -u run.py \
    --is_training 1 \
    --use_wandb \
    --exp exp_forecast \
    --data_paths upbit/KRW-$market'_HOUR4.csv' \
    --model $model_name \
    --id $market'_4h_pred'$pred_len'_3' \
    --data Market \
    --freq h \
    --scaler S \
    --pred_len $pred_len \
    --dropout 0.2 \
    --d_model 100 \
    --test_start_date '2022-04-01 01:00:00' \
    --des 'Paper '$market' Hour 4'\
    --gpu_id 0
done

for pred_len in 3 6 12 24 36
do
python -u run.py \
    --is_training 1 \
    --use_wandb \
    --exp exp_forecast \
    --data_paths upbit/KRW-$market'_DAYS.csv' \
    --model $model_name \
    --id $market'_1d_pred'$pred_len'_3' \
    --data Market \
    --freq h \
    --scaler S \
    --pred_len $pred_len \
    --dropout 0.2 \
    --d_model 100 \
    --test_start_date '2022-04-01 09:00:00' \
    --des 'Paper '$market' Day' \
    --gpu_id 0
done

done