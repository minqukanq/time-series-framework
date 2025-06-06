model_name=SegRNN

for market in BTC ETH XRP
do

for pred_len in 6 12 24 48 96
do
python run.py \
    --is_training 1 \
    --use_wandb \
    --data_paths upbit/KRW-$market'_MINUTES_30'.csv \
    --model $model_name \
    --id $market'_30m_pred'$pred_len'_1' \
    --freq h \
    --scaler S \
    --pred_len $pred_len \
    --d_model 512 \
    --seg_len 6 \
    --dropout 0.5 \
    --des 'Paper '$market' Min 30' \
    --gpu 0
done

for pred_len in 6 12 24 48 96
do
python run.py \
    --is_training 1 \
    --use_wandb \
    --data_paths upbit/KRW-$market'_HOUR'.csv \
    --model $model_name \
    --id $market'_1h_pred'$pred_len'_1' \
    --freq h \
    --scaler S \
    --pred_len $pred_len \
    --d_model 512 \
    --seg_len 6 \
    --dropout 0.5 \
    --des 'Paper '$market' Hour 1' \
    --gpu 0
done

for pred_len in 6 12 24 48 96
do
python run.py \
    --is_training 1 \
    --use_wandb \
    --data_paths upbit/KRW-$market'_HOUR4'.csv \
    --model $model_name \
    --id $market'_4h_pred'$pred_len'_1' \
    --freq h \
    --scaler S \
    --pred_len $pred_len \
    --d_model 512 \
    --seg_len 6 \
    --dropout 0.5 \
    --test_start_date '2023-05-16 01:00:00' \
    --des 'Paper '$market' Hour 4' \
    --gpu 0
done

for pred_len in 3 6 12 24 36
do
python run.py \
    --is_training 1 \
    --use_wandb \
    --data_paths upbit/KRW-$market'_DAYS'.csv \
    --model $model_name \
    --id $market'_1d_pred'$pred_len'_1' \
    --freq d \
    --scaler S \
    --seq_len 36 \
    --label_len 18 \
    --pred_len $pred_len \
    --d_model 512 \
    --seg_len 3 \
    --dropout 0.5 \
    --test_start_date '2023-05-16 09:00:00' \
    --des 'Paper '$market' Day' \
    --gpu 0
done

done