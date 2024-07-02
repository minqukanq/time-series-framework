model_name=ModernTCN

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
    --label_len 0 \
    --pred_len $pred_len \
    --ffn_ratio 1 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 1 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.0 \
    --small_kernel_merged False \
    --des 'Paper '$market' Min 30' \
    --gpu 1
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
    --label_len 0 \
    --pred_len $pred_len \
    --ffn_ratio 1 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 1 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.0 \
    --small_kernel_merged False \
    --des 'Paper '$market' Hour 1' \
    --gpu 1
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
    --label_len 0 \
    --pred_len $pred_len \
    --ffn_ratio 1 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 1 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.0 \
    --small_kernel_merged False \
    --test_start_date '2023-05-16 01:00:00' \
    --des 'Paper '$market' Hour 4' \
    --gpu 1
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
    --label_len 0 \
    --pred_len $pred_len \
    --ffn_ratio 1 \
    --patch_size 8 \
    --patch_stride 4 \
    --num_blocks 1 \
    --large_size 51 \
    --small_size 5 \
    --dims 64 64 64 64 \
    --head_dropout 0.0 \
    --small_kernel_merged False \
    --test_start_date '2023-05-16 09:00:00' \
    --des 'Paper '$market' Day' \
    --gpu 1
done


done