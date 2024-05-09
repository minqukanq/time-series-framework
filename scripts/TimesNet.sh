model_name=TimesNet

python run.py \
    --is_training 1 \
    --task_name forecasting \
    --model $model_name \
    --id test_run \
    --freq h \
    --scaler S \
    --features S \
    --e_layers 2 \
    --factor 3 \
    --c_in 1 \
    --c_out 1 \
    --d_model 32 \
    --d_ff 32 \
    --top_k 4 \
    --epochs 10 \
    --des 'timesnet test run'