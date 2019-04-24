python main.py \
    --data_dir data/processed_data\
    --bert_model pretrained_model\
    --GPU_index "0,1,2,3"\
    --train_batch_size 32\
    --num_train_epochs 10\
    --print_every 100\
    --draft_only\

