export CUDA_VISIBLE_DEVICES=2

python predict.py \
    --model_path output/model_04-25-14:04:10/BertAbsSum_4.bin\
    --config_path output/model_04-25-14:04:10/config.json\
    --eval_path data/processed_data/eval.csv\
    --bert_model pretrained_model\
    --result_path result
