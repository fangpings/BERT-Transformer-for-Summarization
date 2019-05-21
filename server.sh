export CUDA_VISIBLE_DEVICES=2

python server.py \
    --model_path output/model_05-20-14:59:07/BertAbsSum_4.bin\
    --config_path output/model_05-20-14:59:07/config.json\
    --bert_model pretrained_model\