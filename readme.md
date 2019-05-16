## BERT-Transformer for Abstractive Text Summarization
It is a Pytorch implementation for abstractive text summarization model using BERT as encoder and transformer decoder as decoder. It tries to use bert encoder in generative tasks. The Pytorch Bert implementation is brought from [pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) and Transformer implementaion form [attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch). It has been tested on LCSTS Chinese dataset and achieved good results.

## Preparation
### Dataset
We use LCSTS dataset for Chinese task. If you also use this dataset, you can just download the original data into data/raw_data and run data.py. If you use other data set, you should process the dataset into following csv format. Every line stands for one entry in index, summarization, source order.
```
%index%\t%summarization%\t%source%\n
```
### BERT pretrained model
You should download the BERT pretrained model yourself into pretrained_model/ directory. The organization should like this
```
pretrained_model/
├── bert_config.json
├── pytorch_model.bin
└── vocab.txt
```
See the pytorch-pretrained-BERT repo for details.

## Usage
### Training
For training, just run train.sh, there are 2 parameters that you have to specify.

`--data_dir` You have to designate your data directory and it must contains train.csv. If eval.csv is not in the same directory, not evaluation will be carried out during the training.

`--bert_model` Also, you must designate your bert pretrained model directory.

The rest are optional.

`--GPU_index` parameter allows you to choose which GPU to use.

`--output_dir` designate the directory where the model is saved into. If it is not designated, no model will be saved

There are also other optional parameters that you can change. You can refer to train.py. Detailed description for each parameter is on the top of the file.

### Evaluation
For evaluation, just run predict.sh

It is mostly the same as train.sh, but you have to specify 5 required parameters.

`--model_path` The path to trained model for evaluation.

`--config_path` The path to configuration file. Default is inside the model_path.

`--eval_path` The path to evaluation data path.

`--bert_model` The path to bert pretrained model.

`--result_path` The path where you save your results.

