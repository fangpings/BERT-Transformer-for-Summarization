import torch
import argparse
import logging
import os
import json

from flask import Flask, request
from preprocess import convert_one_example
from model import BertAbsSum
from pytorch_pretrained_bert.tokenization import BertTokenizer


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The path to trained model.")
parser.add_argument("--config_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The path to config file.")                    
parser.add_argument("--bert_model", 
                    default=None, 
                    type=str, 
                    required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.") 
parser.add_argument('--address',
                    default='0.0.0.0',
                    type=str,
                    required=False,
                    help='Designate which address your app will listen to. Default is 0.0.0.0 which means listen to all address.')      
parser.add_argument('--port',
                    default=8080,
                    type=int,
                    required=False,
                    help='Designate which port your app will listen to. Default is 8080.')           
parser.add_argument("--max_src_len",
                    default=130,
                    type=int,
                    help="Max sequence length for source text. Sequences will be truncated or padded to this length")

args = parser.parse_args()
app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open(args.config_path, 'r') as f:
    config = json.load(f)
model = BertAbsSum(args.bert_model, config['decoder_config'], device)
model.load_state_dict(torch.load(args.model_path))
model.to(device)

tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, 'vocab.txt'))

'''
    POST data format:
    {
        'text': 'Your text'
    }
'''
@app.route('/summarization', methods=['GET', 'POST'])
def do_sum():
    logger.info('Get request')
    if request.method == 'POST':
        if not request.is_json:
            return 'Only json format is supported'
        data = request.get_json()
        if 'text' not in data:
            return 'Your POST data must contain \'text\' key'
        text = data['text']
        src, src_mask = convert_one_example(text, args.max_src_len, tokenizer)
        pred, _ = model.beam_decode(src.to(device), src_mask.to(device), 3, 3)
        return "".join(tokenizer.convert_ids_to_tokens(pred[0][0])).split('[SEP]')[0] + '\n'
    else:
        return 'GET method is not supported here.'

if __name__ == "__main__":
    logger.info(f'Listening to {args.address} at {args.port}...')
    app.run(args.address, args.port)






