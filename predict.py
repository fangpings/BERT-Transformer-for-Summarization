import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
import os
from tqdm import tqdm, trange
import json

from preprocess import LCSTSProcessor, convert_examples_to_features, create_dataset
from model import BertAbsSum
from pytorch_pretrained_bert.tokenization import BertTokenizer

LOG_PATH = 'log.txt'
BATCH_SIZE = 32

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
parser.add_argument("--eval_path",
                    default=None,
                    type=str,
                    required=True,
                    help="The path to the evaluation data. Should end with .tsv.")
parser.add_argument("--bert_model", 
                    default=None, 
                    type=str, 
                    required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--max_src_len",
                    default=130,
                    type=int,
                    help="Max sequence length for source text. Sequences will be truncated or padded to this length")
parser.add_argument("--max_tgt_len",
                    default=30,
                    type=int,
                    help="Max sequence length for target text. Sequences will be truncated or padded to this length")

if __name__ == "__main__":
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.config_path, 'r') as f:
        config = json.load(f)
    model = BertAbsSum(args.bert_model, config['decoder_config'], device, config['draft_only'])
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    processor = LCSTSProcessor()
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, 'vocab.txt'))
    test_examples = processor.get_examples(args.eval_path)
    test_features = convert_examples_to_features(test_examples, args.max_src_len, args.max_tgt_len, tokenizer)
    test_data = create_dataset(test_features)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE, drop_last=True)
    logger.info('Loading complete. Writing results to %s' % (LOG_PATH))

    model.eval()
    f = open(LOG_PATH, 'w', encoding='utf-8')
    for batch in tqdm(test_dataloader, desc="Iteration"):
        batch = tuple(t.to(device) for t in batch)
        pred, _ = model.beam_decode(batch[0], batch[1], 3, 3)
        src, tgt = batch[0], batch[2]
        for i in range(BATCH_SIZE):
            sample_src = 'Source: ' + "".join(tokenizer.convert_ids_to_tokens(src[i].cpu().numpy())) + '\n'
            sample_tgt = 'Ground: ' + "".join(tokenizer.convert_ids_to_tokens(tgt[i].cpu().numpy())) + '\n'
            sample_pred = 'Generated: ' + "".join(tokenizer.convert_ids_to_tokens(pred[i][0])) + '\n'
            f.write('**********\n')
            f.write(sample_src)
            f.write(sample_tgt)
            f.write(sample_pred)
    logger.info('Evaluation finished.')

        

