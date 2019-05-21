import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import argparse
import logging
import os
import json
import time
import torch.nn.functional as F
from preprocess import LCSTSProcessor, convert_examples_to_features, create_dataset
from model import BertAbsSum
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel
from pytorch_pretrained_bert.optimization import BertAdam
from tqdm import tqdm, trange
from transformer import Constants

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TRAIN_FILE = 'train_big.csv'

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--data_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The input data path. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", 
                    default=None, 
                    type=str, 
                    required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")

# Opitional paramete
parser.add_argument("--GPU_index",
                    default='-1',
                    type=str,
                    help="Designate the GPU index that you desire to use. Should be str. -1 for using all available GPUs.")
parser.add_argument("--learning_rate",
                    default=5e-5,
                    type=float,
                    help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                    default=3.0,
                    type=float,
                    help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                    default=0.1,
                    type=float,
                    help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
parser.add_argument("--max_src_len",
                    default=130,
                    type=int,
                    help="Max sequence length for source text. Sequences will be truncated or padded to this length")
parser.add_argument("--max_tgt_len",
                    default=30,
                    type=int,
                    help="Max sequence length for target text. Sequences will be truncated or padded to this length")
parser.add_argument("--train_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for training.")
parser.add_argument("--decoder_config",
                    default=None,
                    type=str,
                    help="Configuration file for decoder. Must be in JSON format.")
parser.add_argument("--print_every",
                    default=100,
                    type=int,
                    help="Print loss every k steps.")
parser.add_argument('--gradient_accumulation_steps',
                    type=int,
                    default=1,
                    help="Number of updates steps to accumulate before performing a backward/update pass.")
# parser.add_argument('--draft_only',
#                     action='store_true',
#                     help="Only use stage 1 to generate drafts.")
parser.add_argument("--output_dir",
                    default=None,
                    type=str,
                    help="The output directory where the model predictions and checkpoints will be written.")


'''
TODO: beam/greedy search, eval, copy, rouge
'''

def cal_performance(logits, ground, smoothing=True):
    ground = ground[:, 1:]
    logits = logits.view(-1, logits.size(-1))
    ground = ground.contiguous().view(-1)

    loss = cal_loss(logits, ground, smoothing=smoothing)

    pad_mask = ground.ne(Constants.PAD)
    pred = logits.max(-1)[1]
    correct = pred.eq(ground)
    correct = correct.masked_select(pad_mask).sum().item()
    return loss, correct

def cal_loss(logits, ground, smoothing=True):
    def label_smoothing(logits, labels):
        eps = 0.1
        num_classes = logits.size(-1)

        # >>> z = torch.zeros(2, 4).scatter_(1, torch.tensor([[2], [3]]), 1.23)
        # >>> z
        # tensor([[ 0.0000,  0.0000,  1.2300,  0.0000],
        #        [ 0.0000,  0.0000,  0.0000,  1.2300]])
        one_hot = torch.zeros_like(logits).scatter(1, labels.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (num_classes - 1)
        log_prb = F.log_softmax(logits, dim=1)
        non_pad_mask = ground.ne(Constants.PAD)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()
        return loss
    if smoothing:
        loss = label_smoothing(logits, ground)
    else:
        loss = F.cross_entropy(logits, ground, ignore_index=Constants.PAD)
    
    return loss


if __name__ == "__main__":
    args = parser.parse_args()

    if args.GPU_index != '-1':
        os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_index
    if not torch.cuda.is_available():
        raise ValueError('CUDA is not available.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    assert args.train_batch_size % n_gpu == 0
    logger.info(f'Using device:{device}, n_gpu:{n_gpu}')

    if args.output_dir is not None:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        model_path = os.path.join(args.output_dir, time.strftime('model_%m-%d-%H:%M:%S', time.localtime()))
        os.mkdir(model_path)
        logger.info(f'Saving model to {model_path}.')

    if args.decoder_config is not None:
        with open(args.decoder_config, 'r') as f:
            decoder_config = json.load(f)
    else:
        with open(os.path.join(args.bert_model, 'bert_config.json'), 'r') as f:
            bert_config = json.load(f)
            decoder_config = {}
            decoder_config['len_max_seq'] = args.max_tgt_len
            decoder_config['d_word_vec'] = bert_config['hidden_size']
            decoder_config['n_layers'] = 8
            decoder_config['n_head'] = 12
            decoder_config['d_k'] = 64
            decoder_config['d_v'] = 64
            decoder_config['d_model'] = bert_config['hidden_size']
            decoder_config['d_inner'] = bert_config['hidden_size']
            decoder_config['vocab_size'] = bert_config['vocab_size']
    
    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))
    
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
                
    # train data preprocess
    processor = LCSTSProcessor()
    tokenizer = BertTokenizer.from_pretrained(os.path.join(args.bert_model, 'vocab.txt'))
    logger.info('Loading train examples...')
    if not os.path.exists(os.path.join(args.data_dir, TRAIN_FILE)):
        raise ValueError(f'train.csv does not exist.')
    train_examples = processor.get_examples(os.path.join(args.data_dir, TRAIN_FILE))
    num_train_optimization_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    logger.info('Converting train examples to features...')
    train_features = convert_examples_to_features(train_examples, args.max_src_len, args.max_tgt_len, tokenizer)
    example = train_examples[0]
    example_feature = train_features[0]
    logger.info("*** Example ***")
    logger.info("guid: %s" % (example.guid))
    logger.info("src text: %s" % example.src)
    logger.info("src_ids: %s" % " ".join([str(x) for x in example_feature.src_ids]))
    logger.info("src_mask: %s" % " ".join([str(x) for x in example_feature.src_mask]))
    logger.info("tgt text: %s" % example.tgt)
    logger.info("tgt_ids: %s" % " ".join([str(x) for x in example_feature.tgt_ids]))
    logger.info("tgt_mask: %s" % " ".join([str(x) for x in example_feature.tgt_mask]))
    logger.info('Building dataloader...')
    train_data = create_dataset(train_features)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size, drop_last=True)

    # eval data preprocess
    if not os.path.exists(os.path.join(args.data_dir, 'eval.csv')):
        logger.info('No eval data found in data directory. Eval will not be performed.')
        eval_dataloader = None
    else:
        logger.info('Loading eval dataset...')
        eval_examples = processor.get_examples(os.path.join(args.data_dir, 'eval.csv'))
        logger.info('Converting eval examples to features...')
        eval_features = convert_examples_to_features(eval_examples, args.max_src_len, args.max_tgt_len, tokenizer)
        eval_data = create_dataset(eval_features)
        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.train_batch_size, drop_last=True)


    # model
    model = BertAbsSum(args.bert_model, decoder_config, device)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=0.1,
                         t_total=num_train_optimization_steps)
    
    # training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    global_step = 0
    for i in range(int(args.num_train_epochs)):
        # do training
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            logits = model(*batch)
            loss, _ = cal_performance(logits, batch[2])
            if n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += batch[0].size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
            if (step + 1) % args.print_every == 0:
                logger.info(f'Epoch {i}, step {step}, loss {loss.item()}.')
                logger.info(f'Ground: {"".join(tokenizer.convert_ids_to_tokens(batch[2][0].cpu().numpy()))}')
                logger.info(f'Generated: {"".join(tokenizer.convert_ids_to_tokens(logits[0].max(-1)[1].cpu().numpy()))}')
        # do evaluation
        if args.output_dir is not None:
            state_dict = model.module.state_dict() if n_gpu > 1 else model.state_dict()
            torch.save(state_dict, os.path.join(model_path, 'BertAbsSum_{}.bin'.format(i)))
            logger.info('Model saved')
        if eval_dataloader is not None:
            model.eval()
            batch = next(iter(eval_dataloader))
            batch = tuple(t.to(device) for t in batch)
            # beam_decode
            if n_gpu > 1:
                pred, _ = model.module.beam_decode(batch[0], batch[1], 3, 3)
            else:
                pred, _ = model.beam_decode(batch[0], batch[1])
            logger.info(f'Source: {"".join(tokenizer.convert_ids_to_tokens(batch[0][0].cpu().numpy()))}')
            logger.info(f'Beam Generated: {"".join(tokenizer.convert_ids_to_tokens(pred[0][0]))}')
            # if n_gpu > 1:
            #     pred = model.module.greedy_decode(batch[0], batch[1])
            # else:
            #     pred = model.greedy_decode(batch[0], batch[1])
            # logger.info(f'Beam Generated: {tokenizer.convert_ids_to_tokens(pred[0].cpu().numpy())}')
        logger.info(f'Epoch {i} finished.')
    with open(os.path.join(args.bert_model, 'bert_config.json'), 'r') as f:
        bert_config = json.load(f)
    config = {'bert_config': bert_config, 'decoder_config': decoder_config}
    with open(os.path.join(model_path, 'config.json'), 'w') as f:
        json.dump(config, f)
    logger.info('Training finished')


