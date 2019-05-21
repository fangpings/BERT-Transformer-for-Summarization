import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import csv
import os
import logging
from utils import convert_to_unicode
from pytorch_pretrained_bert.tokenization import BertTokenizer
from tqdm import tqdm, trange

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/test example."""

    def __init__(self, guid, src, tgt=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            src: string. The untokenized text of the target sequence.
            tgt: (Optional) string. The untokenized text of the target.
        """
        self.guid = guid
        self.src = src
        self.tgt = tgt

class InputFeatures():
    """A single set of features of data."""

    def __init__(self, src_ids, src_mask, tgt_ids, tgt_mask):
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.tgt_ids = tgt_ids
        self.tgt_mask = tgt_mask 

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

class LCSTSProcessor(DataProcessor):
    """Processor for the LCSTS data set."""

    def get_examples(self, data_path):
        """See base class."""
        return self._create_examples(self._read_tsv(data_path))

    def _create_examples(self, lines):
        examples = [] 
        for data in lines:
            # lines: id, summary, text
            guid = data[0]
            src = convert_to_unicode(data[2])
            tgt = convert_to_unicode(data[1])
            examples.append(InputExample(guid=guid, src=src, tgt=tgt))
        return examples

def convert_examples_to_features(examples, src_max_seq_length, tgt_max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.

    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc='examples')):
        src_tokens = tokenizer.tokenize(example.src)
        tgt_tokens = tokenizer.tokenize(example.tgt)
        if len(src_tokens) > src_max_seq_length - 2:
            src_tokens = src_tokens[:(src_max_seq_length - 2)]
        if len(tgt_tokens) > tgt_max_seq_length - 2:
            tgt_tokens = tgt_tokens[:(tgt_max_seq_length - 2)]
        src_tokens = ["[CLS]"] + src_tokens + ["[SEP]"]
        tgt_tokens = ["[CLS]"] + tgt_tokens + ["[SEP]"]
        # no need to generate segment ids here because if we do not provide
        # bert model will generate dafault all-zero ids for us
        # and we regard single text as one sentence

        src_ids = tokenizer.convert_tokens_to_ids(src_tokens)
        tgt_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        src_mask = [1] * len(src_ids)
        tgt_mask = [1] * len(tgt_ids)
        # Zero-pad up to the sequence length.
        src_padding = [0] * (src_max_seq_length - len(src_ids))
        tgt_padding = [0] * (tgt_max_seq_length - len(tgt_ids))
        src_ids += src_padding
        src_mask += src_padding
        tgt_ids += tgt_padding
        tgt_mask += tgt_padding

        assert len(src_ids) == src_max_seq_length
        assert len(tgt_ids) == tgt_max_seq_length

        features.append(InputFeatures(src_ids=src_ids,
                                      src_mask=src_mask,
                                      tgt_ids=tgt_ids,
                                      tgt_mask=tgt_mask))
    return features

def create_dataset(features):
    all_src_ids = torch.tensor([f.src_ids for f in features], dtype=torch.long)
    all_src_mask = torch.tensor([f.src_mask for f in features], dtype=torch.long)
    all_tgt_ids = torch.tensor([f.tgt_ids for f in features], dtype=torch.long)
    all_tgt_mask = torch.tensor([f.tgt_mask for f in features], dtype=torch.long)
    train_data = TensorDataset(all_src_ids, all_src_mask, all_tgt_ids, all_tgt_mask)
    return train_data

def convert_one_example(text, src_max_seq_length, tokenizer):
    src_tokens = tokenizer.tokenize(text)
    if len(src_tokens) > src_max_seq_length - 2:
        src_tokens = src_tokens[:(src_max_seq_length - 2)]
    src_tokens = ["[CLS]"] + src_tokens + ["[SEP]"]

    src_ids = tokenizer.convert_tokens_to_ids(src_tokens)

    src_mask = [1] * len(src_ids)
    src_padding = [0] * (src_max_seq_length - len(src_ids))
    src_ids += src_padding
    src_mask += src_padding

    return torch.tensor([src_ids]), torch.tensor([src_mask])

if __name__ == "__main__":
    # processor = LCSTSProcessor()
    tokenizer = BertTokenizer.from_pretrained(os.path.join('pretrained_model', 'vocab.txt'))    # examples = processor.get_train_examples('data/processed_data')
    # features = convert_examples_to_features(examples, 130, 20, tokenizer)

    print(convert_one_example('新京报讯（记者 梁辰）5月20日，针对外媒报道因美国政府将华为列入实体名单，而谷歌已暂停与华为部分业务往来的消息，谷歌中国通过邮件回复记者称，“我们正在遵守这一命令，并审查其影响”。', 130, tokenizer))
    
    