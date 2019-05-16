import torch
import re
import numpy as np

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def rouge(hyp, ref, n):
    scores = []
    for h, r in zip(hyp, ref):
        r = re.sub(r'[UNK]', '', r)
        r = re.sub(r'[’!"#$%&\'()*+,-./:：？！《》;<=>?@[\\]^_`{|}~]+', '', r)
        r = re.sub(r'\d', '', r)
        r = re.sub(r'[a-zA-Z]', '', r)
        count = 0
        match = 0
        for i in range(len(r) - n):
            gram = r[i:i + n]
            if gram in h:
                match += 1
            count += 1
        scores.append(match / count)
    return np.average(scores)

if __name__ == "__main__":
    hyp = ['交大闵行校区一实验室发生硫化氢泄漏事故中无学生伤亡', '[UNK]史上最严的环保法[UNK]']
    ref = ['上海交大闵行校区：实验室换瓶时硫化氢泄漏送货员身亡', '#2015全国两会#傅莹：[UNK]史上最严[UNK]环保法是[UNK]有牙齿[UNK]的']
    print(rouge(hyp, ref, 2))

        