import json
from nltk import ngrams
from utils.flatten import flatten
from tqdm import tqdm
from collections import Counter
from typing import Tuple

def load_json(path: str) -> list[dict]:
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

data = load_json('data/derived/python-pytorch.json')[0]
data = [d['content'] for d in data]

blacklist = set(['#', 'Data', '0.001', 'import', '(0.001)', '0.001', 'as', 'None:', 'None', '__name__', 'probably', 'argparse'])
def is_valid_ngram(ngram: Tuple) -> bool:
    return not any([not word not in blacklist for word in ngram])

grams = flatten([ngrams(d.split(),n=3) for d in tqdm(data)])
grams = [ng for ng in grams if is_valid_ngram(ng)]

print(Counter(grams).most_common(100))