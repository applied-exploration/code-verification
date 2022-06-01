import pandas as pd
import json
from data.dataset import get_features, preprocess_source
from transformers import pipeline
from tqdm import tqdm


df = pd.read_json('data/derived/python-pytorch.json')[:2]

all_source = df['content'].to_numpy()

extractor= pipeline("feature-extraction", framework='pt')
feature_dict = {i:preprocess_source(get_features(original, extractor))[1] for i, original in tqdm(enumerate(all_source))}

with open('data/derived/features.json', 'w') as f: 
    json.dump(feature_dict, f)