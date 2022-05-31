#%%
import json
import pandas as pd

df = pd.read_json('data/derived/python-pytorch.json')

from transformers import pipeline

extractor = pipeline("feature-extraction")

print(extractor("Why are there so many cars in Berlin?"))

df['features'] = extractor(df['content'])
df.head()