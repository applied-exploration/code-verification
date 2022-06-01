#%%
import json
import pandas as pd
import torch as t
from transformers import pipeline
import re
from utils.text_process import strip_comments
from data.dataset import SourceCodeDataset

df = pd.read_json('data/derived/python-pytorch.json')[:2]

new_dataset = SourceCodeDataset(df)

first_element = next(iter(new_dataset))
print("")



