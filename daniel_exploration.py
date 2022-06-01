#%%
import json
import pandas as pd
import torch as t
from transformers import pipeline
import re
from utils.text_process import remove_comments_and_docstrings, stripComments

df = pd.read_json('data/derived/python-pytorch.json')


list_of_objects = df.iloc[0]['content'].split("\n")
stripCommentsstripped_lines = [stripComments(line) for line in list_of_objects]
no_empty_lines = [line for line in stripped_lines if line != '']

longest_line = len(max(list_of_objects, key=len))


extractor = pipeline("feature-extraction", framework='pt')
features = t.tensor([extractor(line) for line in list_of_objects])

print("")



