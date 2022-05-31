#%%
import json
import pandas as pd

# with open('data/derived/python-pytorch.json') as json_file:
#     data = json.load(json_file)

# # %%
# print(data[0])

df = pd.read_json('data/derived/python-pytorch.json')


# %%
from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# %%
classifier("Why are there so many cars in Berlin?")

# %%
classifier("No me gusta los pantallones")

# %%
