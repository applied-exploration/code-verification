import pandas as pd
import torch as t
from data.dataset import SourceCodeDataset


def load_data():
    df = pd.read_json("data/derived/python-pytorch.json")
    all_embeddings = t.load("data/derived/embeddings_per_file.pt")

    dataset = SourceCodeDataset(df, all_embeddings)

    return dataset
