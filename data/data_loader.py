import pandas as pd
import torch as t
from data.dataset import SourceCodeDataset


def load_data():
    df = pd.read_json("data/derived/python-pytorch.json")
    all_features = t.load("data/derived/features_all.pt")

    dataset = SourceCodeDataset(df, all_features)

    return dataset
