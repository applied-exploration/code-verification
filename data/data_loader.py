import pandas as pd
import torch as t
# from data.dataset import SourceCodeDataset
from config import PreprocessConfig


# def load_data():
#     df = pd.read_json("data/derived/python-pytorch.json")
#     all_embeddings = t.load("data/derived/embeddings_per_file.pt")

#     # dataset = SourceCodeDataset(df, all_embeddings)

#     # return dataset


def load_data(config: PreprocessConfig) -> pd.DataFrame:

    df = pd.read_json(f"data/original/TSSB-3M/file-0.jsonl", lines=True)[
        : config.dataset_size
    ]
    return df