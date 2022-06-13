import pandas as pd
import torch as t

from config import ClassifyConfig


def load_data(dataset_size: int) -> pd.DataFrame:
    return pd.read_json(
        f"data/original/TSSB-3M/file-0.jsonl.gz", lines=True, compression="gzip"
    )[: dataset_size]
