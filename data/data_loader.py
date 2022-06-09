import pandas as pd
import torch as t

from config import PreprocessConfig


def load_data(config: PreprocessConfig) -> pd.DataFrame:
    df = pd.read_json(
        f"data/original/TSSB-3M/file-0.jsonl.gz", lines=True, compression="gzip"
    )[: config.dataset_size]
    return df
