import pandas as pd
import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import pipeline

from utils.text_process import strip_comments
from tqdm import tqdm


class RawDataset(Dataset):
    def __init__(self, all_code: pd.DataFrame):
        self.all_code = all_code

    def __len__(self):
        return len(self.all_code)

    def __getitem__(self, idx):
        return preprocess_source(self.all_code.iloc[idx]["content"])


class SourceCodeDataset(Dataset):
    def __init__(self, all_code: pd.DataFrame, all_features: dict):
        self.all_code = all_code
        self.all_features = all_features

    def __len__(self):
        return len(self.all_code)

    def __getitem__(self, idx):
        source_code = preprocess_source(self.all_code.iloc[idx]["content"])
        features = self.all_features[idx]

        return source_code, features


class SourceCodeDatasetRuntime(Dataset):
    def __init__(self, all_code: pd.DataFrame):
        self.all_code = all_code
        self.extractor = pipeline("feature-extraction", framework="pt")

    def __len__(self):
        return len(self.all_code)

    def __getitem__(self, idx):
        source_code = preprocess_source(self.all_code.iloc[idx]["content"])
        features = get_features(source_code)

        return source_code, features


def get_features(original: str, extractor) -> t.Tensor:
    feature_tensors = [t.tensor(extractor(line)[0]) for line in tqdm(original)]
    features = pad_sequence(feature_tensors, batch_first=False)
    return features


def preprocess_source(original: str) -> list[str]:
    split_by_line = original.split("\n")
    stripped_lines = [strip_comments(line) for line in split_by_line]
    no_empty_lines = [line for line in stripped_lines if line != ""]

    return no_empty_lines


def get_features_batched(df: pd.DataFrame):
    dataset = RawDataset(df)

    pipe = pipeline(
        "feature-extraction", framework="pt", device=0, model="distilbert-base-cased"
    )
    for out in tqdm(pipe(dataset, batch_size=64), total=len(dataset)):
        print(out)
