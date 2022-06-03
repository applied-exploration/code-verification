import pandas as pd
import numpy as np
import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import List
from config import PreprocessConfig

from utils.text_process import strip_comments
from tqdm import tqdm
import re
import math
from typing import List


class RawDataset(Dataset):
    def __init__(self, single_code: str):
        self.single_code = preprocess_source(single_code)

        self.sorted_indecies = sorted(
            range(len(self.single_code)), key=lambda k: len(self.single_code[k])
        )

        self.sorted_lines = np.array(self.single_code)[self.sorted_indecies]

    def __len__(self):
        return len(self.sorted_lines)

    def __getitem__(self, idx):
        return self.sorted_lines[idx]


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


def preprocess_source(original: str) -> List[str]:
    strip_tabs = re.sub(
        r" +[\t]*", " ", original
    )  # Strips tabs and multiple white spaces
    split_by_line = strip_tabs.split("\n")

    stripped_lines = [strip_comments(line) for line in split_by_line]
    no_empty_lines = [line for line in stripped_lines if line != ""]

    return no_empty_lines



def get_features_batched(source_code: str, pipe, config: PreprocessConfig):

    dataset = RawDataset(source_code)

    feature_tensors = []

    # pipeline padding adds a unique padding token, that also gets passed into inference, hence we get a feature tensor for the padding tokens aswell.
    for i, out in tqdm(
        enumerate(pipe(dataset, batch_size=config.batch_size, padding=True)),
        total=len(dataset),
    ):
        # Shape is: num_lines | num_words in line | feature_length of word (token)
        out_tensor = t.tensor(out)[0]
        avaraged_features = t.mean(out_tensor, dim=0, keepdim=False)
        feature_tensors.append(avaraged_features)

    single_source_features = pad_sequence(feature_tensors, batch_first=True)

    return single_source_features
