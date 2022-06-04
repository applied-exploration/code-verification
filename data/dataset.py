import pandas as pd
import numpy as np
import torch as t
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from typing import List
from config import PreprocessConfig
from utils.text_process import strip_comments
import re


class RawDataset(Dataset):
    def __init__(self, single_code: str):
        self.single_code = preprocess_source(single_code)

        self.sorted_indices = sorted(
            range(len(self.single_code)), key=lambda k: len(self.single_code[k])
        )

        self.sorted_lines = np.array(self.single_code)[self.sorted_indices]

    def __len__(self):
        return len(self.sorted_lines)

    def __getitem__(self, idx):
        return self.sorted_lines[idx]

    def get_sort_indices(self):
        return self.sorted_indices


class SourceCodeDataset(Dataset):
    def __init__(self, all_code: pd.DataFrame, all_embeddings: dict):
        self.all_code = all_code
        self.all_embeddings = all_embeddings

    def __len__(self):
        return len(self.all_code)

    def __getitem__(self, idx):
        source_code = preprocess_source(self.all_code.iloc[idx]["content"])
        embeddings = self.all_embeddings[idx]

        return source_code, embeddings


def preprocess_source(original: str) -> List[str]:
    strip_tabs = re.sub(
        r" +[\t]*", " ", original
    )  # Strips tabs and multiple white spaces
    split_by_line = strip_tabs.split("\n")

    stripped_lines = [strip_comments(line) for line in split_by_line]
    no_empty_lines = [line for line in stripped_lines if line != ""]

    return no_empty_lines


def get_embeddings_batched(
    source_code: str, pipe, config: PreprocessConfig
) -> t.Tensor:

    dataset = RawDataset(source_code)

    feature_tensors = []

    # pipeline padding adds a unique padding token, that also gets passed into inference, hence we get a feature tensor for the padding tokens aswell.
    for out in pipe(dataset, batch_size=config.batch_size, padding=True):
        # Shape is: num_lines | num_words in line | feature_length of word (token)
        out_tensor = t.tensor(out)[0]
        avaraged_embeddings = t.mean(out_tensor, dim=0, keepdim=False)

        feature_tensors.append(avaraged_embeddings)

    if len(feature_tensors) > 0:
        single_source_embeddings = pad_sequence(feature_tensors, batch_first=True)
    else:
        single_source_embeddings = t.empty(0)

    if config.rearrange_to_original:
        original_order = dataset.get_sort_indices()
        single_source_embeddings = single_source_embeddings[original_order]

    return single_source_embeddings
