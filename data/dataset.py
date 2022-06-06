import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class RawDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

        self.sorted_indices = sorted(
            range(len(self.dataset)), key=lambda k: len(self.dataset["content"][k])
        )

        self.sorted_lines = np.array(self.dataset["content"])[self.sorted_indices]

    def __len__(self):
        return len(self.sorted_lines)

    def __getitem__(self, idx):
        return self.sorted_lines[idx]

    def get_sort_indices(self):
        return self.sorted_indices


# I'm not sure if we'll ever need this or if I should keep maintaing it? As we don't have an objective function to feed a model with the embeddings right now
# class SourceCodeDataset(Dataset):
#     def __init__(self, all_code: pd.DataFrame, all_embeddings: dict):
#         self.all_code = all_code
#         self.all_embeddings = all_embeddings

#     def __len__(self):
#         return len(self.all_code)

#     def __getitem__(self, idx):
#         source_code = preprocess_source_code(self.all_code.iloc[idx]["content"])
#         embeddings = self.all_embeddings[idx]

#         return source_code, embeddings
