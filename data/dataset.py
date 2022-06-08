import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class RawDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, column_name: str):
        self.dataset = dataset

        self.sorted_indices = sorted(
            range(len(self.dataset)), key=lambda k: len(self.dataset[column_name][k])
        )

        self.sorted_lines = np.array(self.dataset[column_name])[self.sorted_indices]

    def __len__(self):
        return len(self.sorted_lines)

    def __getitem__(self, idx):
        return self.sorted_lines[idx]

    def get_sort_indices(self):
        return self.sorted_indices
