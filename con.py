from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    save_every: int = 1
    dataset_size: int = -1


preprocess_config = PreprocessConfig(save_every=1, dataset_size=5)
