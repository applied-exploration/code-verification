from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    save_every: int
    dataset_size: int
    batch_size: int


preprocess_config = PreprocessConfig(save_every=1, dataset_size=-1, batch_size=64)
