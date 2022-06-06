from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    save_every: int
    dataset_size: int
    batch_size: int
    force_cpu: bool
    dataset: str


preprocess_config = PreprocessConfig(
    save_every=1,
    dataset_size=1000,
    batch_size=16,
    force_cpu=True,
    dataset="python-pytorch",
)
