from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    dataset_size: int
    batch_size: int
    force_cpu: bool
    dataset: str


preprocess_config = PreprocessConfig(
    dataset_size=1000,
    batch_size=16,
    force_cpu=True,
    dataset="python-pytorch",
)
