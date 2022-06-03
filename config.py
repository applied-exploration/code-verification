from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    save_every: int
    dataset_size: int
    batch_size: int
    force_cpu: bool
    dataset: str
    rearrange_order:bool


preprocess_config = PreprocessConfig(
    save_every=1,
    dataset_size=5,
    batch_size=8,
    force_cpu=True,
    dataset="python-pytorch",
    rearrange_order=True
)
