from dataclasses import dataclass


@dataclass
class PreprocessConfig:
    dataset_size: int
    batch_size: int
    force_cpu: bool
    dataset: str


preprocess_config = PreprocessConfig(
    dataset_size=100,
    batch_size=16,
    force_cpu=True,
    dataset="python-pytorch",
)


@dataclass
class ClassifyConfig:
    force_cpu: bool
    dataset_size: int
    batch_size: int
    grad_accum_step: int
    test_split_ratio: float
    val_split_ratio: float


classify_config = ClassifyConfig(
    dataset_size=-1,
    force_cpu=True,
    batch_size=32,
    grad_accum_step=1,
    test_split_ratio=0.1,
    val_split_ratio=0.1,
)
