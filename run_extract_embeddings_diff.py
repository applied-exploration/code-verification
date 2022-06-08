import pandas as pd
from transformers.pipelines.base import Pipeline
from config import PreprocessConfig, preprocess_config
from data.dataset import RawDataset
from transformers.pipelines import pipeline
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from extractors.pipeline import MyFeatureExtractionPipeline
import torch as t


def extract_embeddings_from_pipeline(
    pipe: Pipeline, dataset: RawDataset, batch_size: int, length: int
) -> t.Tensor:
    # pipeline padding adds a unique padding token, that also gets passed into inference, hence we get a feature tensor for the padding tokens aswell.
    embeddings = [
        t.mean(t.tensor(out)[0], dim=0, keepdim=False)
        for out in tqdm(
            pipe(dataset, batch_size=batch_size, padding=True),
            total=length,
        )
    ]

    print("| Restore original order for embeddings...")
    embeddings: t.Tensor = pad_sequence(embeddings, batch_first=True)
    embeddings = embeddings[dataset.get_sort_indices()]
    return embeddings


def run_extract_embeddings(config: PreprocessConfig):
    device = 0 if t.cuda.is_available() else -1

    if config.force_cpu:
        print("| Setting device to cpu...")
        device = -1

    print("| Running preprocessing...")
    df = pd.read_json(
        f"data/original/TSSB-3M/file-0.jsonl.gz", lines=True, compression="gzip"
    )[: config.dataset_size]

    print("| Setting up pipeline...")
    pipe = pipeline(
        "feature-extraction",
        framework="pt",
        model="microsoft/codebert-base",
        device=device,
        pipeline_class=MyFeatureExtractionPipeline,
    )

    print("| Converting original source to embeddings...")
    before_dataset = RawDataset(df, "before")
    before_embeddings = extract_embeddings_from_pipeline(
        pipe, before_dataset, config.batch_size, len(df)
    )

    after_dataset = RawDataset(df, "after")
    after_embeddings = extract_embeddings_from_pipeline(
        pipe, after_dataset, config.batch_size, len(df)
    )

    diff_embeddings = after_embeddings - before_embeddings

    print("| Saving embeddings to disk...")
    t.save(diff_embeddings, "data/derived/embeddings_flat.pt")

    print("✅ Feature preprocessing done...")


if __name__ == "__main__":
    run_extract_embeddings(preprocess_config)
