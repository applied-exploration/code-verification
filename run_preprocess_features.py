import pandas as pd
from config import PreprocessConfig, preprocess_config
from data.dataset import get_features_batched
from transformers import pipeline
import torch as t


def shorten_dataset(config: PreprocessConfig):
    df = pd.read_json("data/derived/python-pytorch.json")[2 : config.dataset_size + 2]
    df.to_json("data/derived/python-pytorch-short.json")


def run_preprocessing(config: PreprocessConfig):
    device = 0 if t.cuda.is_available() else -1

    if config.force_cpu:
        print("| Setting device to cpu...")
        device = -1

    print("| Running preprocessing...")
    df = pd.read_json(f"data/derived/{config.dataset}.json")[: config.dataset_size]

    print("| Setting up pipeline...")
    pipe = pipeline(
        "feature-extraction",
        framework="pt",
        model="microsoft/codebert-base",
        device=device,
    )
    source_code_all = df["content"].to_numpy()

    print("| Converting original source to features...")
    feature_dict = dict()
    for i in range(len(df)):
        feature_dict[i] = get_features_batched(source_code_all[i], pipe, config)

        if i % config.save_every == 0:
            print(f"| Saving features to file...")
            t.save(feature_dict, f"data/temporary/features_{i}.pt")
        if i == len(df) - 1:
            t.save(feature_dict, "data/derived/features_all.pt")

    print("✅ Feature preprocessing done...")


def combine_temporary_preprocessed_features(num_files: int):
    print("| Combining features and saving to disk...")
    combined_dict = dict()
    for i in range(num_files):
        combined_dict.update(t.load(f"data/temporary/features_{i}.pt"))

    print("| Saving combined feature_dicts to disk...")
    t.save(combined_dict, "data/derived/features_all.pt")

    print("✅ Feature preprocessing done...")


if __name__ == "__main__":
    run_preprocessing(preprocess_config)
