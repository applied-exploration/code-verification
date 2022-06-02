import pandas as pd
from config import PreprocessConfig, preprocess_config
from data.dataset import get_features, preprocess_source, get_features_batched
from transformers import pipeline
from tqdm import tqdm
import torch as t


def run_preprocessing(config: PreprocessConfig):
    print("| Running preprocessing...")
    df = pd.read_json("data/derived/python-pytorch.json")[: config.dataset_size]

    print("| Setting up pipeline...")
    pipe = pipeline(
        "feature-extraction", framework="pt", model="microsoft/codebert-base"
    )
    source_code_all = df["content"].to_numpy()

    print("| Converting original source to features...")
    feature_dict = dict()
    for i in range(len(df)):
        feature_dict[i] = get_features_batched(source_code_all[i], pipe, config)

        if i % config.save_every == 0:
            print(f"| Saving features to file...")
            t.save(feature_dict, f"data/temporary/features_{i}.pt")
        elif i == len(df) - 1:
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
