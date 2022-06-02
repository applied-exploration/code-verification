import pandas as pd
from config import PreprocessConfig, preprocess_config
from data.dataset import get_features, preprocess_source, get_features_batched
from transformers import pipeline
from tqdm import tqdm
import torch as t


def run_preprocessing_batched(config: PreprocessConfig):
    print("| Running preprocessing...")
    df = pd.read_json("data/derived/python-pytorch.json")[: config.dataset_size]

    get_features_batched(df)


def run_preprocessing(config: PreprocessConfig):
    print("| Running preprocessing...")
    df = pd.read_json("data/derived/python-pytorch.json")[: config.dataset_size]

    print("| Loading Huggingface pipeline...")
    all_source = df["content"].to_numpy()
    extractor = pipeline(
        "feature-extraction", framework="pt", model="distilbert-base-cased"
    )

    print("| Converting original source to features...")
    feature_dict = dict()
    for i, original in enumerate(tqdm(all_source)):
        feature_dict[i] = get_features(preprocess_source(original), extractor)

        if i % config.save_every == 0:
            print(f"| Saving features to file...")
            t.save(feature_dict, f"data/temporary/features_{i}.pt")

    print("| Combining features and saving to disk...")
    combined_dict = dict()
    for i in range(len(all_source)):
        combined_dict.update(t.load(f"data/temporary/features_{i}.pt"))

    print("| Saving combined feature_dicts to disk...")
    t.save(combined_dict, "data/derived/features_all.pt")

    print("✅ Feature preprocessing done...")


if __name__ == "__main__":
    run_preprocessing_batched(preprocess_config)
