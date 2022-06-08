from data.data_loader import load_data
from config import preprocess_config, PreprocessConfig
from extractors.negative_cases import add_negative_cases


def train_classifier(preprocess_config: PreprocessConfig):
    df = load_data(preprocess_config)
    df = add_negative_cases(df)


if __name__ == "__main__":
    train_classifier(preprocess_config)
