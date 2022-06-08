from data.data_loader import load_data
from config import preprocess_config, PreprocessConfig
from extractors.split_utils import add_negative_cases, split_data
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_metric, load_dataset, Dataset
import numpy as np


def train_classifier(preprocess_config: PreprocessConfig):
    df = load_data(preprocess_config)
    df = add_negative_cases(df)
    train, val, test = split_data(df)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    def tokenize_function(examples):
        return tokenizer(examples["source_code"], padding="max_length", truncation=True)

    train_dataset = Dataset.from_pandas(train)
    val_dataset = Dataset.from_pandas(val)

    small_train_dataset = train_dataset.map(tokenize_function, batched=True)
    small_eval_dataset = val_dataset.map(tokenize_function, batched=True)

    # small_train_dataset = tokenized_datasets.shuffle(seed=42).select(range(1000))
    # small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", num_labels=2
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch"
    )
    metric = load_metric("accuracy")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    train_classifier(preprocess_config)
