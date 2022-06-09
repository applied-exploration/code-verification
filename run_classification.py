from data.data_loader import load_data
from config import preprocess_config, PreprocessConfig
from extractors.split_utils import add_negative_cases, split_data
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_metric, load_dataset, Dataset, Features, Value, ClassLabel
import numpy as np


def train_classifier(preprocess_config: PreprocessConfig):
    df = load_data(preprocess_config)
    df = add_negative_cases(df)
    train, val, test = split_data(df)

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    def tokenize_function(examples):
        return tokenizer(examples["source_code"], padding="max_length", truncation=True)

    train_dataset = Dataset.from_pandas(
        train,
        features=Features({"source_code": Value("string"), "label": ClassLabel(2)}),
    )
    val_dataset = Dataset.from_pandas(
        val, features=Features({"source_code": Value("string"), "label": ClassLabel(2)})
    )

    tokenized_dataset_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_dataset_val = val_dataset.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_dataset_train.shuffle(seed=42).select(range(50))
    small_eval_dataset = tokenized_dataset_val.shuffle(seed=42).select(range(50))

    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", num_labels=2
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch", report_to="none"
    )
    # TrainingArguments(
    #         "test_trainer",
    #         evaluation_strategy="steps",
    #         eval_steps = 40,
    #         logging_steps = 40,
    #         per_device_train_batch_size= 24,
    #         per_device_eval_batch_size= 24,
    #         gradient_accumulation_steps=16,
    #         learning_rate=3e-5,
    #         prediction_loss_only=True,
    #     )
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
