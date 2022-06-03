import pandas as pd
from config import PreprocessConfig, preprocess_config
from data.dataset import get_embeddings_batched
from transformers.pipelines import pipeline
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
import torch as t
from typing import Dict
from tqdm import tqdm


def shorten_dataset(config: PreprocessConfig):
    df = pd.read_json("data/derived/python-pytorch.json")[2 : config.dataset_size + 2]
    df.to_json("data/derived/python-pytorch-short.json")


class MyPipeline(FeatureExtractionPipeline):
    def preprocess(self, inputs, truncation=None) -> Dict[str, t.Tensor]:
        return_tensors = self.framework

        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model_inputs = self.tokenizer(inputs, return_tensors=return_tensors)

        if hasattr(self, "input_tokenized_length") == False:
            self.input_tokenized_length = []

        self.input_tokenized_length.append(model_inputs["input_ids"].shape[1])

        return model_inputs

    def postprocess(self, model_outputs):
        # [0] is the first available tensor, logits or last_hidden_state.
        outputs = model_outputs[0].tolist()

        for i, output in enumerate(outputs):
            outputs[i] = output[: self.input_tokenized_length[i]]

        return outputs


def save_embeddings(embeddings_dict: dict):
    print("| Saving embeddings to disk...")
    t.save(embeddings_dict, "data/derived/embeddings_per_file.pt")
    id_map = []
    embeddings = t.tensor([])
    for file_key, lines in embeddings_dict:
        for line_key, line_value in lines:
            id_map.append((file_key, line_key))
            embeddings = t.cat((embeddings, line_value[""]), dim=0)
    t.save(embeddings, "data/derived/embeddings_flat.pt")
    t.save(id_map, "data/derived/id_to_info_map.pt")


def run_extract_embeddings(config: PreprocessConfig):
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
        pipeline_class=MyPipeline,
    )
    source_code_all = df["content"].to_numpy()

    print("| Converting original source to embeddings...")
    embeddings = dict()
    for i in tqdm(range(len(df))):
        embeddings[i] = get_embeddings_batched(source_code_all[i], pipe, config)

        if i % config.save_every == 0:
            print(f"| Saving embeddings to file...")
            save_embeddings(embeddings)

    print("✅ Feature preprocessing done...")


if __name__ == "__main__":
    run_extract_embeddings(preprocess_config)
