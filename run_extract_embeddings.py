import pandas as pd
from config import PreprocessConfig, preprocess_config
from data.dataset import RawDataset
from transformers.pipelines import pipeline
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
import torch as t
from typing import Dict, List
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


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

    print("| Converting original source to embeddings...")
    dataset = RawDataset(df)
    embeddings: List[t.Tensor] = []

    # pipeline padding adds a unique padding token, that also gets passed into inference, hence we get a feature tensor for the padding tokens aswell.
    for out in tqdm(
        pipe(dataset, batch_size=config.batch_size, padding=True),
        total=len(df),
    ):
        # Shape is: num_lines | num_words in line | feature_length of word (token)
        out_tensor = t.tensor(out)[0]
        avaraged_embeddings = t.mean(out_tensor, dim=0, keepdim=False)
        embeddings.append(avaraged_embeddings)

    embeddings = pad_sequence(embeddings, batch_first=True)

    print("| Saving embeddings to disk...")
    original_order = dataset.get_sort_indices()
    embeddings = embeddings[original_order]
    t.save(embeddings, "data/derived/embeddings_flat.pt")

    print("✅ Feature preprocessing done...")


if __name__ == "__main__":
    run_extract_embeddings(preprocess_config)
