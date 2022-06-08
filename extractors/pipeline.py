from typing import Dict, List
from transformers.pipelines.feature_extraction import FeatureExtractionPipeline
import torch as t


class MyFeatureExtractionPipeline(FeatureExtractionPipeline):
    def preprocess(self, inputs, truncation=None) -> Dict[str, t.Tensor]:
        print(self.tokenizer)
        return_tensors = self.framework

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
