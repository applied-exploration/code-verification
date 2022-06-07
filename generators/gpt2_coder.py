# from autocomplete.gpt2_coder import GPT2Coder

# m = GPT2Coder("shibing624/code-autocomplete-distilgpt2-python")
# print(m.generate("import torch.nn as")[0])

import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Callable, Tuple


def gpt2_coder() -> Callable:
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    tokenizer = GPT2Tokenizer.from_pretrained(
        "shibing624/code-autocomplete-distilgpt2-python"
    )
    model = GPT2LMHeadModel.from_pretrained(
        "shibing624/code-autocomplete-distilgpt2-python"
    )

    def inference(prompt: str) -> Tuple[str, str]:
        input_ids = tokenizer.encode(
            prompt, add_special_tokens=False, return_tensors="pt"
        )
        outputs = model.generate(
            input_ids=input_ids,
            max_length=64 + len(prompt),
            temperature=1.0,
            top_k=50,
            top_p=0.95,
            repetition_penalty=1.0,
            do_sample=True,
            num_return_sequences=1,
            length_penalty=2.0,
            early_stopping=True,
        )
        generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return prompt, generated_code

    return inference
