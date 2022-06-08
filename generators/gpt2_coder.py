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

        start = input_ids.size(1)

        out = model.generate(
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
        generated_code = tokenizer.decode(out[0][start:], skip_special_tokens=True)

        return prompt, generated_code

    return inference
