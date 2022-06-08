from transformers import AutoModelForCausalLM, AutoTokenizer, FlaxAutoModelForCausalLM
import torch as t
from typing import List, Callable, Tuple


def gpt_neo_flax() -> Callable:
    device = "cuda:0" if t.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        "flax-community/gpt-neo-125M-code-clippy"
    )

    tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-neo-125M-code-clippy")

    def inference(prompt: str) -> Tuple[str, str]:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        start = input_ids.size(1)

        out = model.generate(
            input_ids,
            do_sample=True,
            max_length=150,
            num_beams=5,
            early_stopping=False,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_code = tokenizer.decode(out[0][start:])

        return prompt, generated_code

    return inference
