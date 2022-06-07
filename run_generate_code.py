from generators.gpt_neo_flax import gpt_neo_flax
from generators.gpt2_coder import gpt2_coder
from utils.logging import RichConsole


rc = RichConsole()

prompts = [
    """from torch import nn
    class LSTM(Module):
        def __init__(self, *,
                    n_tokens: int,
                    embedding_size: int,
                    hidden_size: int,
                    n_layers: int):""",
    """import numpy as np
    import torch
    import torch.nn as""",
    "import java.util.ArrayList",
    "def factorial(n):",
]

pipelines = {"gpt2_coder": gpt2_coder(), "gpt_neo_flax": gpt_neo_flax()}

for i, prompt in enumerate(prompts):
    print("-" * 20)
    print(f"{prompt}")

    generated_code = []
    for key, pipeline in pipelines.items():
        prompt, generated_code = pipeline(prompts)
        generated_code.append(generated_code)

    table_name = prompt
    columns = [key for key, item in pipelines.items()]
    
    rc.new_table(table_name, f'Prompt {i}')
    rc.define_columns(table_name, columns)
    rc.add_row_list_to_table(table_name, rows=generated_code)
    rc.display()
