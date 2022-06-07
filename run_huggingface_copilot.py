from transformers import AutoModelForCausalLM, AutoTokenizer, FlaxAutoModelForCausalLM
import torch as t

device = "cuda:0" if t.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained("flax-community/gpt-neo-125M-code-clippy")

tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-neo-125M-code-clippy")

prompt = """def greet(name):
  '''A function to greet user. Given a user name it should say hello'''
"""

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

start = input_ids.size(1)

out = model.generate(
    input_ids,
    do_sample=True,
    max_length=50,
    num_beams=2,
    early_stopping=True,
    eos_token_id=tokenizer.eos_token_id,
)

print(tokenizer.decode(out[0][start:]))
