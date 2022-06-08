import pandas as pd

df = pd.read_json("data/original/tssb-0.jsonl", lines=True)

df.head()

df.to_csv("data/derived/tssb-text.csv", columns=["diff"], index=False)
