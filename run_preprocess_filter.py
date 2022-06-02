import json
from tqdm import tqdm
from typing import List

src_dir = "data/original"


def filter_if(d: dict) -> bool:
    return (
        ".py" in d["path"]
        and "DataLoader(" in d["content"]
        and "import torch" in d["content"]
        and "zero_grad()" in d["content"]
        and int(d["size"]) < 1500000
    )


def load_json(path: str) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def run_filter_data():
    filtered_data = []
    for i in tqdm(range(1, 54)):
        data = load_json(f"{src_dir}/file-0000000000{i:02d}.json")
        filtered_data = filtered_data + [d for d in data if filter_if(d)]
        print(len(filtered_data))

    json.dump(filtered_data, open("data/derived/python-pytorch.json", "w"))


if __name__ == "__main__":
    run_filter_data()
