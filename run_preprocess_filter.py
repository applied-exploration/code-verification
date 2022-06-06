import json
from tqdm import tqdm
from typing import List
from utils.text_process import preprocess_source_code
from utils.json import load_json

src_dir = "data/original"


def filter_if(d: dict) -> bool:
    return (
        ".py" in d["path"]
        and "DataLoader(" in d["content"]
        and "import torch" in d["content"]
        and "zero_grad()" in d["content"]
        and int(d["size"]) < 1500000
    )


def run_filter_data():

    filtered_files = []
    for i in tqdm(range(1, 54)):
        files = load_json(f"{src_dir}/file-0000000000{i:02d}.json")
        filtered_files = filtered_files + [f for f in files if filter_if(f)]

    lines = []
    for file in filtered_files:
        lines.extend(
            [
                {"repo": file["repo_name"], "content": line, "path": file["path"]}
                for line in preprocess_source_code(file["content"])
            ]
        )

    json.dump(lines, open("data/derived/python-pytorch.json", "w"))


if __name__ == "__main__":
    run_filter_data()
