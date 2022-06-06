from typing import List
import json

def load_json(path: str) -> List[dict]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f]
