import pyparsing
import re
from typing import List

__commentFilter = pyparsing.pythonStyleComment.suppress()


def preprocess_source_code(original: str) -> List[str]:
    strip_tabs = re.sub(
        r" +[\t]*", " ", original
    )  # Strips tabs and multiple white spaces
    split_by_line = strip_tabs.split("\n")

    stripped_lines = [__strip_comments(line) for line in split_by_line]
    no_empty_lines = [
        line for line in stripped_lines if __is_empty(line) == False
    ]

    return no_empty_lines


def __strip_comments(source: str) -> str:
    return __commentFilter.transformString(source)

def __is_empty(line: str) -> bool:
    return line == "" or line.startswith("'''") or line.startswith("\"\"\"")