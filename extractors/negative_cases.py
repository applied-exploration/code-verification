import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def add_negative_cases(df: pd.DataFrame) -> pd.DataFrame:
    positive = df.copy()[["before"]]
    negative = df.copy()[["after"]]

    print(f"| Adding labels to rows...")
    negative["label"] = 0.0
    negative = negative.rename(columns={"after": "source_code"})
    positive["label"] = 1.0
    positive = positive.rename(columns={"before": "source_code"})

    print(f"| Combining and shuffling dataframe...")
    df = pd.concat([positive, negative])

    df = df.iloc[np.random.permutation(len(df))]
    return df
