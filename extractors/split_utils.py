import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple


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


from sklearn.model_selection import train_test_split


def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test_ratio = 0.1
    val_ratio = 0.1
    train, val_test = train_test_split(df, test_size=test_ratio + val_ratio)
    val, test = train_test_split(val_test, test_size=test_ratio)
    
    return train, val, test
