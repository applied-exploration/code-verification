import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def add_negative_cases(df: pd.DataFrame) -> pd.DataFrame:
    split_ratio = 0.5
    print(f"| Splitting dataset by {split_ratio}")
    positive, negative = train_test_split(
        df, test_size=split_ratio, random_state=1, shuffle=True
    )

    print(f"| Copying after to before on negative rows...")
    negative["before"] = negative["after"]

    print(f"| Adding labels to rows...")
    negative["label"] = -1.0
    positive["label"] = 1.0

    print(f"| Combining and shuffling dataframe...")
    df = pd.concat([positive, negative])

    df = df.iloc[np.random.permutation(len(df))]
    return df
