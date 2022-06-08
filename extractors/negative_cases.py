import pandas as pd
from sklearn.model_selection import train_test_split


def add_negative_cases(df: pd.DataFrame) -> pd.DataFrame:

    positive, negative = train_test_split(
        df, test_size=0.50, random_state=1, shuffle=True
    )

    negative["before"] = negative["after"]
    negative["label"] = -1.0

    positive["label"] = 1.0

    df = pd.concat([positive, negative])

    return df
