import hashlib

import pandas as pd


def sig_from_df(df: pd.DataFrame) -> str:
    # fast-enough, stable signature; change this if you prefer something else
    s = pd.util.hash_pandas_object(df, index=True).to_numpy()
    return hashlib.sha256(s).hexdigest()
