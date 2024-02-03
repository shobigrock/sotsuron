import numpy as np
import pandas as pd

"""
各種人工データを作るためのファイル
"""
def make_data_3dims():
    df = pd.DataFrame()
    length = 10000
    n_arms = 3
    n_dims = 3

    df["Age"] = np.random.randint(0, 10, size=length)
    df["edu_background"] = np.random.randint(0, 5, size=length)
    df["strength"] = np.random.randint(0, 10, size=length)

    return df, n_arms, n_dims
