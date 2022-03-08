import pandas as pd
import numpy as np


# *- https://archive.ics.uci.edu/ml/datasets/seeds -*
# area perimeter compactness lkernel wkernel assymcoeff lkgrove

def read_wine(wine_path):
    df = pd.read_csv(wine_path)
    df = df.drop(columns=['Magnesium','Proline'])
    return df.values
