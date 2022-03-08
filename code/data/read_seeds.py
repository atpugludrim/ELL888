import numpy as np
# *- https://archive.ics.uci.edu/ml/datasets/seeds -*
# area perimeter compactness lkernel wkernel assymcoeff lkgrove

def read_seeds(seeds_path):
    data = []
    with open(seeds_path,'r') as f:
        for line in f:
            tokens = [float(k.strip()) for k in line.strip().split()]
            data.append(tokens)
    data = np.array(data)
    return data[:,:-1]
