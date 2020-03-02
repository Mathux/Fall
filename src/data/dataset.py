import numpy as np
from src.utils.tools import add_bias, get_random_split


class Datasets:
    def __init__(self, data, y):
        self.data = data
        self.y = y if len(y.shape) > 1 else y[:, np.newaxis]
        self.n, self.d = data.shape

    def __call__(self, prop=0.9, seed=10, bias=False):
        X = add_bias(self.data) if bias else self.data
        Xtr, Xte, ytr, yte = get_random_split(X, self.y, prop, seed)
        return {"train": {"X": Xtr,
                          "Y": ytr},
                "test": {"X": Xte,
                         "Y": yte}}
