import numpy as np
import pandas as pd

from .dataset import Datasets

NAME = "datasets/superconduct/train.csv"


class SuperConduct:
    """SuperConduct dataset

    """
    def __init__(self):
        csv = pd.read_csv(NAME, dtype=float)
        labelname = "critical_temp"

        self.n = csv.shape[0]
        self.y = np.array(csv[labelname])
        self.data = np.array(csv.drop(labelname, axis=1))
        self.datasets = Datasets(self.data, self.y)


if __name__ == "__main__":
    conduct = SuperConduct()
