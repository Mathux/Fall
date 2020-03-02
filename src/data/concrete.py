import numpy as np
import pandas as pd

from .dataset import Datasets

NAME = "datasets/concrete/Concrete_Data.xls"


class Concrete:
    """Concrete compressive strength dataset
    https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

    """
    
    def __init__(self):
        csv = pd.read_excel(NAME, dtype=float)
        labelname = "Concrete compressive strength(MPa, megapascals) "

        self.n = csv.shape[0]
        self.y = np.array(csv[labelname])
        self.data = np.array(csv.drop(labelname, axis=1))
        self.datasets = Datasets(self.data, self.y)


if __name__ == "__main__":
    concrete = Concrete()
