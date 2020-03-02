import numpy as np
import pandas as pd

from .dataset import Datasets

NAME = "datasets/parkinsons_updrs/parkinsons_updrs.data"


class Parkinson_UPDRS_motor:
    """Parkinson_UPDRS dataset
    https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
    """    
    def __init__(self):
        csv = pd.read_csv(NAME)
        labelname = "motor_UPDRS"
        remove = "total_UPDRS" # this is the other output
        self.n = csv.shape[0]
        self.y = np.array(csv[labelname], dtype=float)
        self.data = np.array(csv.drop([labelname, remove, "subject#"], axis=1), dtype=float)
        self.datasets = Datasets(self.data, self.y)


class Parkinson_UPDRS_total:
    """Parkinson_UPDRS dataset
    https://archive.ics.uci.edu/ml/datasets/Parkinsons+Telemonitoring
    """
    def __init__(self):
        csv = pd.read_csv(NAME)
        labelname = "total_UPDRS"
        remove = "motor_UPDRS" # this is the other output
        self.n = csv.shape[0]
        self.y = np.array(csv[labelname], dtype=float)
        self.data = np.array(csv.drop([labelname, remove, "subject#"], axis=1), dtype=float)
        self.datasets = Datasets(self.data, self.y)


if __name__ == "__main__":
    parkinson1 = Parkinson_UPDRS_motor()
    parkinson2 = Parkinson_UPDRS_total()
