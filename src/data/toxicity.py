import numpy as np
import pandas as pd

from .dataset import Datasets

FISH = "datasets/toxicity/qsar_fish_toxicity.csv"
AQUATIC = "datasets/toxicity/qsar_aquatic_toxicity.csv"

FISH_HEAD = ["CIC0", "SM1_Dz(Z)", "GATS1i", "NdsCH", "NdssC", "MLOGP", "label"]
AQUATIC_HEAD = ["TPSA(Tot)", "SAacc", "H-050",
                "MLOGP", "RDCHI", "GATS1p",
                "nN", "C-040", "label"]


def FishToxicity(*args, **kargs):
    return Toxicity(FISH, FISH_HEAD, *args, **kargs)


def AquaticToxicity(*args, **kargs):
    return Toxicity(AQUATIC, AQUATIC_HEAD, *args, **kargs)


FishToxicity.isclassification = False
AquaticToxicity.isclassification = False

FishToxicity.anchor_params = {"k": 10,
                              "K": 10,
                              "alpha": 10,
                              "estimator": "Ridge",
                              "bias": False}
FishToxicity.params = {"K": 5,
                       "lam": 1}
    
AquaticToxicity.anchor_params = {"k": 60,
                                 "K": 16,
                                 "alpha": 10,
                                 "estimator": "Ridge",
                                 "bias": True}
AquaticToxicity.params = {"K": 5,
                          "lam": 1,
                          "reg": 10}


class Toxicity:
    """Toxicity dataset

    """
    
    def __init__(self, name, headers):
        csv = pd.read_csv(name, sep=";", names=headers, dtype=float)
        labelname = "label"
        
        self.n = csv.shape[0]
        self.y = np.array(csv[labelname])
        self.data = np.array(csv.drop(labelname, axis=1))
        self.datasets = Datasets(self.data, self.y)
        

if __name__ == "__main__":
    datasets = FishToxicity()
