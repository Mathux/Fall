import numpy as np
import matplotlib.pyplot as plt
from .dataset import Datasets


class Sin:
    """Simple dataset

    """
    isclassification = False
    anchor_params = {"k": 50,
                     "K": 10,
                     "alpha": 10,
                     "estimator": "Ridge",
                     "bias": False}
    params = {"K": 5,
              "lam": 0.1}

    def __init__(self, value=10.):
        n = 1000
        X = np.linspace(0, 30, n).reshape((n, 1))
        eps = np.random.normal(loc=0.0, scale=0.1, size=(n, 1))
        y = np.sin(X) + eps

        self.n = n
        self.y = y
        self.data = X
        self.datasets = Datasets(self.data, self.y)

    def show_model(self, model, dataset):
        X = np.linspace(0, 30, 1000).reshape((-1, 1))
        y = model.predict(X)
        Xtr, ytr = dataset["train"]["X"], dataset["train"]["Y"].flatten()
        plt.plot(X, y)
        plt.scatter(Xtr[:, 0], ytr, c="black")
        plt.show()


