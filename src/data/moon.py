import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons
from .dataset import Datasets


class Moon:
    """Moon dataset

    """
    settings = {"classification": True}
    
    params = {"k": 2,
              "K_anchors": 5,
              "alpha": 10,
              "l1_ratio": 0,
              "bias": True,
              "lam": 1000,
              "K_prediction": 5}
    
    def __init__(self, n=1000, noise=0.1, prop=0.9, seed=10):
        data, y = make_moons(n, noise=noise, random_state=seed)

        self.n = n
        self.y = y
        self.data = data
        self.datasets = Datasets(self.data, self.y)

    def create_figures(self, modelclass, figpath="figures", show=True):
        # To select the "good" anchors for visualization
        dataset = self.datasets(seed=5)
        model = modelclass(k=2, K_anchors=1, alpha=10,
                           bias=True, lam=1000, K_prediction=5)

        # Remove convergence warnings
        import warnings
        from sklearn.exceptions import ConvergenceWarning
        warnings.simplefilter("ignore", ConvergenceWarning)
        
        model.fit(dataset["train"]["X"], dataset["train"]["Y"])
        
        nxy = 100 
        grid = np.meshgrid(np.linspace(-1.5, 2.5, nxy),
                           np.linspace(-1, 1.5, nxy))
        datatest = np.stack(grid).transpose(1, 2, 0).reshape((-1, 2))
        gridpred = model.predict(datatest).reshape((nxy, nxy))

        colors = np.zeros_like(gridpred)
        colors[gridpred > 0.5] = 1
        colors[gridpred < 0.5] = 0
        colors[gridpred == 0.5] = 0.5

        figsize = (4.5, 4.5)
        plt.figure(figsize=figsize)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        Xtr, _ = dataset["train"]["X"], dataset["train"]["Y"].flatten()
        # Xte, _ = dataset["test"]["X"], dataset["test"]["Y"]

        for i, color in enumerate(["b", "r"]):
            mask = model.assignements_ == i
            plt.scatter(Xtr[:, 0][mask], Xtr[:, 1][mask], s=20, c=color, label="Class {}".format(i+1))
        legend = plt.legend(loc='upper right')
        plt.gca().add_artist(legend)
        
        plt.contour(grid[0], grid[1], colors, cmap=plt.get_cmap("rainbow"))
        
        plt.legend()
        plt.savefig("figures/moon.png")
        if show:
            plt.show()
        plt.close()

    
if __name__ == "__main__":
    moon = Moon()
    dataset = moon.datasets()
