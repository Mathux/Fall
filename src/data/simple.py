import numpy as np
import os
import matplotlib.pyplot as plt
from .dataset import Datasets

from src.utils.tools import create_dir


class Simple:
    """Simple dataset
    The model is a step function. The observed data have some noise added.

    """

    def __init__(self, value=5., seed=42):
        n = 101
        X = np.linspace(0, 11, n).reshape((n, 1))
        np.random.seed(seed)
        y = (value * ((X > 3) & (X < 8))).flatten() + \
            np.random.uniform(-0.5, 0.5, size=(n))
        self.n = len(y)
        self.y = y
        self.value = value
        self.data = X
        self.datasets = Datasets(self.data, self.y)

    def create_figures(self, modelclass, figpath="figures", show=True):
        figpath = os.path.join(figpath, "simple/")
        create_dir(figpath)

        # To select the "good" anchors for visualization
        dataset = self.datasets(seed=12)

        # Take a model suitable for visualization
        model = modelclass(alpha=1000, k=2, K_prediction=2, bias=True)

        colorlist = ["b", "r", "g", "c", "m", "y", "k", "w"]
        figsize = (4.5, 4.5)

        X = np.linspace(0, 11, 3000).reshape((-1, 1))
        Xtr, ytr = dataset["train"]["X"], dataset["train"]["Y"].flatten()

        # For each lambda, fit the entire model
        predictions = []
        exp_lambdas = []
        for exp in np.linspace(-3, 3, 13):  # range(-3, 4):
            exp_lambdas.append(exp)
            model.lam = pow(10, exp)
            model.fit(dataset["train"]["X"], dataset["train"]["Y"])
            predictions.append(model.predict(X))

        # a) plot model + input
        plt.figure(figsize=figsize)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.plot(X, (self.value * ((X > 3) & (X < 8))).flatten(),
                 c="r", label="True model")
        plt.scatter(Xtr[:, 0], ytr, s=20, c="black", label="Train set")
        plt.legend(loc="upper left")
        plt.savefig(figpath + "1.png")
        if show:
            plt.show()
        plt.close()

        # b) anchors models + input and neighboors
        plt.figure(figsize=figsize)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        amodel = model.anchormodel_
        apredicts = amodel.predict(X)  # slopes

        X_anchors = amodel.X_[amodel.aindex_]
        y_anchors = amodel.y_[amodel.aindex_]

        X_sets = amodel.X_[amodel.aind_sets_].transpose(1, 0, 2)
        y_sets = amodel.y_[amodel.aind_sets_].transpose(1, 0, 2)

        # show first the noisy points
        plt.scatter(Xtr[:, 0], ytr, s=20, c="black", label="Train set")
        legend = plt.legend(loc='upper left')
        plt.gca().add_artist(legend)

        locs = ["center left", "center right"]
        iterator = zip(X_sets, y_sets, X_anchors, y_anchors,
                       apredicts, colorlist, locs)
        for X_S, Y_S, Xa, ya, anchors_predict, color, loc in iterator:
            # anchor model
            l1, = plt.plot(X, anchors_predict, c=color, label="Model")
            # anchor neighboors
            l2 = plt.scatter(X_S[1:], Y_S[1:], marker="*",
                             s=100, c=color, label="Neighboors")
            l3 = plt.scatter(Xa[:, np.newaxis], ya[:, np.newaxis], label="Anchor",
                             marker="X", s=100, c=color)
            legend = plt.legend(handles=[l3, l2, l1], loc=loc)
            plt.gca().add_artist(legend)
        plt.savefig(figpath + "2.png")
        if show:
            plt.show()
        plt.close()

        # c) clustering effect
        plt.figure(figsize=figsize)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        locs = ["center left", "center right"]
        iterator2 = enumerate(zip(apredicts, colorlist, locs))
        for i, (_, color, loc) in iterator2:
            mask = model.assignements_ == i
            ls = plt.scatter(Xtr[:, 0][mask], ytr[mask],
                             s=20, c=color, label="Cluster")
            legend = plt.legend(handles=[ls], loc=loc)
            plt.gca().add_artist(legend)

        plt.savefig(figpath + "3.png")
        if show:
            plt.show()
        plt.close()

        # d) model with various lambda
        plt.figure(figsize=figsize)
        ax = plt.axes([0, 0, 1, 1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        nlambda = len(predictions)
        cm = plt.get_cmap("rainbow")
        rcolors = [cm(1*x/nlambda) for x in range(nlambda)]
        # zip(reversed(predictions), reversed(rcolors)):
        for pred, exp, color in zip(predictions, exp_lambdas, rcolors):
            plt.plot(X, pred, c=color,
                     label="$\\lambda=10^{{{}}}$".format(exp))
        legend = plt.legend(loc='upper right')
        plt.gca().add_artist(legend)

        lt = plt.scatter(Xtr[:, 0], ytr, s=20, c="black", label="Train set")
        legend = plt.legend(handles=[lt], loc='upper left')
        plt.gca().add_artist(legend)

        plt.savefig(figpath + "4.png")
        if show:
            plt.show()
        plt.close()


if __name__ == "__main__":
    simple = Simple()
