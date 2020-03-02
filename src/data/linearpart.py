import numpy as np
import matplotlib.pyplot as plt
from .dataset import Datasets


class LinearPart:
    """LinearPart dataset

    """
    def __init__(self, value=5., seed=42):
        n = 101
        X = np.linspace(0, 100, n).reshape((n, 1))
        y = np.zeros((n))
        y[(X < 25).flatten()] = -3 * X[X < 25] + 2
        y[(X >= 25).flatten() & (X < 50).flatten()] = 4*X[(X >= 25) & (X < 50)]-5
        y[(X >= 50).flatten() & (X < 75).flatten()] = X[(X >= 50) & (X < 75)] + 10
        y[(X >= 75).flatten()] = X[(X >= 75)] + 10
        np.random.seed(seed)
        y = y + 4.5 * np.random.uniform(-1, 1, size=(n))
        self.n = len(y)
        self.y = y
        self.value = value
        self.data = X
        self.datasets = Datasets(self.data, self.y)

    def show_model(self, model, dataset):
        """X = np.linspace(0, 100, 1000).reshape((-1, 1))
        y = model.predict(X)
        Xtr, ytr = dataset["train"]["X"], dataset["train"]["Y"].flatten()
        # plt.plot(X, (self.value * ((X > 4) & (X < 9))).flatten(), "r--")
        plt.plot(X, y, "b")
        plt.scatter(Xtr[:, 0], ytr, c="k")
        plt.show()

        for value in np.unique(model.assignements_):
            mask = model.assignements_ == value
            plt.scatter(model.X_[:, 0][mask], model.y_[mask])
            
        plt.show()
        """
        colorlist = ["b", "r", "g", "c", "m", "y", "k", "w"]
        figsize = (4.5, 4.5)

        X = np.linspace(0, 100, 3000).reshape((-1, 1))
        y = np.zeros((3000))
        y[(X < 25).flatten()] = -3 * X[X < 25] + 2
        y[(X >= 25).flatten() & (X < 50).flatten()] = 4*X[(X >= 25) & (X < 50)]-5
        y[(X >= 50).flatten() & (X < 75).flatten()] = X[(X >= 50) & (X < 75)] + 10
        y[(X >= 75).flatten()] = X[(X >= 75)] + 10

        Xtr, ytr = dataset["train"]["X"], dataset["train"]["Y"].flatten()
        
        # do the lambda
        predictions = []
        exp_lambdas = []
        for exp in np.linspace(-4, 5, 13): # range(-3, 4):
            exp_lambdas.append(exp)
            model.lam = pow(10, exp)
            model.fit(dataset["train"]["X"], dataset["train"]["Y"])
            predictions.append(model.predict(X))
        
        # a) plot model + input
        plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        plt.plot(X, y.flatten(), c="r", label="True model")
        plt.scatter(Xtr[:, 0], ytr, s=20, c="black", label="Train set")
        plt.legend(loc="upper left")
        plt.savefig("figures/linearpart/1.png")
        plt.show()
        plt.close()

        # b) anchors models + input and neighboors
        plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        amodel = model.anchormodel_
        apredicts = amodel.predict(X) # slopes
        
        X_anchors = amodel.X_[amodel.aindex_]
        y_anchors = amodel.y_[amodel.aindex_]

        X_sets = amodel.X_[amodel.aind_sets_].transpose(1, 0, 2)
        y_sets = amodel.y_[amodel.aind_sets_].transpose(1, 0, 2)

        # show first the noisy points
        plt.scatter(Xtr[:, 0], ytr, s=20, c="black", label="Train set")
        legend = plt.legend(loc='upper left')
        plt.gca().add_artist(legend)
        
        locs = ["center left", "center right", "upper right"]
        iterator = zip(X_sets, y_sets, X_anchors, y_anchors, apredicts, colorlist, locs)
        for X_S, Y_S, Xa, ya, anchors_predict, color, loc in iterator:
            # anchor model
            mask = (anchors_predict <= y.max()) & (y.min() <= anchors_predict)
            l1, = plt.plot(X[mask], anchors_predict[mask], c=color, label="Model")
            # anchor neighboors
            l2 = plt.scatter(X_S[1:], Y_S[1:], marker="*", s=100, c=color, label="Neighboors")
            l3 = plt.scatter(Xa[:, np.newaxis], ya[:, np.newaxis], label="Anchor",
                             marker="X", s=100, c=color)
            legend = plt.legend(handles=[l3, l2, l1], loc=loc)
            plt.gca().add_artist(legend)
        plt.savefig("figures/linearpart/2.png")
        plt.show()
        plt.close()
        
        # c) clustering effect
        plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        locs = ["center left", "center right", "upper right"]
        iterator2 = enumerate(zip(apredicts, colorlist, locs))
        for i, (_, color, loc) in iterator2:
            mask = model.assignements_ == i
            ls = plt.scatter(Xtr[:, 0][mask], ytr[mask], s=20, c=color, label="Cluster")
            legend = plt.legend(handles=[ls], loc=loc)
            plt.gca().add_artist(legend)
            
        plt.savefig("figures/linearpart/3.png")
        plt.show()
        plt.close()

        # d) model with various lambda
        plt.figure(figsize=figsize)
        ax = plt.axes([0,0,1,1], frameon=False)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        
        nlambda = len(predictions)
        cm = plt.get_cmap("rainbow")
        rcolors = [cm(1*x/nlambda) for x in range(nlambda)]
        for pred, exp, color in zip(predictions, exp_lambdas, rcolors): # zip(reversed(predictions), reversed(rcolors)):
            plt.plot(X, pred, c=color, label="$\\lambda=10^{{{}}}$".format(exp))
        legend = plt.legend(loc='upper right')
        plt.gca().add_artist(legend)
        
        lt = plt.scatter(Xtr[:, 0], ytr, s=20, c="black", label="Train set")
        legend = plt.legend(handles=[lt], loc='upper left')
        plt.gca().add_artist(legend)
        
        plt.savefig("figures/linearpart/4.png")
        plt.show()
        plt.close()
        import ipdb; ipdb.set_trace()
        pass
