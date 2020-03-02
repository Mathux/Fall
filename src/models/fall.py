import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import ElasticNet

from src.utils.tools import find_nn
from src.utils.tools import select_random, select_kmeans, select_kmeans_with_gt

EPS = 0.00001


class Anchors_model:
    """Anchors models

    Method
    ------
    Compute local anchor models with ElasticNet

    1) Choose k anchors vectors with the selection function
    2) Find the corresponding K neighrest neighboors of these vectors
    3) For each set of points, compute anchor model A_l (with ElasticNet)

    Parameters
    ----------
    k: int
       The number of anchors vectors
    K: int
       The number of neighrest neighboors
    alpha: double (1.0)
       The regularization parameter in the ElasticNet anchor models
    l1_ratio: double (0.0)
       Ratio of the l1 norm in ElasticNet (0 for Ridge and 1 for Lasso)
    selection: "random" or "kmeans" or "kmeans_with_gt" or a function to select points
       The process of choosing the anchor points
    bias: bool (True)
       Use bias or not    
    """

    def __init__(self, k, K, alpha=1.0, l1_ratio=0.0,
                 selection="random", bias=True, seed=42):
        
        def get_coef(clf, bias=bias):
            coef = clf.coef_
            intercept = clf.intercept_
            if len(coef.shape) > 1:
                coef = coef.T
            else:
                coef = coef[:, np.newaxis]
            if self.bias:
                coef = np.vstack((coef, intercept[np.newaxis]))

            return coef

        self.get_coef = get_coef
        self.l1_ratio = l1_ratio

        self.k = k
        self.K = K
        self.alpha = alpha
        self.bias = bias
        if selection == "random":
            self.select = lambda X, y, k: select_random(X, k, seed=seed)
        elif selection == "kmeans":
            self.select = lambda X, y, k: select_kmeans(X, k, seed=seed)
        elif selection == "kmeansGT":
            self.select = lambda X, y, k: select_kmeans_with_gt(X, y, k, seed=seed)
        else:
            self.select = selection

    def fit(self, X, y):
        n, d = X.shape
        n, m = y.shape

        # Select anchors points
        X_anchors, index = self.select(X, y, self.k)

        # Find the sets S of neighrest neighboors
        X_sets, ind_sets, y_sets = find_nn(X, X_anchors, self.K, y)

        # Solve for each problems
        A = []
        for X_S, Y_S in zip(X_sets, y_sets):
            clf = ElasticNet(alpha=self.alpha,
                             l1_ratio=self.l1_ratio,
                             max_iter=100000, tol=1,
                             fit_intercept=self.bias)
            clf.fit(X_S, Y_S)
            coef = self.get_coef(clf, bias=self.bias)
            A.append(coef)

        A = np.array(A)
        
        # Update the parameter used
        self.X_ = X
        self.y_ = y
        self.aindex_ = index
        self.aind_sets_ = ind_sets
        
        # Update the result
        self.A_ = A
        return self

    def predict(self, X):
        X = np.hstack((X, np.ones((len(X), 1)))) if self.bias else X
        return np.einsum("kdm, nd->knm", self.A_, X)


class Fall(BaseEstimator):
    """Fast local linear regression with anchors regularization

    Method
    ------
    Compute local linear models with anchor points and use them to create a 
    regularized local linear model.
    
    1) Compute the anchor models
    2) Compute for each point its best anchor model
    3) Compute for each point the model W_i

    Parameters
    ----------
    k: int (10)
       Number of anchors points
    K_anchors: int (5)
       The number of neighrest neighboors for the anchors model
    alpha: double (1.0)
       The regularization parameter in the ElasticNet anchor models
    l1_ratio: double (0.0)
       Ratio of the l1 norm in ElasticNet (0 for Ridge and 1 for Lasso)
    K_prediction: int (3)
       The number of neighrest neighboors for the prediction
    lam: double (1.0)
       The regularization parameter for FALL
    bias: bool (True)
       Use bias or not
    """

    def __init__(self, k=10, K_anchors=5, alpha=1.0,
                 l1_ratio=0.0, bias=True, lam=1.0, K_prediction=3):
        # Anchors part
        self.k = k
        self.K_anchors = K_anchors
        self.alpha = alpha
        self.l1_ratio = l1_ratio        
        self.bias = bias
        
        # Regularization part
        self.lam = lam

        # Prediction part
        self.K_prediction = K_prediction
    
    def fit(self, X, y):
        # Get the anchors models
        self.anchormodel_ = Anchors_model(k=self.k, K=self.K_anchors,
                                          alpha=self.alpha,
                                          l1_ratio=self.l1_ratio,
                                          bias=self.bias).fit(X, y)
        self.A_ = A = self.anchormodel_.A_
        self.k_, self.d_, self.m_ = A.shape
        self.n_, _ = X.shape

        lam = self.lam
        
        # Add bias if needed
        self.X_ = X = np.hstack((X, np.ones((len(y), 1)))) if self.bias else X
        self.y_ = y

        # Compute all the norms 
        xnorms = np.einsum("ij,ij->i", X, X)[:, np.newaxis, np.newaxis]
        
        # Compute the index (l_i) of the best assignements
        # that is to say p_i = delta_{l_i} [the best model with smaller error]
        assignements = np.argmin(np.linalg.norm(y[:, np.newaxis, :] - np.einsum("kdm,nd->nkm", A, X), axis=2), axis=1)
        
        # Take the best anchors for every i (A_{l_i})
        anchors_selected = np.take_along_axis(A, assignements[:, np.newaxis, np.newaxis], axis=0)

        # Compute each W_i by combining the best anchor model A_{l_i} with the correction term
        self.W_ = np.einsum("nd,nm->ndm", X, y - np.einsum("ndm,nd->nm", anchors_selected, X))/(xnorms + lam) + anchors_selected

        # Save the p_i in case we want to use them later
        self.P_ = np.zeros((self.n_, self.k_))
        np.put_along_axis(self.P_, assignements[:, np.newaxis], 1, axis=1)
        self.assignements_ = assignements
        
        # Estimated y
        self.y_est = np.einsum("ijk,ij->ik", self.W_, self.X_)
        return self
    
    def predict(self, X, K=None):
        # Use the number of neighboors provided
        K = self.K_prediction if K is None else K
        
        # Add bias if needed
        X = np.hstack((X, np.ones((len(X), 1)))) if self.bias else X

        # Finding the neighboors
        _, ind_sets = find_nn(self.X_, X, K)
        W_neigh = self.W_[ind_sets]
        X_neigh = self.X_[ind_sets]

        # Compute distances (eps added to avoid 1/0)
        distances = EPS + np.linalg.norm(X - X_neigh, axis=2)
        
        # Weighted sum (it will be convex coefficients)
        weights = np.einsum("ij,j->ij",
                            1 / distances,
                            1 / (1 / distances).sum(axis=0))

        # Compute the aggregated model
        W_pred = np.einsum("ijkl,ij->jkl", W_neigh, weights)

        # Compute the predicted y
        y_pred = np.einsum("ij,ijk->ik", X, W_pred)
        return y_pred
