import numpy as np

from .pyLocalizedLasso.pyLocalizedLasso import OLD_LocalizedLasso
from sklearn.base import BaseEstimator
from src.utils.tools import find_nn


class LocalizedLasso(BaseEstimator):
    """ Localized Lasso with iterative-least squares optimization (Squared loss)


    Model:

    y_i = w_i^t x_i + b_i + e_i (i = 1, ..., n)
    #w_i is the d dimensional vector and b_i is the bias for sample i.

    Parameters
    ----------
    num_iter: int
       The number of iteration for iterative-least squares update: (default: 10)
    lam_net: double
       The regularization parameter for the network regularization term (default:1)
    lam_exc: double
       The regularization parameter for the exclusive regularization (l12) term (default:0.01)
    biasflag : int
       1: Add bias term b_i
       0: No bias term
    W: (d + 1) x n dimensional matrix
       i-th column vector corresponds to the model for w_i
    vecW: double
       dn + 1 dimensional parameter vector vec(W)
    Yest: double
       Estimated training output vector.
    """

    def __init__(self, num_iter=10,
                 lam_net=1, lam_exc=0,
                 biasflag=False, K=5,
                 verbose=False):

        self.verbose = False
        self.num_iter = num_iter
        self.lam_net = lam_net
        self.lam_exc = lam_exc
        self.biasflag = biasflag

        # Neighboors to construct R
        self.K = K
        
    def fit(self, X, y=None):
        self.model_ = OLD_LocalizedLasso(num_iter=self.num_iter,
                                         lam_net=self.lam_net,
                                         lam_exc=self.lam_exc,
                                         biasflag=self.biasflag,
                                         verbose=self.verbose)
        
        self.R_ = R = self._construct_R(X)
        self.model_.fit_regression(X.T, y, R)

        self.X_ = X
        self.R_ = R
            
        return self

    def predict(self, X):
        R = self._construct_R(self.X_, X)

        y = []
        for i in range(len(X)):
            yt, _ = self.model_.prediction(X[i], R[i])
            y.append(yt)
        return np.array(y)

    def _construct_R(self, X, _Xtest=None, eps=0.0001, makesym=False):
        if _Xtest is None:
            X2 = X
            # Taking the K neighrest neighboors
            _, index = find_nn(X, X2, self.K + 1)
            index = index[1:]  # remove itself
        else:
            X2 = _Xtest
            # Taking the K neighrest neighboors
            _, index = find_nn(X, X2, self.K)
            
        X_neigh = X[index]

        R = np.zeros((len(X2), len(X)))
        weights = 1 / (eps + np.linalg.norm(X2 - X_neigh, axis=2))

        np.put_along_axis(R, index.T, weights.T, axis=1)    
        
        # Make a summetric R if needed
        if makesym:
            R = R + R.T
        
        return R
