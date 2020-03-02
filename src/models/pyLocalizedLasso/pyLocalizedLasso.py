import numpy as np
import scipy.sparse as sp
from numpy.matlib import repmat
from scipy.sparse.linalg import spsolve
from scipy.sparse import kron
from scipy.spatial.distance import cdist


class OLD_LocalizedLasso:
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

    def __init__(self, num_iter=10, lam_net=1,
                 lam_exc=0.01, biasflag=False,
                 verbose=False):
        self.num_iter = num_iter
        self.lam_net = lam_net
        self.lam_exc = lam_exc
        self.biasflag = biasflag
        self.verbose = verbose

    # Prediction with Weber optimization
    def prediction(self, Xte, Rte):
        [d, ntr] = self.W.shape

        wte = np.zeros((1, d))

        loss_weber = np.zeros((20, 1))

        if np.sum(Rte) == 0:
            wte = self.W.mean(1)[np.newaxis, :]
        else:
            for k in range(0, 20):
                dist2 = cdist(wte, self.W.transpose())
                invdist2 = Rte / (2 * dist2 + 10e-5)
                sum_dist2 = np.sum(invdist2)
                wte = np.dot(invdist2, self.W.transpose()) / sum_dist2
                loss_weber[k] = np.sum(Rte * dist2)

        if self.biasflag:
            yte = (wte[0][0:(d - 1)] * Xte).sum() + wte[0][d - 1]
            # yte = wte[:, :-1].dot(Xte).sum(axis=0) + wte[0][-1]
        else:
            yte = (wte[0] * Xte).sum()
            # yte = wte.dot(Xte).sum(axis=0)

        return yte, wte[0]

    # Regression
    def fit_regression(self, X, Y, R):
        [d, ntr0] = X.shape
        
        if self.biasflag:
            Xtr = np.concatenate((X, np.ones((1, ntr0))), axis=0)
        else:
            Xtr = X

        Ytr = Y

        [d, ntr] = Xtr.shape
        dntr = d * ntr

        vecW = np.ones((dntr, 1))
        index = np.arange(0, dntr)
        val = np.ones(dntr)
        D = sp.csc_matrix((val, (index, index)), shape=(dntr, dntr))

        # Generate input matrix
        # A = np.zeros((ntr, dntr))
        A = sp.lil_matrix((ntr, dntr))

        for ii in range(0, ntr):
            ind = range(ii, dntr, ntr)
            A[ii, ind] = Xtr[:, ii]

        A = sp.csr_matrix(A)

        one_ntr = np.ones(ntr)
        I_ntr = sp.diags(one_ntr, 0, format='csc')

        one_d = np.ones(d)
        I_d = sp.diags(one_d, 0, format='csc')

        fval = np.zeros(self.num_iter)
        for iter in range(0, self.num_iter):

            DinvA = spsolve(D, A.transpose())
            B = I_ntr + A.dot(DinvA)
            tmp = spsolve(B, Ytr.reshape((-1, 1)))
            vecW = DinvA.dot(tmp)

            W = np.reshape(vecW, (ntr, d), order='F')

            tmpNet = cdist(W, W)
            tmp = tmpNet * R

            # Network regularization
            U_net = tmp.sum()

            tmp = 0.5 / (tmpNet + 10e-10) * R

            td1 = sp.diags(tmp.sum(0), 0)
            td2 = sp.diags(tmp.sum(1), 0)

            AA = td1 + td2 - 2.0 * tmp
            AA = (AA + AA.transpose()) * 0.5 + 0.001 * sp.eye(ntr, ntr)

            D_net = kron(I_d, AA, format='csc')

            # Exclusive regularization
            if self.biasflag:
                tmp = abs(W[:, 0:(d - 1)]).sum(1)
                U_exc = (tmp * tmp).sum()

                tmp_repmat = repmat(np.c_[tmp], 1, d)
                tmp_repmat[:, d - 1] = 0
                tmp = tmp_repmat.flatten(order='F')

            else:
                tmp = abs(W).sum(1)
                U_exc = (tmp * tmp).sum()

                tmp_repmat = repmat(np.c_[tmp], 1, d)
                tmp = tmp_repmat.flatten(order='F')

            tmp = tmp / (abs(vecW) + 10e-10)
            D_exc = sp.diags(tmp, 0, format='csc')

            D = self.lam_net * D_net + self.lam_exc * D_exc

            if self.verbose:
                fval[iter] = (
                    (Ytr - A.dot(vecW))**
                    2).sum() + self.lam_net * U_net + self.lam_exc * U_exc
                print('fval: %f' % fval[iter])

        self.vecW = vecW
        self.W = np.reshape(vecW, (ntr, d), order='F').transpose()
        self.Yest = A.dot(vecW)

    def fit_clustering(self, X, R):

        [d, ntr] = X.shape
        dntr = d * ntr

        vecW = np.ones((dntr, 1))
        index = np.arange(0, dntr)
        val = np.ones(dntr)
        D = sp.csc_matrix((val, (index, index)), shape=(dntr, dntr))

        # Generate input matrix
        vecXtr = X.transpose().flatten(order='F')

        one_d = np.ones(d)
        I_d = sp.diags(one_d, 0, format='csc')

        one_dntr = np.ones(dntr)
        I_dntr = sp.diags(one_dntr, 0, format='csc')

        fval = np.zeros(self.num_iter)
        for iter in range(0, self.num_iter):

            vecW = spsolve(D, vecXtr)

            W = np.reshape(vecW, (ntr, d), order='F')

            tmpNet = cdist(W, W)
            tmp = tmpNet * R

            # Network regularization
            U_net = tmp.sum()

            tmp = 0.5 / (tmpNet + 10e-10) * R

            td1 = sp.diags(tmp.sum(0), 0)
            td2 = sp.diags(tmp.sum(1), 0)

            AA = td1 + td2 - 2.0 * tmp
            AA = (AA + AA.transpose()) * 0.5 + 0.00001 * sp.eye(ntr, ntr)

            D_net = kron(I_d, AA, format='csc')

            # Exclusive regularization
            tmp = abs(W).sum(1)
            U_exc = (tmp * tmp).sum()

            tmp_repmat = repmat(np.c_[tmp], 1, d)
            tmp = tmp_repmat.flatten(order='F')
            tmp = tmp / (abs(vecW) + 10e-10)
            D_exc = sp.diags(tmp, 0, format='csc')

            D = I_dntr + self.lam_net * D_net + self.lam_exc * D_exc

            if self.verbose:
                fval[iter] = (
                    (vecXtr - vecW)**
                    2).sum() + self.lam_net * U_net + self.lam_exc * U_exc
                print('fval: %f' % fval[iter])

        self.vecW = vecW
        self.W = np.reshape(vecW, (ntr, d), order='F').transpose()
