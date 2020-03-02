import numpy as np
import os

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans


def select_random(X, k, seed):
    n, d = X.shape
    np.random.seed(seed)
    index = np.arange(n)
    # np.random.shuffle(index)
    index = index[:k]
    return X[index], index


def select_kmeans(X, k, seed):
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=seed).fit(X)
    return kmeans.cluster_centers_, np.argmin(cdist(X, kmeans.cluster_centers_), axis=0)


def select_kmeans_with_gt(X, y, k, seed):
    newX = np.concatenate((X, y), axis=1)
    kmeans = KMeans(n_clusters=k, init="k-means++", random_state=seed).fit(newX)
    return kmeans.cluster_centers_[..., :-1], np.argmin(cdist(newX, kmeans.cluster_centers_), axis=0)


def find_nn(X, X_anchors, K, y=None):
    C = cdist(X, X_anchors)
    ind_sets = np.argsort(C, axis=0)[:K]
    X_sets = X[ind_sets].transpose(1, 0, 2)
    if y is None:
        return X_sets, ind_sets
    else:
        return X_sets, ind_sets, y[ind_sets].transpose(1, 0, 2)


def add_bias(X):
    return np.hstack((X, np.ones((len(X), 1))))
    

def get_random_split(X, y, prop, seed):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=(1-prop), random_state=seed)


def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
