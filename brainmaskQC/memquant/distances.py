from typing import Union, Tuple

import numpy as np


def mahalanobis(setA: np.ndarray, setB: np.ndarray) -> np.ndarray:
    setA_minus_mu = setA - np.mean(setB, axis=0)
    cov = np.cov(setB.T)
    # in case features are 1D
    if setA.shape[-1] == 1:
        cov = np.array([[cov]])
    if np.isnan(cov).any():
        cov = np.eye(*cov.shape)
    inv_cov = np.linalg.pinv(cov)
    mahal = np.dot(np.dot(setA_minus_mu, inv_cov), setA_minus_mu.T)

    return mahal.diagonal()


def eval_mahalanobis_metrics(
    setA: np.ndarray, setB: np.ndarray, labels: np.ndarray
) -> Tuple[np.ndarray]:
    # get n_clusters
    n_clusters = np.max(labels) + 1

    # keep track of weighted mean mahalanobis distance
    w_mean_mahal_dist = np.zeros((len(setA)), dtype=float)
    # keep track of all mahalanobis distances for evaluating mean
    all_mahal_dist = np.zeros((n_clusters, len(setA)), dtype=float)

    # iterate through clusters
    for l in range(n_clusters):
        cluster = setB[labels == l]
        if len(cluster) >= 1:
            all_mahal_dist[l] = mahalanobis(setA, cluster)
            w_mean_mahal_dist += all_mahal_dist[l] * (len(cluster) / len(setB))
        else:
            all_mahal_dist[l] = np.infty

    min_mahal_dist = np.min(all_mahal_dist, axis=0)
    median_mahal_dist = np.median(all_mahal_dist, axis=0)

    return (min_mahal_dist, w_mean_mahal_dist, median_mahal_dist)


def eval_intra_cohort_metrics(X: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray]:
    # given existing cohort, return the min and mean mahalanobis

    # construct filter array used for filtering out each individual subject one at a time
    filter_arr = np.ones((X.shape[0]), dtype=int)

    # initialize minimum and mean mahalanobis distances
    min_mahal_dists = np.zeros((X.shape[0]), dtype=float)
    mean_mahal_dists = np.zeros((X.shape[0]), dtype=float)
    median_mahal_dists = np.zeros((X.shape[0]), dtype=float)

    # iterate through subjects
    for i in range(X.shape[0]):
        # set filter to exlude ith index
        filter_arr[i] = 0

        # split into two sets for evaluating mahalanobis
        set_A = X[filter_arr == 0]
        set_B = X[filter_arr == 1]

        # filter labels as well
        labels_B = labels[filter_arr == 1]

        # get distances and store
        distances = eval_mahalanobis_metrics(set_A, set_B, labels_B)
        min_mahal_dists[i] = distances[0]
        mean_mahal_dists[i] = distances[1]
        median_mahal_dists[i] = distances[2]

        # reinclude ith index for next iteration
        filter_arr[i] = 1

    return (min_mahal_dists, mean_mahal_dists, median_mahal_dists)


def get_nth_percentile(arr: np.ndarray, n: Union[int, float]) -> float:
    # assert that n is n appropriate value
    if isinstance(n, int):
        assert n >= 0 and n < 100
        cutoff = n / 100
    else:
        assert n >= 0 and n < 1
        cutoff = n

    arr_l = arr.tolist()
    arr_l.sort()

    return arr_l[int(len(arr_l) * cutoff)]
