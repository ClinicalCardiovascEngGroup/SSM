#!/usr/bin/python
# coding: utf8

"""
Simplified version of stat_t_test (without multiprocessing)
used to compute Hotelling statistic

threshold for independant testing:
  rv = scipy.stats.t(df=(n1+n2-2))
  th = rv.ppf(0.95)
"""

import numpy as np

from .statistics import *


################################################################################


def zmap_1perm_2samp(X, cat1, cat2=None, rand_seed=-1, fstat=None, name=None):
    """ une permutation
    X (D, N, P) K points, N subjects, D dim
    return:
    Y (D,) zvalue at each point
    """
    if fstat is None:
        fstat = hotelling_2samples
        #name = "MP-Hotelling"

    if cat2 is None:
        cat2 = np.logical_not(cat1)

    # Données
    if rand_seed < 0:
        # Sans permutation (on peut remplacer cat par idx[cat])
        ix1 = cat1
        ix2 = cat2
    else:
        # Avec permutation
        np.random.seed(rand_seed)
        idx = np.arange(X.shape[1])[cat1 | cat2]
        per = np.random.permutation(idx.size)
        nsplit = cat1.sum()
        ix1 = idx[per][:nsplit]
        ix2 = idx[per][nsplit:]

    # Run
    Y = fstat(X[:, ix1, :], X[:, ix2, :])

    if name is not None:
        print(name + " {0}, {1}\n".format(Y.min(), Y.max()))
    return Y


def zval_kperm_2samp(X, cat1, cat2=None, nperm=100, fstat=None):
    """
    simple loop, no optimization
    return the extrema sorted
    return:
    lmin, lmax
    """
    if cat2 is None:
        cat2 = np.logical_not(cat1)

    lmin = np.zeros((nperm,))
    lmax = np.zeros((nperm,))
    for i in range(nperm):
        Y0 = zmap_1perm_2samp(X, cat1, cat2, rand_seed=i, fstat=fstat)
        lmax[i] = Y0.max()
        lmin[i] = Y0.min()
    lmax.sort()
    lmin.sort()
    return lmin, lmax


################################################################################


def zmap_1perm_2pairedsamp(X, rand_seed=-1, name=None):
    """ une permutation (changement de signe)
    X (D, N, P) D points, N subjects, P dim
    return:
    Y (D,) zvalue at each point
    """
    (d, n, p) = X.shape

    # Données
    if rand_seed < 0:
        # Sans permutation
        sign_swap = np.ones((1, n, 1))
    else:
        # Avec permutation
        sign_swap = np.random.randint(0, 2, size=(1, n, 1))
        sign_swap = 2*sign_swap - 1

    # Run
    Y = hotelling_1sample(X * sign_swap)

    if name is not None:
        print(name + " {0}, {1}\n".format(Y.min(), Y.max()))
    return Y


def zval_kperm_2pairedsamp(X, nperm):
    """
    simple loop, no optimization
    return the extrema sorted
    return:
    lmin, lmax
    """
    lmin = np.zeros((nperm,))
    lmax = np.zeros((nperm,))
    for i in range(nperm):
        Y0 = zmap_1perm_2pairedsamp(X, rand_seed=i)
        lmax[i] = Y0.max()
        lmin[i] = Y0.min()
    lmax.sort()
    lmin.sort()
    return lmin, lmax


################################################################################

def zmap_kperm_llh(X, y, nperm, nvar0=1, mode="cst"):
    """
    log-likelihood testing for non parametric permutation regression testing
    nvar0 is used to set the reference model

    2 options are possible (mode)
    - mode == "cst" the reference model is not permuted
    - mode == "per" the reference model is permuted with the same permutations (useful for joint model for example)

    X (n, d, p) deformation
    y (n, q)    clinicals

    return
    L (d, nperm+1)
    L[:, 0],                   no-permutation map
    np.max(L[:, 1:], axis=0)   permutation maxs
    """
    n, d, p = X.shape
    if mode == "cst":
        K = np.zeros((d, 1))
    elif mode == "per":
        K = np.zeros((d, nperm+1))
    else:
        raise ValueError("mode for zmap_kperm_llh unknown:", mode)
    L = np.zeros((d, nperm+1))

    # reference model and no permutation
    K[:, 0] = regression_loglikelihood(X, y[:, :nvar0])
    L[:, 0] = regression_loglikelihood(X, y)

    # with perm
    for k in range(nperm):
        #np.random.seed(k)
        per = np.random.permutation(n)
        if mode == "cst":
            yp = y.copy()
            yp[:, nvar0:] = y[per, nvar0:]
            L[:, k+1] = regression_loglikelihood(X, yp)
        elif mode == "per":
            K[:, k+1] = regression_loglikelihood(X, y[per, :nvar0])
            L[:, k+1] = regression_loglikelihood(X, y[per, :])
        else:
            pass

    L = L - K
    return L, L[:, 0], np.max(L[:, 1:], axis=0)



################################################################################
def compute_pvalues(zmap, zsamp, side="right", alpha_threshold=1., do_sort=True):
    """
    Compute p-values using:
    - zmap   (vdim,)   z-values
    - zsamp  (nsamp,)  empirical z samples
    - side   'right' or 'left'

    return smallest k such that !(zmap[i] > zsamp[k])

    (voir np.searchsorted)
    """

    vdim = zmap.size
    nsamp = zsamp.size
    if nsamp == 1:
        zsamp.shape = (1,)

    if do_sort:
        zsamp.sort()

    if side == "right":
        pass
    elif side == "left":
        zmap = -zmap
        zsamp = -zsamp[::-1]
    else:
        raise ValueError("unrecognized pvalue side {}".format(side))

    pval = np.searchsorted(zsamp, zmap, 'left')

    pval[(pval/nsamp) < (1. - alpha_threshold)] = 0

    return pval
