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

def hotelling_2samples(x, y):
    """
    Calcule la statistique associée au test d'Hotelling d'égalité des moyennes de 2 échantillons.
    Un test est effectué pour chaque voxel.
    Le calcul est (partiellement) vectorisé.
    input:
      - x, np.array (d, nx, p)
      - y, np.array (d, ny, p)
    output:
      - t np.array (d)
    """

    #os.system("taskset -p ff %d" % os.getpid())
    d = x.shape[0]
    p = x.shape[2]

    #print("Hotelling stat of {} vx".format(d))

    # Moyennes
    xm = x.mean(axis=1) # d,p
    ym = y.mean(axis=1) # d,p

    z = xm - ym # d,p

    xd = x - xm[:, np.newaxis, :]
    yd = y - ym[:, np.newaxis, :]

    W = np.zeros((d, p, p))
    for i in range(p):
        W[:, i, i] = np.sum(xd[:, :, i]**2, axis=1) + np.sum(yd[:, :, i]**2, axis=1)
        for j in range(i+1, p):
            W[:, i, j] = np.sum(xd[:, :, i] * xd[:, :, j], axis=1) + np.sum(yd[:, :, i] * yd[:, :, j], axis=1)
            W[:, j, i] = W[:, i, j]
    W /= float(x.shape[1] + y.shape[1] - 2)

    # t2 statistique
    try:
        t = np.sum(z*np.linalg.solve(W, z), axis=1)
    except np.linalg.linalg.LinAlgError as e:
        # matrice singulière ou version de numpy trop vielle (< 1.8.0)
        # where np.linalg.solve can't process (..., M, M) * (..., M,) inputs and return (..., M,)
        print("exception during linalg.solve: ",e)
        t = np.zeros((d))
        for k in range(d):
            try:
                t[k] = np.sum(z[k, :] * np.linalg.solve(W[k, :, :], z[k, :]), axis=0)
            except np.linalg.linalg.LinAlgError:
                print(k, W[k, :, :])
                t[k] = 0.

    t *= float(x.shape[1]*y.shape[1]) / float(x.shape[1]+y.shape[1])
    return t


def fisher_2samples(x, y, joint_mean=True):
    """
    Calcule la statistique associée au test de Fisher d'égalité des variances de 2 échantillons.
    Un test est effectué pour chaque voxel.
    input:
      - x, np.array (d, nx, p)
      - y, np.array (d, ny, p)
    output:
      - t np.array (d)
    """
    nx = x.shape[1]
    ny = y.shape[1]

    if joint_mean:
        m = (x.sum(axis=1) + y.sum(axis=1)) / (nx + ny)
        m = m[:, np.newaxis, :]
        mx = m
        my = m
    else:
        mx = x.mean(axis=1)
        mx = mx[:, np.newaxis, :]

        my = x.mean(axis=1)
        my = my[:, np.newaxis, :]

    z = ((nx - 1) / (ny - 1)) * np.sum((x-mx)**2, axis=(1,2)) / np.sum((y-my)**2, axis=(1,2))
    return z


def aux_1perm_multivariate(X, cat1, cat2=None, rand_seed=-1, fstat=None, name=None):
    """ une permutation
    X (K, N, D) K points, N subjects, D dim

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


def aux_kperm_multivariate(X, cat1, cat2, nperm, fstat=None):
    """
    simple loop, no optimization
    return the maxima sorted
    """
    lmin = np.zeros((nperm,))
    lmax = np.zeros((nperm,))
    for i in range(nperm):
        Y0 = aux_1perm_multivariate(X, cat1, cat2, rand_seed=i, fstat=fstat)
        lmax[i] = Y0.max()
        lmin[i] = Y0.min()
    lmax.sort()
    lmin.sort()
    return lmin, lmax
