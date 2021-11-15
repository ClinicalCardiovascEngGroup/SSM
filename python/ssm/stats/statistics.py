#!/usr/bin/python
# coding: utf8

"""
Statistics

be careful of the dimension ordering:
d,n,p or n,d,p

"""

import numpy as np


def hotelling_1sample(x):
    """
    Calcule la statistique associée au test d'Hotelling de nullité de la moyenne.
    Un test est effectué pour chaque voxel.
    Le calcul est (partiellement) vectorisé.
    input:
      - x, np.array (d, nx, p)
    output:
      - t np.array (d)
    """

    d, n, p = x.shape

    # Moyenne
    xm = x.mean(axis=1) # d,p

    # Covariance empirique
    W = np.zeros((d, p, p))
    for k in range(d):
        W[k, :, :] = np.dot(x[k, :, :].transpose(), x[k, :, :])
    W /= float(n - 1)

    # t2 statistique
    try:
        t = np.sum(xm * np.linalg.solve(W, xm), axis=1)
    except np.linalg.linalg.LinAlgError as e:
        # matrice singulière ou version de numpy trop vielle (< 1.8.0)
        # where np.linalg.solve can't process (..., M, M) * (..., M,) inputs
        print("exception during linalg.solve: ", e)
        t = np.zeros((d))
        for k in xrange(d):
            try:
                t[k] = np.sum(xm[k, :] * np.linalg.solve(W[k, :, :], xm[k, :]), axis=0)
            except np.linalg.linalg.LinAlgError:
                print(k, W[k, :, :])
                t[k] = 0.

    t *= n
    return t


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

        my = y.mean(axis=1)
        my = my[:, np.newaxis, :]

    z = ((ny - 1) / (nx - 1)) * np.sum((x-mx)**2, axis=(1,2)) / np.sum((y-my)**2, axis=(1,2))
    return z


def regression_loglikelihood(X, y):
    """
    W: this one uses: n, d, p and not d, n, p !!
    without optimization this one makes more sense


    for each point:
    x = y @ A + E
    return
    l = -(n/2) * log(det(E.T @ E))

    X (n, d, p)
    y (n, q)
    """
    n, d, _ = X.shape

    Z = np.zeros((d,))
    for i in range(d):
        x = X[:, i, :]
        A, _,_,_ = np.linalg.lstsq(y, x, rcond=None)
        E = x - y@A
        Z[i] = -(n/2) * np.log(np.linalg.det(E.T @ E))

    return Z
