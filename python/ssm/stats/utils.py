#!/usr/bin/python
# coding: utf8

""" common utils """

import numpy as np


def compute_pvalues(zmap, zsamp, alpha_threshold=1.,):
    """
    Compute p-values on the right only using:
    - zmap   (vdim,)   z-values
    - zsamp  (nsamp,)  empirical z samples

    return smallest k such that !(zmap[i] > zsamp[k])

    can perform asymetric search on right and left

    (todo voir np.searchsorted)
    """
    vdim = zmap.size
    nsamp = zsamp.size

    if nsamp == 1:
        zsamp.shape = (1,)

    k_threshold = (1. - alpha_threshold)*nsamp

    pval = np.zeros(vdim, dtype="uint16")
    for i in range(vdim):
        k = 0
        while k < nsamp and zmap[i] > zsamp[k]:
            k += 1
        if k >= k_threshold:
            pval[i] = k
    return pval
