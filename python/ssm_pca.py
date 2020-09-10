#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
PCA modes visualisation for momenta related to a Deformetrica atlas estimation


"""


import subprocess as sp
import os, sys

import numpy as np


import scipy, scipy.linalg
import matplotlib
import matplotlib.pyplot as plt

import ssm_tools


import logging
logger = logging.getLogger("ssm_pca")
logger.setLevel(logging.INFO)

# deformetrica messing with the verbosity level...
logging.getLogger('matplotlib.font_manager').disabled = True




################################################################################
##  Deformetrica outputs data class

class DeformetricaAtlasPCA():
    """ result of an DeterministicAtlas estimation """

    def __init__(self, idir, odir):
        self.idir = idir
        self.odir = odir
        self.momenta = None

        if not os.path.exists(odir):
            sp.call(["mkdir", "-p", self.odir])

    def read_momenta(self):
        """
        read the momenta file, first line contain the shape
        """
        a_momenta = np.loadtxt(self.idir + "DeterministicAtlas__EstimatedParameters__Momenta.txt")
        shape = a_momenta[0,:].astype("int")
        self.momenta = a_momenta[1:, :].reshape(shape)
        print("subjects, control_points, dim:", self.momenta.shape)

        return self.momenta

    def central_point(self):
        """ using the momenta, return:
            - the mean momentum
            - the closest subject (id)
        """
        m = self.momenta.mean(axis=0)
        d = np.sum((self.momenta - m)**2, axis=(1,2))
        i = d.argmin()
        return m, i

    def compute_pca(self, with_plots=True):
        """ M == U @ S @ V """
        self.read_momenta()

        nsbj = self.momenta.shape[0]
        ndim = self.momenta.shape[1] * self.momenta.shape[2]
        x = self.momenta.reshape((nsbj, ndim))
        self.pca_u, self.pca_s, self.pca_v = scipy.linalg.svd(x)

        if with_plots:
            self.plot_pca_inertia()
            self.plot_pca_projection()

    def plot_pca_inertia(self, save_fig=True):
        """ plot inertia and cumulative variance of pca modes """

        matplotlib.rcParams.update({'font.size': 16})

        fig, (ax0, ax1, ax2) = plt.subplots(1,3, figsize=(20,7))

        ax0.plot(self.pca_s, "-", linewidth=3)
        ax0.set_title("Eigvalues")
        ax0.grid(True)

        ax1.plot(self.pca_s[:10], "+-", linewidth=3, markersize=12, markeredgewidth=3)
        ax1.set_title("First eigvalues (log-scale)")
        ax1.grid(True)
        ax1.set_yscale("log", nonposy='mask')

        ax2.plot((self.pca_s**2).cumsum()/(self.pca_s**2).sum(), "+-", linewidth=3, markersize=8)
        ax2.set_title("Cumulative variance")
        ax2.grid(True)

        if save_fig:
            fig.savefig(self.odir + "fig_pca_inertia.png")
        return fig

    def plot_pca_projection(self, save_fig=True, color=None):
        """ plot projection along the 4 first axes """

        matplotlib.rcParams.update({'font.size': 16})

        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(14, 6))

        if color is None:
            ax0.plot(self.pca_s[0]*self.pca_u[:, 0], self.pca_s[1]*self.pca_u[:, 1], ".", ms=7)
            ax1.plot(self.pca_s[2]*self.pca_u[:, 2], self.pca_s[3]*self.pca_u[:, 3], ".", ms=7)
        else:
            mp = ax0.scatter(self.pca_s[0]*self.pca_u[:, 0], self.pca_s[1]*self.pca_u[:, 1], c=color, s=30)
            mp = ax1.scatter(self.pca_s[2]*self.pca_u[:, 2], self.pca_s[3]*self.pca_u[:, 3], c=color, s=30)
            plt.colorbar(mp)

        ax0.set_xlabel("eig0")
        ax0.set_ylabel("eig1")
        ax0.grid(True)
        ax0.axis('equal')


        ax1.set_xlabel("eig2")
        ax1.set_ylabel("eig3")
        ax1.grid(True)
        ax1.axis('equal')

        if save_fig:
            fig.savefig(self.odir + "fig_pca_projection.png")
        return fig

    def project_subject_on_pca(self, x):
        """
        project subjects (momentum) on pca axes
        x (n, mdim), mdim=nctrl*3 or momenta file
        """
        if isinstance(x, str):
            m = np.loadtxt(x)
            shape = m[0,:].astype("int")
            x = m[1:, :].reshape((shape[0], shape[1]*shape[2]))

        elif x.ndim == 3:
            x = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
        elif x.ndim == 1:
            x = x.reshape((1, x.size))

        y = x @ self.pca_v.T

        fig = self.plot_pca_projection(save_fig=False)
        ax0, ax1 = fig.axes

        ax0.plot(y[:, 0], y[:, 1], "+", ms=12, color="C1")
        ax1.plot(y[:, 2], y[:, 3], "+", ms=12, color="C1")

        return fig

    def get_eigv(self, k):
        """ write momemta for the specified axis """
        ncp = self.momenta.shape[1] # number of control points
        A = self.pca_u[:,k].std() * self.pca_s[k] * self.pca_v[k,:].reshape(ncp, 3)
        return A

    def save_eigv(self, k, with_controlpoints=False):
        """ write momemta for the specified axis """
        A = self.get_eigv(k)
        fv = os.path.normpath(os.path.join(self.odir, "mode{}.txt".format(k)))
        np.savetxt(fv, A, fmt="%.6f")

        if with_controlpoints:
            ctrlpts = np.loadtxt(self.idir + "DeterministicAtlas__EstimatedParameters__ControlPoints.txt")
            vtkp = ssm_tools.controlpoints_to_vtkPoints(ctrlpts, A)
            fu = os.path.normpath(os.path.join(self.odir, "mode{}.vtk".format(k)))
            ssm_tools.WritePolyData(fu, vtkp)

        return fv
