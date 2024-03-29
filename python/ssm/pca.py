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

from . import iovtk

# deformetrica messing with the verbosity level...
#logging.getLogger('matplotlib.font_manager').disabled = True


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

        if self.momenta is None:
            self.read_momenta()

        nsbj = self.momenta.shape[0]
        ndim = self.momenta.shape[1] * self.momenta.shape[2]
        x = self.momenta.reshape((nsbj, ndim))
        self.pca_u, self.pca_s, self.pca_v = scipy.linalg.svd(x)

        if with_plots:
            self.plot_pca_inertia()
            self.plot_pca_projection()

    def predict_low_dim(self, u=None, N=None):
        """
        low dimensional prediction
        u (n, s.size) [default self.pca_u]
        N number of dimension to use [default all=s.size]

        return M0 (n, nctrlpoints, 3)
         """
        if u is None:
            u = self.pca_u
        if N is None:
            N = self.pca_s.size # full prediction should be exact for training data

        M0 = u @ np.diag(self.pca_s)[:, :N] @ self.pca_v[:N, :]
        M0 = M0.reshape((u.shape[0], self.pca_v.shape[1]//3, 3))
        return M0

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
        ax1.set_yscale("log", nonpositive='mask')

        ax2.plot(np.concatenate((np.zeros(1), (self.pca_s**2).cumsum()/(self.pca_s**2).sum())), "+-", linewidth=3, markersize=8)
        ax2.set_title("Cumulative variance")
        ax2.grid(True)

        if save_fig:
            fig.savefig(self.odir + "fig_pca_inertia.png")
        return fig

    def plot_pca_projection(self, axes=(0,1,2,3), save_fig=True, color=None, size=30, labels=None, nmaxlabels=100, with_scale=True, **kwargs):
        """
        plot projection along the 4 axes
        axes,       tuple (4,) axes to show the projection on
        save_fig,   bool       does it save the figure as "fig_pca_projection.png"?
        color,      array (N,) to color the points
        labels,     bool or array (N,)
        nmaxlabels  int       number maximum of labels, labels points away from the center first
        kwargs is used to pass arguments to ax.scatter
                (in particular matplotlib colormap cmap and vmin/vmax)

        return
        fig
        """
        matplotlib.rcParams.update({'font.size': 16})

        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(12, 5), constrained_layout=True)

        if with_scale:
            x1 = self.pca_s[axes[0]]*self.pca_u[:, axes[0]]
            y1 = self.pca_s[axes[1]]*self.pca_u[:, axes[1]]
            x2 = self.pca_s[axes[2]]*self.pca_u[:, axes[2]]
            y2 = self.pca_s[axes[3]]*self.pca_u[:, axes[3]]
        else:
            sigma = 1 / np.sqrt(ao.pca_u.shape[0]) # approx std if centered
            x1 = self.pca_u[:, axes[0]] / sigma
            y1 = self.pca_u[:, axes[1]] / sigma
            x2 = self.pca_u[:, axes[2]] / sigma
            y2 = self.pca_u[:, axes[3]] / sigma

        if color is None:
            ax0.plot(x1, y1, ".", ms=7)
            ax1.plot(x2, y2, ".", ms=7)
        else:
            mp = ax0.scatter(x1, y1, c=color, s=size, **kwargs)
            mp = ax1.scatter(x2, y2, c=color, s=size, **kwargs)
            fig.colorbar(mp)

        if labels:
            N = self.pca_u.shape[0]
            if isinstance(labels, bool):
                labels = [str(i) for i in range(N)]
            assert (len(labels) == N)

            d1 = (x1**2 + y1**2)
            d2 = (x2**2 + y2**2)
            k = N - 1 - nmaxlabels
            if k < 0:
                t1 = -1.
                t2 = -1.
            else:
                t1 = np.sort(d1)[k]
                t2 = np.sort(d2)[k]

            for i in range(N):
                if d1[i]>t1:
                    ax0.text(x1[i], y1[i], labels[i])
                if d2[i]>t2:
                    ax1.text(x2[i], y2[i], labels[i])


        ax0.set_xlabel("pca ax {}".format(axes[0]))
        ax0.set_ylabel("pca ax {}".format(axes[1]))
        ax0.grid(True)
        ax0.axis('equal')

        ax1.set_xlabel("pca ax {}".format(axes[2]))
        ax1.set_ylabel("pca ax {}".format(axes[3]))
        ax1.grid(True)
        ax1.axis('equal')

        if save_fig:
            fig.savefig(self.odir + "fig_pca_projection.png")
        return fig


    def plot_pca_projection2(self, axes=(0,1), color=None, size=30, labels=None, nmaxlabels=100, **kwargs):
        """
        plot projection along the 2 axes
        axes,       tuple (2,) axes to show the projection on
        color,      array (N,) to color the points
        labels,     bool or array (N,)
        nmaxlabels  int       number maximum of labels, labels points away from the center first
        kwargs is used to pass arguments to ax.scatter
                (in particular matplotlib colormap cmap and vmin/vmax)

        return
        fig
        """
        fig, ax = plt.subplots(1,1, figsize=(7, 5))

        sigma = 1 / np.sqrt(self.pca_u.shape[0]) # approx std if centered

        x1 = self.pca_u[:, axes[0]] / sigma
        y1 = self.pca_u[:, axes[1]] / sigma

        if color is None:
            ax.plot(x1, y1, ".", ms=7)
        else:
            mp = ax.scatter(x1, y1, c=color, s=size, **kwargs)
            fig.colorbar(mp, shrink=0.8)

        if labels:
            N = self.pca_u.shape[0]
            if isinstance(labels, bool):
                labels = [str(i) for i in range(N)]
            if (len(labels) == N):
                d1 = (x1**2 + y1**2)
                k = N - 1 - nmaxlabels
                t1 = -1 if k < 0 else np.sort(d1)[k]
                for i in range(N):
                    if d1[i]>t1:
                        ax.text(x1[i], y1[i], labels[i])

        #ax.set_xlabel("pca ax {}".format(axes[0]))
        #ax.set_ylabel("pca ax {}".format(axes[1]))
        ax.grid(True)

        ratio = self.pca_s[axes[1]] / self.pca_s[axes[0]]
        ax.set_aspect(ratio, adjustable='box', anchor='C')

        ax.set_xlim(-3, +3)
        ax.set_ylim(-3, +3)
        ax.set_xticks([-2, -1, +1, +2])
        ax.set_yticks([-2, -1, +1, +2])
        ax.set_xticklabels(['', '-1$\sigma$', '+1$\sigma$', ''])
        ax.set_yticklabels(['', '-1$\sigma$', '+1$\sigma$', ''])

        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return fig



    def project_subject_on_pca(self, x, labels=False):
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

        fig = self.plot_pca_projection(save_fig=False, axes=(0,1,2,3))
        ax0, ax1 = fig.axes

        ax0.plot(y[:, 0], y[:, 1], "+", ms=12, color="C1")
        ax1.plot(y[:, 2], y[:, 3], "+", ms=12, color="C1")

        if labels:
            N = y.shape[0]
            if isinstance(labels, bool):
                labels = [str(i) for i in range(N)]
            else:
                assert (len(labels) == N)

            for i in range(y.shape[0]):
                ax0.text(y[i, 0], y[i, 1], labels[i])
                ax1.text(y[i, 2], y[i, 3], labels[i])

        return fig

    def get_eigv(self, k):
        """ get momemta for the specified axis """
        ncp = self.momenta.shape[1] # number of control points
        A = self.pca_u[:,k].std() * self.pca_s[k] * self.pca_v[k,:].reshape(ncp, 3)
        return A

    def save_eigv(self, k, with_controlpoints=False, fout=None):
        """
        write momemta for the specified axis
        with_controlpoints (bool) write a vtk file with points and pointdata (moments)
        fout (string) file out prefix, append "{k}.txt" (resp. "{k}.vtk")
        """

        if fout is None:
            fv = os.path.normpath(os.path.join(self.odir, "mode{}".format(k)))
        else:
            fv = os.path.normpath(fout + "{}".format(k))

        A = self.get_eigv(k)
        np.savetxt(fv + ".txt", A, fmt="%.6f")

        if with_controlpoints:
            ctrlpts = np.loadtxt(self.idir + "DeterministicAtlas__EstimatedParameters__ControlPoints.txt")
            vtkp = iovtk.controlpoints_to_vtkPoints(ctrlpts, A)
            iovtk.WritePolyData(fv + ".vtk", vtkp)

        return fv
