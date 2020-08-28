#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
PCA modes visualisation for momenta related to a Deformetrica atlas estimation


"""


import subprocess as sp
import os, sys

import numpy as np
import vtk
from vtk.util import numpy_support as nps
import torch


import scipy, scipy.linalg
import matplotlib
import matplotlib.pyplot as plt

# modules api, core, in_out are part of deformetrica \o/
import support.kernels as kernel_factory



import logging
logger = logging.getLogger("ssm_pca")
logger.setLevel(logging.INFO)

# deformetrica messing with the verbosity level...
logging.getLogger('matplotlib.font_manager').disabled = True




################################################################################
##  Deformetrica outputs data class

class DeformetricaAtlasOutput():
    """ result of an DeterministicAtlas estimation """

    def __init__(self, idir="", odir=""):
        self.idir = idir
        self.odir = odir
        self.momenta = None
        self.kw = None

        if odir != "":
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

    def plot_pca_inertia(self):
        """ plot inertia and cumulative variance of pca modes """
        fig, (ax0, ax1) = plt.subplots(1,2)
        ax0.plot(self.pca_s)
        ax1.plot((self.pca_s**2).cumsum()/(self.pca_s**2).sum())
        ax0.set_title("Eigvalues")
        ax1.set_title("Cumulative variance")

        fig.savefig(self.odir + "fig_pca_inertia.png")
        return fig

    def plot_pca_projection(self):
        """ plot projection along the 4 first axes """

        fig, (ax0,ax1) = plt.subplots(1,2, figsize=(12,6))

        ax0.plot(self.pca_u[:, 0], self.pca_u[:, 1], ".", ms=7)
        ax0.set_xlabel("eig0")
        ax0.set_ylabel("eig1")
        ax0.grid(True)

        ax1.plot(self.pca_u[:, 2], self.pca_u[:, 3], ".", ms=7)
        ax1.set_xlabel("eig2")
        ax1.set_ylabel("eig3")
        ax1.grid(True)

        fig.savefig(self.odir + "fig_pca_projection.png")
        return fig

    def get_eigv(self, k):
        """ write momemta for the specified axis """
        ncp = self.momenta.shape[1] # number of control points
        A = self.pca_u[:,k].std() * self.pca_s[k] * self.pca_v[k,:].reshape(ncp, 3)
        return A

    def save_eigv(self, k):
        """ write momemta for the specified axis """
        A = self.get_eigv(k)
        fv = os.path.normpath(os.path.join(self.odir, "mode{}.txt".format(k)))
        np.savetxt(fv, A, fmt="%.6f")
        return fv

    def read_template(self, name):
        """ return polydata of the template """
        ft = self.idir + "DeterministicAtlas__EstimatedParameters__Template_{}.vtk".format(name)
        ft = os.path.abspath(ft).encode()

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(ft)
        reader.Update()
        v_pd = reader.GetOutput()
        return v_pd

    def convolve_momentum(self, m, x):
        """
        kernel convolution of momenta at points x
            m : np.array K, 3,  K=number of controls points,
                                ex: m=self.momenta[0,:], m=get_eigv(0)
            kw: float,          kernel width
            x : np.array N, 3,  coordinates
        """

        kern = kernel_factory.factory("torch", gpu_mode=True, kernel_width=self.kw)

        a_cp = np.loadtxt(self.idir + "DeterministicAtlas__EstimatedParameters__ControlPoints.txt")
        assert a_cp.shape == m.shape

        t_y = torch.tensor(a_cp, device="cpu")
        t_x = torch.tensor(x, device="cpu")
        t_p = torch.tensor(m, device="cpu")

        t_z = kern.convolve(t_x, t_y, t_p)
        return np.array(t_z)


    def render_momenta_norm(self, moments, name):
        """ render the norm of the momenta on the template geometry """

        # read mesh
        v_pd = self.read_template(name)
        N = v_pd.GetPoints().GetNumberOfPoints()
        points = nps.vtk_to_numpy(v_pd.GetPoints().GetData()).astype('float64')

        # compute attributes
        z = self.convolve_momentum(moments, points)
        print(z.shape)
        d = np.sum(z**2, axis=1)
        assert d.size == N

        # set attributes
        scalars = vtk.vtkFloatArray()
        for i in range(N):
            scalars.InsertTuple1(i, d[i])
        v_pd.GetPointData().SetScalars(scalars)

        # render
        renderVtkPolyData(v_pd, vmin=0., vmax=d.max())

################################################################################
##  IO


import glob, re

def rename_df2pv(prefix):
    """ changing suffixe to easily load in paraview """

    def key_tp(f):
        m = re.search("tp_(\d+)__age", f)
        return int(m.group(1))

    fl = glob.glob(prefix + "__tp_*__age_*.vtk")
    fl.sort(key=key_tp)

    print("Renaming files for paraview")
    print("  regex: ", prefix + "__tp_*__age_*.vtk")
    print("  nb of files: ", len(fl))

    for i,f in enumerate(fl):
        sp.call(["mv", f, prefix + "_tp{:03}.vtk".format(i)])


def renderVtkPolyData(pd, vmin=0., vmax=1.):
    """ render a vtk polydata mesh with scalar pointdata """

    # Now we'll look at it.
    cubeMapper = vtk.vtkPolyDataMapper()
    cubeMapper.SetInputData(pd)
    cubeMapper.SetScalarRange(vmin, vmax)
    cubeActor = vtk.vtkActor()
    cubeActor.SetMapper(cubeMapper)
    print("vtk render: mapping scalar between: ", vmin, vmax)

    # The usual rendering stuff.
    camera = vtk.vtkCamera()
    camera.SetPosition(1,1,1)
    camera.SetFocalPoint(0,0,0)

    renderer = vtk.vtkRenderer()
    renWin   = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)

    iren = vtk.vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)

    renderer.AddActor(cubeActor)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()
    renderer.SetBackground(1,1,1)

    renWin.SetSize(300,300)

    # interact with data
    renWin.Render()
    iren.Start()
