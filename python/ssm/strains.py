#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Strain measures from deformation

- area strain:
    change in surface of mesh cells
        Kleijn2011 https://www.onlinejase.com/article/S0894-7317(11)00047-2/pdf
        Jia2017 https://hal.inria.fr/hal-01574831/document


"""

import subprocess as sp
import os, sys

import numpy as np
import torch

import vtk

from vtk.util import numpy_support as nps

import deformetrica

################################################################################
##  Vessel coordinate system
def _closest_points(x, y):
    """
    x P, D
    y K, D
    return
    idx P, such that y[idx[i]] closest point to x[i] in y
    """
    P = x.shape[0]
    idx = np.zeros((P,), dtype="uint32")
    for i in range(P):
        d = np.sum((y - np.expand_dims(x[i], 0))**2, axis=1) # K,
        idx[i] = np.argmin(d)
    return idx


def surface_coordinate_systems(pd, cl=None):
    """
    for each cell of polydata pd compute:
        - x, center      # P, D
        - cx, normal     # P, D
        - cy, longitudinal direction using centerline  # P, D
        - cz, last vector for orthogonal coords        # P, D
    """

    # cell centers
    cell_centers = vtk.vtkCellCenters()
    cell_centers.SetInputData(pd)
    cell_centers.Update()
    vtk_x = cell_centers.GetOutput().GetPoints().GetData()
    x = nps.vtk_to_numpy(vtk_x) # P, D

    # cell normals
    cell_normals = vtk.vtkPolyDataNormals()
    cell_normals.SetInputData(pd)
    cell_normals.ComputeCellNormalsOn()
    cell_normals.ComputePointNormalsOff()
    cell_normals.ConsistencyOn()
    cell_normals.AutoOrientNormalsOn()
    cell_normals.Update()
    vtk_cn = cell_normals.GetOutput().GetCellData().GetNormals()
    cx = nps.vtk_to_numpy(vtk_cn) # P, D

    # direction of closest point on centerline
    if cl is None:
        cy=None
        cz=None
    else:
        apts = nps.vtk_to_numpy(cl.GetPoints().GetData()) # K, D
        adir = apts[+1:, :] - apts[:-1, :] # could get some smoothing
        adir = np.concatenate((adir, adir[-1:, :]), axis=0)

        closest_points_idx = _closest_points(x, apts)
        cy = adir[closest_points_idx, :]
        cy -= np.expand_dims(np.sum(cx * cy, axis=1)/np.sum(cx * cx, axis=1), 1) * cx # orthogonal
        cy /= np.expand_dims(np.sqrt((cy**2).sum(axis=1)), 1)         # normal

        # cross-product
        cz = np.cross(cx, cy)

    return x, cx, cy, cz

def coordinate_systems_to_vtk(x, cx, cy, cz):
    """ vtk points with pointdata """

    vpts = vtk.util.numpy_support.numpy_to_vtk(x)
    points = vtk.vtkPoints()
    points.SetData(vpts)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)

    vax = vtk.util.numpy_support.numpy_to_vtk(cx)
    vax.SetName("cx")
    pd.GetPointData().AddArray(vax)
    vax = vtk.util.numpy_support.numpy_to_vtk(cy)
    vax.SetName("cy")
    pd.GetPointData().AddArray(vax)
    vax = vtk.util.numpy_support.numpy_to_vtk(cz)
    vax.SetName("cz")
    pd.GetPointData().AddArray(vax)

    return pd


################################################################################
##  Area strain
def compute_cells_area(m):
    """ numpy array with cells area """
    areaf = vtk.vtkMeshQuality()
    areaf.SetInputData(m)
    areaf.SetTriangleQualityMeasureToArea()
    areaf.SaveCellQualityOn() ##default: quality is stored as cell data
    areaf.Update()
    vtk_x = areaf.GetOutput().GetCellData().GetArray("Quality")
    np_x = nps.vtk_to_numpy(vtk_x)
    return np_x


def area_strain(mflo, mref):
    """ area strain from mflo to mref """
    xf = compute_cells_area(mflo)
    xr = compute_cells_area(mref)
    s = (xf - xr) / xr

    strainArray = nps.numpy_to_vtk(s)
    strainArray.SetName("AreaStrain")
    mflo.GetCellData().AddArray(strainArray)

    return mflo, s

################################################################################
##  Mechanical strain
def gradU_from_momenta(x, p, y, sigma):
    """
    strain F'(x) for momenta p defined at control points y
    a method "convolve_gradient" is doing a similar job but only compute (gradF . z)

    x (M, D)
    p (N, D)
    y (N, D)

    return
    gradU (M, D, D)
    """
    kern = deformetrica.support.kernels.factory("torch", gpu_mode=False, kernel_width=sigma)

    # move tensors with respect to gpu_mode
    t_x = torch.tensor(x, device="cpu")
    t_y = torch.tensor(y, device="cpu")
    t_p = torch.tensor(p, device="cpu")

    # A = exp(-(x_i - y_j)^2/(ker^2)).
    sq = kern._squared_distances(t_x, t_y)
    A = torch.exp(-sq / (sigma ** 2)) # M, N

    # B = -2/(ker^2) * (x_i - y_j)*exp(-(x_i - y_j)^2/(ker^2)).
    B = (-2/(sigma ** 2)) * kern._differences(t_x, t_y) * A   # (D, M, N)

    res = torch.matmul(B, t_p) # (D, M, D)
    return np.array(res.transpose(0,1))



def strain_tensor_on_surface(pd, M, C, sigma, cl=None):
    """
    normal and shear strain at each cell of polydata pd
    deformation given by momenta M, control points C and kernel width sigma
    cl is a centerline polydata used to define surface coord sys

    ## Notes:
    a good measure could be 'strain energy density function'
    that is also quite simple for hyperelastic material
    (https://en.wikipedia.org/wiki/Hyperelastic_material)
    and I can probably derive something for a fine surface
    W(E) = ... tr(E)**2 + ... tr(E**2) avec E Lagrangian Green strain (symetric strain)

    For now, I use strain/geometric changes
    - normal_strain = n^T . S . n (S strain tensor)
    - shear_strain  = trace(S) - normal_stress
    more or less dx and dy+dz... (if x along normal)

    ## Inputs
    pd P cells
    M (N, D)
    C (N, D)
    sigma kernel width
    cl centerline polydata (use to define coord sys)

    ## Return
    pd, with new cell arrays
    normal_stress (P, )
    shear_stress  (P, )
    """

    # cell centers
    cx, cnx, cny, cnz = surface_coordinate_systems(pd, cl) # P, D

    # Gradient of infinitesimal displacement
    S = gradU_from_momenta(cx, M, C, sigma) # P, D, D

    # Metrics
    divergence = np.trace(S, axis1=1, axis2=2)
    cncn = np.expand_dims(cnx, 1) * np.expand_dims(cnx, 2)
    normal_strain = np.sum(S * cncn, axis=(1,2)) # sum( ci * cj * sij)
    #shear_stress = divergence - normal_stress

    # vtk arrays
    stressArray = nps.numpy_to_vtk(divergence)
    stressArray.SetName("Trace")
    pd.GetCellData().AddArray(stressArray)

    stressArray = nps.numpy_to_vtk(normal_strain)
    stressArray.SetName("NormalStrain")
    pd.GetCellData().AddArray(stressArray)

    # base change
    if cny is not None and cnz is not None:

        R = np.zeros((S.shape[0], 9))
        for i in range(S.shape[0]):
            B = np.stack((cnx[i, :], cny[i, :], cnz[i, :]), axis=1)
            R[i, :] = (B.T @ S[i, :, :] @ B).flatten()

        strainArray = nps.numpy_to_vtk(R)
        strainArray.SetName("Strain")
        pd.GetCellData().AddArray(strainArray)

    else:
        R = None

    return pd, S, R
