#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Distances between meshes
"""

import subprocess as sp
import os, sys

import numpy as np
import vtk

from vtk.util import numpy_support as nps

################################################################################
##  Mesh distance

def polydata_distance(mflo, mref, do_signed=True):
    """
    distance from mflo to mref
    /!\ (silently) do not work properly if the cells are not triangles 
    """
    pdd = vtk.vtkDistancePolyDataFilter()
    pdd.SetSignedDistance(do_signed) #pdd.SignedDistanceOff()
    pdd.SetInputData(0, mflo)
    pdd.SetInputData(1, mref)

    pdd.Update()
    return pdd.GetOutput()

def polydata_distance_on_ref(mflo, mref, mtpl, do_signed=True):
    """ distance from mflo to mref recorded on mtpl
    mflo and mtpl must have same number (and matching) points
    """
    mout = vtk.vtkPolyData()
    mout.DeepCopy(mtpl)

    pdd = vtk.vtkImplicitPolyDataDistance()
    pdd.SetInput(mref)

    numPts = mflo.GetNumberOfPoints()
    distArray = vtk.vtkDoubleArray()
    distArray.SetName("Distance")
    distArray.SetNumberOfComponents(1)
    distArray.SetNumberOfTuples(numPts)
    for i in range(numPts):
        pt = mflo.GetPoint(i)
        d = pdd.EvaluateFunction(pt)
        if do_signed:
            distArray.SetValue(i, d)
        else:
            distArray.SetValue(i, np.abs(d))

    mout.GetPointData().AddArray(distArray)
    return mout

def average_surface_distance(mflo, mref):
    """ average on points so not reliable if the sampling is unhomogeneous """
    pd = polydata_distance(mflo, mref, do_signed=False)
    xv = pd.GetPointData().GetArray("Distance")
    xn = nps.vtk_to_numpy(xv)
    return xn.mean()


################################################################################
##  Deformetrica attachment
def current_distance(fflo, fref, kw, kernel_type="torch"):
    """
    current distance between two surface meshes using deformetrica
    instantiate a torch kernel
    """
    import deformetrica

    reader = deformetrica.in_out.DeformableObjectReader()
    obj1 = reader.create_object(fflo, 'SurfaceMesh', )
    obj2 = reader.create_object(fref, 'SurfaceMesh', )

    kern = deformetrica.support.kernels.factory(kernel_type, kernel_width=kw)

    # static method
    # moa = deformetrica.core.model_tools.attachments.MultiObjectAttachment(['current'], [kern])
    current_distance = deformetrica.core.model_tools.attachments.MultiObjectAttachment.current_distance

    return current_distance(obj1.points, obj1, obj2, kern)
