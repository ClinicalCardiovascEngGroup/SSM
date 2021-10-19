#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Mesh processing tools (vtk) for atlas estimation

- Rigid/similitud registration (using vtkIterativeClosestPoint, vtkTransform)
- Decimation and clipping

"""

import subprocess as sp
import os, sys

import numpy as np
import vtk


################################################################################
##  Mesh processing and registrations

def DecimatePolyData(pd, reduction=0.9, ntarget=None):
    """
    reduction == 0.9 => réduction to 10% original size
    if ntarget is not None override reduction factor
    vtkDecimatePro only process triangles we use vtkTriangleFilter first
    """
    n = pd.GetPoints().GetNumberOfPoints()
    if ntarget is not None:
        reduction = (n-ntarget)/n
    print("reduction: ", n, reduction)

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(pd)
    triangulate.Update()

    decimate = vtk.vtkDecimatePro()
    #decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(triangulate.GetOutput())
    decimate.SetTargetReduction(reduction)
    decimate.PreserveTopologyOn()
    decimate.Update()

    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputData(decimate.GetOutput())
    triangleFilter.Update()

    return triangleFilter.GetOutput()


def ExtractGeometryZ(pd, nx, ny, nz, z0):
    """
    cut the mesh using z > z0
    could try vtkClipPolyData too
    """
    filter = vtk.vtkExtractPolyDataGeometry()

    function = vtk.vtkPlane()
    function.SetNormal(nx, ny, nz)
    function.SetOrigin(0, 0, z0)

    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputData(pd)
    triangleFilter.Update()

    filter.SetImplicitFunction(function)
    filter.SetInputData(triangleFilter.GetOutput())
    filter.Update()

    #geometryFilter = vtk.vtkGeometryFilter()
    #geometryFilter.SetInputData(filter.GetOutput())
    #geometryFilter.Update()

    connectFilter = vtk.vtkPolyDataConnectivityFilter()
    connectFilter.SetExtractionModeToLargestRegion()
    connectFilter.SetInputData(filter.GetOutput())
    connectFilter.Update()

    return connectFilter.GetOutput()

def LandmarkSimilitudRegistration(fix, mov):
    """ landmarks registration from two vtkPoints """
    transform = vtk.vtkLandmarkTransform()
    transform.SetSourceLandmarks(mov)
    transform.SetTargetLandmarks(fix)
    transform.SetModeToSimilarity()
    transform.Update()
    return transform


def ICPSimilitudRegistration(fix, mov, fix_ldm=None, mov_ldm=None, do_rigid=False):
    """
    IterativeClosestPoint registration between two vtkPolyData
    return the vtkTransform

    can be initialized using landmarks
    (then return a composed vtkTransform instead of a vtkIterativeClosestPointTransform)
    """
    transform = vtk.vtkIterativeClosestPointTransform()
    if do_rigid:
        transform.GetLandmarkTransform().SetModeToRigidBody()
    else:
        transform.GetLandmarkTransform().SetModeToSimilarity()
    transform.SetTarget(fix)

    #transform.SetMaximumNumberOfIterations(10)
    #transform.SetMaximumNumberOfLandmarks(100)

    # initialize using landmarks
    if fix_ldm is not None and mov_ldm is not None:
        ldm_transform = vtk.vtkLandmarkTransform()
        ldm_transform.SetSourceLandmarks(mov_ldm)
        ldm_transform.SetTargetLandmarks(fix_ldm)

        if do_rigid:
            ldm_transform.SetModeToRigidBody()
        else:
            ldm_transform.SetModeToSimilarity()
        ldm_transform.Update()

        warper = vtk.vtkTransformPolyDataFilter()
        warper.SetTransform(ldm_transform)
        warper.SetInputData(mov)
        warper.Update()

        # ICP registration
        transform.SetSource(warper.GetOutput())
        transform.Update()

        composed = vtk.vtkTransform()
        composed.SetInput(transform)
        composed.PreMultiply()
        composed.Concatenate(ldm_transform)
        return composed

    else:
        transform.SetStartByMatchingCentroids(True)

        # ICP registration
        transform.SetSource(mov)
        transform.Update()
        return transform


def apply_transform(transform, mesh):
    """ call vtkTransformPolyDataFilter """
    warper = vtk.vtkTransformPolyDataFilter()
    warper.SetTransform(transform)
    warper.SetInputData(mesh)
    warper.Update()
    return warper.GetOutput()


################################################################################
### Flipping mesh
def flipx(pd, c):
    """ flip mesh along x-axis and center c (only cx matters) """

    # setting up the transform (could be factorised if applied t o several subjects), but not a big overhead
    d = tuple([-x for x in c])

    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(d)
    transform.Scale(-1, 1, 1)
    transform.Translate(c)

    t = vtk.vtkTransformPolyDataFilter()
    t.SetTransform(transform)

    rev = vtk.vtkReverseSense()

    # apply
    t.SetInputData(pd)
    t.Update()

    rev.SetInputData(t.GetOutput())
    rev.Update()

    return rev.GetOutput()
