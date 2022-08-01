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

def _read_polydata_from_unstructuredgrid(fi):
    '''
    we need polydata,
    two filters that could be useful if you have unstructured grid
    kind of a (stupid) cheatsheet, should it two functions?
    '''
    ugreader = vtk.vtkUnstructuredGridReader()
    ugreader.SetFileName(fi)
    ugreader.Update()
    ug = ugreader.GetOutput()
    gf = vtk.vtkGeometryFilter()
    gf.SetInputData(ug)
    gf.Update()
    return gf.GetOutput()

def triangle_filter(pd):
    ''' only triangle cells '''
    tri_filter = vtk.vtkTriangleFilter()
    tri_filter.SetInputData(pd)
    tri_filter.Update()
    return tri_filter.GetOutput()


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

def apply_transform(transform, mesh):
    """ call vtkTransformPolyDataFilter """
    warper = vtk.vtkTransformPolyDataFilter()
    warper.SetTransform(transform)
    warper.SetInputData(mesh)
    warper.Update()
    return warper.GetOutput()


def ICPSimilitudRegistration(fix, mov, fix_ldm=None, mov_ldm=None, do_rigid=False, do_apply=False):
    """
    IterativeClosestPoint registration between two vtkPolyData
    return the vtkTransform (and the transformed mesh if do_apply)

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
        tfm = composed


    else:
        transform.SetStartByMatchingCentroids(True)
        #transform.SetCheckMeanDistance(True)
        transform.SetSource(mov)
        transform.Update()
        #transform.GetMeanDistance()
        tfm = transform

    if do_apply:
        return tfm, apply_transform(tfm, mov)
    else:
        return tfm


################################################################################
### Flipping mesh
def reverse(pd):
    rev = vtk.vtkReverseSense()
    rev.SetInputData(pd)
    rev.Update()
    return rev.GetOutput()

def flip(pd, c=(0,0,0), ax=(-1,1,1)):
    """ flip mesh along some axes and center c (only cx matters) """

    # setting up the transform (could be factorised if applied t o several subjects), but not a big overhead
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(tuple([-x for x in c]))
    transform.Scale(ax[0], ax[1], ax[2])
    transform.Translate(c)

    pd = apply_transform(transform, pd)

    # reverse normals if odd number of flips
    if ax[0]*ax[1]*ax[2] < 0:
        rev = vtk.vtkReverseSense()
        rev.SetInputData(pd)
        rev.Update()
        return rev.GetOutput(), transform
    else:
        return pd, transform


def flipx(pd, c):
    """ flip mesh along x-axis and center c (only cx matters) """
    print('obsolete (ssm.tools.flipx): use ssm.tools.flip')
    return flip(pd, c, (-1,1,1))[0]
