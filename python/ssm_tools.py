#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Mesh processing tools (vtk) for atlas estimation

- Read/Write polydata
- Rigid/similitud registration (using vtkIterativeClosestPoint, vtkTransform)
- Decimation and clipping
- numpy array to vtk points

"""

import subprocess as sp
import os, sys

import numpy as np
import vtk
import vtk.util.numpy_support




################################################################################
##  numpy, vtk, landmarks

def controlpoints_to_vtkPoints(cp, mt=None):
    """
    convert control points array given by deformetrica to a vtk point cloud
    if momenta are given add vector data to points
    cp (k, 3)
    mt (k, d)
    """
    vldm = vtk.util.numpy_support.numpy_to_vtk(cp)
    points = vtk.vtkPoints()
    points.SetData(vldm)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)

    if mt is not None:
        vmt = vtk.util.numpy_support.numpy_to_vtk(mt)
        if mt.ndim == 1 or mt.shape[1] == 1:
            pd.GetPointData().SetScalars(vmt)
        else:
            pd.GetPointData().SetVectors(vmt)

    return pd

def controlpoints_to_vtkPoints_files(cpt_file, vtk_file, mmt_file=None):
    """ same than controlpoints_to_vtkPoints but with read/write """
    cp = np.loadtxt(cpt_file)
    if mmt_file is not None:
        mt = np.loadtxt(mmt_file)
    else:
        mt = None
    pd = controlpoints_to_vtkPoints(cp, mt)
    WritePolyData(vtk_file, pd)

def load_momenta(fi):
    """ loading momenta from a deformetrica file, 1st line = shape """
    a = np.loadtxt(fi)
    shape = a[0, :].astype("int")
    return  a[1:, :].reshape(shape)


################################################################################
##  IO

def ReadPolyData(file_name):
    """ read a polydata from a .vtk or .stl file """
    if file_name[-4:] == ".stl":
        reader = vtk.vtkSTLReader()
    elif file_name[-4:] == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise ValueError("unknown extension in ReadPolyData: " + file_name)

    if os.path.exists(file_name):
        reader.SetFileName(file_name)
        reader.Update()
        return reader.GetOutput()
    else:
        raise FileNotFoundError("no file: " + file_name)

def WritePolyData(file_name, pd):
    """ write in a .vtk, .stl or .vtp file """
    if file_name[-4:] == ".stl":
        wr = vtk.vtkSTLWriter()
    elif file_name[-4:] == ".vtk":
        wr = vtk.vtkPolyDataWriter()
    elif file_name[-4:] == ".vtp":
        wr = vtk.vtkXMLPolyDataWriter()
    else:
        raise RuntimeError("unknown extension in WritePolyData: " + file_name)

    wr.SetFileName(file_name)
    wr.SetInputData(pd)

    try:
        return wr.Update()
    except FileNotFoundError as e:
        odir = os.path.dirname(file_name)
        if not os.path.exists(odir):
            sp.call(["mkdir", "-p", odir])
            print("Directory created:", odir)
            return wr.Update()
        else:
            raise e

def ConvertPolyData(fi, fo):
    """ read then write """
    WritePolyData(fo, ReadPolyData(fi))


################################################################################
##  Mesh processing and registrations

def DecimatePolyData(pd, reduction=0.9, ntarget=None):
    """
    reduction == 0.9 => réduction to 10% original size
    if ntarget is not None override reduction factor
    vtkDecimatePro only process triangles we use vtkTriangleFilter first
    """
    if ntarget is not None:
        n = pd.GetPoints().GetNumberOfPoints()
        reduction = (n-ntarget)/n
        print("reduction: ", n, reduction)

    triangulate = vtk.vtkTriangleFilter()
    triangulate.SetInputData(pd)
    triangulate.Update()

    #decimate = vtk.vtkDecimatePro()
    decimate = vtk.vtkQuadricDecimation()
    decimate.SetInputData(triangulate.GetOutput())
    decimate.SetTargetReduction(reduction)
    #decimate.PreserveTopologyOn()
    decimate.Update()

    decimated = vtk.vtkPolyData()
    decimated.ShallowCopy(decimate.GetOutput())
    return decimated


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
    # (I could not find how to initialize the deformation so we warp a mesh)
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
##  Mesh distance

def polydata_distance(mflo, mref, do_signed=True):
    """ distance from mflo to mref """
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
