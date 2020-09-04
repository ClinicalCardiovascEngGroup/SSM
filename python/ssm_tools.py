#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Atlas estimation using deformetrica

- VTK IterativeClosestPoint registration (similitud)
- Atlas construction (similitud)
- Deformetrica atlas construction (diffeomorphic)


"""

import subprocess as sp
import os, sys

import numpy as np
import vtk
import vtk.util.numpy_support

sys.path.append("/home/face3d/Desktop/segment_faces/script/")
import affine_deformation_landmarks


################################################################################
##  numpy, vtk, landmarks

def controlpoints_to_vtkPoints(cp, mt=None):
    """
    convert control points array given by deformetrica to a vtk point cloud
    if momenta are given add vector data to points
    """
    vldm = vtk.util.numpy_support.numpy_to_vtk(cp)
    points = vtk.vtkPoints()
    points.SetData(vldm)

    pd = vtk.vtkPolyData()
    pd.SetPoints(points)

    if mt is not None:
        vmt = vtk.util.numpy_support.numpy_to_vtk(mt)
        pd.GetPointData().SetVectors(vmt)

    return pd

def controlpoints_to_vtkPoints_files(cpt_file, vtk_file, mmt_file=None):
    """
    same but with read/write
    """
    cp = np.loadtxt(cpt_file)
    if mmt_file is not None:
        mt = np.loadtxt(mmt_file)
    else:
        mt = None
    pd = controlpoints_to_vtkPoints(cp, mt)
    WritePolyData(vtk_file, pd)

################################################################################
##  IO

def ReadPolyData(file_name):
    """ read a polydata from a .vtk or .stl file """
    if file_name[-4:] == ".stl":
        reader = vtk.vtkSTLReader()
    elif file_name[-4:] == ".vtk":
        reader = vtk.vtkPolyDataReader()
    else:
        raise RuntimeError("unknown extension in ReadPolyData: " + file_name)

    reader.SetFileName(file_name)
    reader.Update()
    return reader.GetOutput()

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
    return wr.Update()


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

    decimate = vtk.vtkDecimatePro()
    decimate.SetInputData(triangulate.GetOutput())
    decimate.SetTargetReduction(reduction)
    decimate.PreserveTopologyOff()
    decimate.Update()

    decimated = vtk.vtkPolyData()
    decimated.ShallowCopy(decimate.GetOutput())
    return decimated


def ExtractGeometryZ(pd, nx, ny, nz, z0):
    """
    cut the mesh using z > z0
    could try vtkClipPolyData too
    """
    filter = vtk.vtkExtractGeometry()

    function = vtk.vtkPlane()
    function.SetNormal(nx, ny, nz)
    function.SetOrigin(0, 0, z0)

    triangleFilter = vtk.vtkTriangleFilter()
    triangleFilter.SetInputData(pd)
    triangleFilter.Update()

    filter.SetImplicitFunction(function)
    filter.SetInputData(triangleFilter.GetOutput())
    filter.Update()

    geometryFilter = vtk.vtkGeometryFilter()
    geometryFilter.SetInputData(filter.GetOutput())
    geometryFilter.Update()

    connectFilter = vtk.vtkPolyDataConnectivityFilter()
    connectFilter.SetExtractionModeToLargestRegion()
    connectFilter.SetInputData(geometryFilter.GetOutput())
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
    warper = vtk.vtkTransformPolyDataFilter()
    warper.SetTransform(transform)
    warper.SetInputData(mesh)
    warper.Update()
    return warper.GetOutput()