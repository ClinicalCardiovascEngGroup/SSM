#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Mesh IO tools (vtk) for atlas estimation

- Read/Write controlpoints and momenta as vtkPoints
- Read/Write polydata
"""

import os

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
            os.mkdir(odir)
            print("Directory created:", odir)
            return wr.Update()
        else:
            raise e

def ConvertPolyData(fi, fo):
    """ read then write """
    WritePolyData(fo, ReadPolyData(fi))
