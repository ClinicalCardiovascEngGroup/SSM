#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" rename for paraview """

import os, sys, glob
import re
import subprocess as sp



# rename for paraview
def key_tp(f):
    m = re.search("tp_(\d+)__age", f)
    return int(m.group(1))


def rename_df2pv(odir, v):

    fl = glob.glob(odir + "*__"+ v + "__face__tp_*__age_*.vtk")
    fl.sort(key=key_tp)
    print(len(fl))

    for i,f in enumerate(fl):
        sp.call(["mv", f, odir + v + "_tp{:03}.vtk".format(i)])


def momenta_df2vtk(odir):
    """ transform the momenta/control points output to a vtkPolyData file """

    import numpy as np
    import vtk
    import vtk.util.numpy_support

    # reading deformetrica files
    fcp = glob.glob(odir + "*__EstimatedParameters__ControlPoints.txt")
    if len(fcp) != 1:
        print("err looking for control points", fcp)
        return
    cp = np.loadtxt(fcp[0])

    fcp = glob.glob(odir + "*__EstimatedParameters__Momenta.txt")
    if len(fcp) != 1:
        print("err looking for momenta", fcp)
        return
    mt = np.loadtxt(fcp[0])
    mt = mt[1:, :]

    # vtk objects
    vcp = vtk.util.numpy_support.numpy_to_vtk(cp)
    vmt = vtk.util.numpy_support.numpy_to_vtk(mt)

    points = vtk.vtkPoints()
    points.SetData(vcp)
    pd = vtk.vtkPolyData()
    pd.SetPoints(points)
    pd.GetPointData().SetVectors(vmt)

    # vtk writing
    wr = vtk.vtkPolyDataWriter()
    wr.SetFileName(odir + "controls_points.vtk")
    wr.SetInputData(pd)
    wr.Update()

if __name__ == "__main__":
    odir = sys.argv[1]
    rename_df2pv(odir, "GeodesicFlow")
    rename_df2pv(odir, "Reconstruction")

    momenta_df2vtk(odir)
