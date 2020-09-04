#!/usr/bin/env python
# -*- coding: utf-8 -*


import glob, re
import subprocess as sp
import vtk

################################################################################
##  Files and IO

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


################################################################################
##  vtk 3D rendering

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
