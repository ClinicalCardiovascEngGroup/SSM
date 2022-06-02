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


def read_loglikelihood_from_log(flog, with_it=False):
    """ likehood from deformetrica log file """
    ll = []
    it = 0
    with open(flog, "r") as fl:
        for l in fl:
            m = re.search("Iteration: (\d+)", l)
            if m:
                it = int(m.group(1))
            else:
                m = re.search("Log-likelihood = ([+-.E\d]+)", l)
                if m:
                    try:
                        x = float(m.group(1))
                        if with_it:
                            ll.append((it, x))
                        else:
                            ll.append(x)
                    except ValueError:
                        print(m)
    return ll

def plot_loglikelihood(ll):
    import matplotlib.pyplot as plt

    ll = np.array(ll)

    plt.plot(ll[:, 0], -ll[:, 1], ".");
    plt.semilogy(base=10)
    plt.xlabel("iteration")
    plt.ylabel("- log-likelihood")
    plt.grid(axis="y", which="major")

    max_llh = ll[:, 1].max()
    #lll = np.log(-1 * ll[:, 1]) # 1 - exp(-x) ~ x

    print("max llh = {:.1f}".format(max_llh))
    print("last increment = {:.4f}".format((ll[-2,1] - ll[-1,1])/ll[-1,1]))
    return

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
