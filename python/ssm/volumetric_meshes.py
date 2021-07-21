#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Volumetric meshes (vtkUnstructuredGrid) are not supported by Deformetrica

some operations are nevertheless possible in particular when no metric is involved

We implement here:
- shooting (it could be added more properly to deformetrica)
"""

import subprocess as sp
import vtk
import deformetrica


def shoot_vtu(fin, fvtk, fmoments, fctrlpts, odir,  fout, kw, noise, name):
    """
    Shooting in 3 steps:
    - extract points from vtkUnstructuredGrid in fin
    - shoot using the points
    - set the warped points to the original object

    fin         input file (.vtu)
    fvtk        intermediate file for point set (.vtk)
    fmoments    moments (.txt)
    fctrlpts    control points (.txt)
    odir        output directory
    kw          deformation kernel
    noise       geometric noise (idk what for)
    name        object name (deformetrica)

    the first line of the momenta file should give the number of subjects
    it is possible to do multiple shooting at once.
    """

    sp.call(["mkdir", "-p", odir])

    # reading unstructured grid
    reader = vtk.vtkUnstructuredGridReader()
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(fin)
    reader.Update()
    vus = reader.GetOutput()

    # writing points as polydata
    vps = vtk.vtkPolyData()
    vps.SetPoints(vus.GetPoints())
    print("shooting N points:", vus.GetNumberOfPoints(), vps.GetNumberOfPoints())
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(fvtk)
    writer.SetInputData(vps)
    writer.Update()

    # warping polydata
    template_specifications = {
        name: {'deformable_object_type': 'pointcloud', 'noise_std': noise,
                  'kernel_type':'torch', 'kernel_width':kw,
                  'filename': fvtk}}
    model_options={
                'dimension': 3,
                'deformation_kernel_type': 'torch',
                'deformation_kernel_width': kw,
                'tmin':0,
                'tmax':1,
                "initial_control_points": fctrlpts,
                "initial_momenta": fmoments}
    Deformetrica = deformetrica.api.Deformetrica(verbosity="INFO", output_dir=odir)
    Deformetrica.compute_shooting(template_specifications, model_options=model_options)

    # number of momenta used
    fm = open(fmoments,"r")
    l = fm.readline().strip().split(" ")
    N = int(l[0])

    if N == 1:
        lshotvtk = [odir + "Shooting__GeodesicFlow__"+name+"__tp_10__age_1.00.vtk"]
        lfout = [fout]
    else:
        lshotvtk = [odir +"Shooting_"+str(i)+"__GeodesicFlow__"+name+"__tp_10__age_1.00.vtk" for i in range(N)]
        lfout = [fout[:-4] + "_{}.vtu".format(i) for i in range(N)]

    # using connectivity of 'vus' on shot points
    reader = vtk.vtkPolyDataReader()
    writer = vtk.vtkXMLUnstructuredGridWriter()
    for fv, fo in zip(lshotvtk, lfout):

            # loading deformed points
            reader.SetFileName(fv)
            reader.Update()
            vpsd = reader.GetOutput()

            # modifing points in original unstructured grid
            vus.SetPoints(vpsd.GetPoints())

            # writing
            writer.SetFileName(fo)
            writer.SetInputData(vus)
            writer.Update()
