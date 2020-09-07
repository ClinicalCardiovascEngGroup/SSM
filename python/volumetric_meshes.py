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
    """

    sp.call(["mkdir", "-p", odir])

    # reading unstructured grid
    reader = vtk.vtkUnstructuredGridReader()
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(fin)
    reader.Update()
    vus = reader.GetOutput()
    #print(vus)

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

    # loading deformed points
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(odir + "Shooting__GeodesicFlow__"+name+"__tp_10__age_1.00.vtk")
    reader.Update()
    vpsd = reader.GetOutput()

    # modifing points in original unstructured grid
    vus.SetPoints(vpsd.GetPoints())

    # writing
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetFileName(fout)
    writer.SetInputData(vus)
    writer.Update()
