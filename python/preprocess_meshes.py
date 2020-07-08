#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Mesh processing before atlas estimation

- Alignment using landmarks
- Cutting
- Resampling


"""

import subprocess as sp
import os, sys, glob

import numpy as np
import vtk
import vtk.util.numpy_support
import ssm_tools



################################################################################
## File list
def get_mesh_list(list_file, imgdir, outdir):
    lsbj = []
    with open(list_file, "r") as fids:
        for line in fids:
            try:
                id = line.strip()
                print(id)
                msh = meshdir + "s{0}.stl".format(id)

                l = id.split("-")
                k = l[0].lstrip("0")
                if len(l) == 2:
                    t = "_" + l[1]
                else:
                    t = "_0"

                limg = glob.glob(imgdir + "Case_" + k + "_*" + "/resampled_c*" + k + t + ".nii.gz")
                if len(limg) == 1:
                    img = limg[0]
                else:
                    print("not found image:", imgdir, k, t, limg)
                    break

                limg = glob.glob(imgdir + "Case_" + k + "_*" + "/landmarks_c*" + k + t + ".txt")
                if len(limg) == 1:
                    ldm = limg[0]
                else:
                    print("not found landmarks:", imgdir, k, t, limg)
                    break


                lsbj.append({"mesh": msh,
                        "ldm": ldm,
                        "img": img,
                        "out": outdir + "mesh_{}.vtk".format(id),
                        "cid": k,
                        "aid": t[1],
                        })
            except ValueError as e:
                print(e, line)
    return lsbj



################################################################################
## Processing

# Template mesh
def init_template(ref_mesh, ref_ldm, Ntarget=None, odir=""):
    """
    read ref mesh and ref landmarks
    if Ntarget is not None create an initial template by processing ref mesh
    """
    m0f = ssm_tools.ReadPolyData(ref_mesh)
    ldm0 = np.loadtxt(ref_ldm)
    vldm = vtk.util.numpy_support.numpy_to_vtk(ldm0)
    p0 = vtk.vtkPoints()
    p0.SetData(vldm)

    if Ntarget is None:
        return m0f, p0, ""
    else:
        # previous landmarks:  nx=0.05, ny=-0.15, nz=1., z0=50
        m0c = ssm_tools.ExtractGeometryZ(m0f, nx=0., ny=-0.1, nz=1., z0=37)
        m0d = ssm_tools.DecimatePolyData(m0c, ntarget=Ntarget)
        ref_out = odir + "template_{}.vtk".format(m0d.GetPoints().GetNumberOfPoints())
        ssm_tools.WritePolyDataVTK(ref_out, m0d)
        return m0f, p0, ref_out

# Subject meshes
def init_subject(s, m0f, p0, Ntarget, do_rigid):
    """
    registration, decimation, cutting
    s        dictionary     file names: 'mesh', 'ldm', 'img','out'
    m0f      vtkPolyData    reference mesh
    p0       vtkPoints      reference landmarks
    Ntarget  int            number of points in the output mesh
    do_rigid bool           rigid registration only (else similitud)

    """
    if os.path.exists(s["out"]):
        return s["out"]

    m1f = ssm_tools.ReadPolyData(s["mesh"])
    p1 = ssm_tools.read_landmarks_as_vtkPoints(s["ldm"], s["img"])

    transform = ssm_tools.ICPSimilitudRegistration(m0f, m1f, p0, p1, do_rigid=do_rigid)
    m1r = ssm_tools.apply_transform(transform, m1f)


    # previous landmarks:  nx=0.05, ny=-0.15, nz=1., z0=50
    m1c = ssm_tools.ExtractGeometryZ(m1r, nx=0., ny=-0.1, nz=1., z0=37)
    m1d = ssm_tools.DecimatePolyData(m1c, ntarget=Ntarget)
    print(m1f.GetPoints().GetNumberOfPoints(),
          m1c.GetPoints().GetNumberOfPoints(),
          m1d.GetPoints().GetNumberOfPoints())
    ssm_tools.WritePolyData(s["out"], m1d)

    return s["out"]



################################################################################
## Parameters
Ntarget = 2000

ref_dir = "/home/face3d/Desktop/ssm/guimond_atlas_affine/atlas_adni_analysis/run4/"
ref_mesh = ref_dir + "Atlas_4.stl"
ref_ldm =  ref_dir + "ref_landmarks_wunit.txt"
ref_img =  ""

imgdir  = "/home/face3d/Desktop/faces/0_segmentation_results/"
meshdir = "/home/face3d/Desktop/faces/1_meshes/"
outdir  = "/home/face3d/Desktop/ssm/atlas_{}/data/".format(Ntarget)

list_file = meshdir + "../list_ids_uniques.txt"

################################################################################
## Main

if __name__ == "__main__":

    m0f, p0, _ = init_template(ref_mesh, ref_ldm, None)

    flc = open("/home/face3d/Desktop/faces/list_june_done.txt", "r")
    k = 0
    for c in flc:
        c = c.strip()
        print(c)

        s = {}
        idir = "/home/face3d/Desktop/faces/1_segmentation_june/" + c
        s['mesh']= idir + "/mesh_" + c + ".vtk"
        s['ldm'] = idir + "/landmarks_" + c + ".txt"
        s['img'] = idir + "/resampled_" + c + ".nii.gz"
        s['out'] = "/home/face3d/Desktop/faces/3_ssm_june/m_{:02}.vtk".format(k)
        k += 1

        m = init_subject(s, m0f, p0, Ntarget, do_rigid=True)



    """
    lsbj = get_mesh_list(list_file, imgdir, outdir)

    # add age
    import pandas as pd
    df = pd.read_csv("/home/face3d/Desktop/faces/data.csv")
    for x in lsbj:
        cid = x["cid"]
        aid = x["aid"]
        res = df.query("cid == @cid & acqid == @aid")
        days = 7*res["week"].iloc[0] + res["day"].iloc[0]
        x["age"] = days

    # process meshes
    sp.call(["mkdir", "-p", outdir])
    lmeshes = []
    lages = []
    m0f, p0 = init_template(outdir, ref_mesh, ref_ldm, Ntarget)
    for s in lsbj:
        m = init_subject(s, m0f, p0, Ntarget, do_rigid=True)
        lmeshes.append(m)

    # create xml
    import create_data_set_xml
    import glob
    #lfiles = glob.glob(outdir + "mesh_*.vtk")
    foxml = outdir + "../data-set-reg-{}.xml".format(len(lmeshes))
    create_data_set_xml.create_xml_regression(lmeshes, lsbj, foxml)

    foxml = outdir + "../data-set-atl-{}.xml".format(len(lmeshes))
    create_data_set_xml.create_xml_atlas(lmeshes, foxml)
    """
