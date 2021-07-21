#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Linear registration to a reference
Register meshes to a target template using similarity or rigid deformation

Usage:
    main_linear_registration (rigid|similarity|affine) -t <target> -o <odir> <mesh>...
    main_linear_registration -h

Options:
    -t PATH  target template
    -o PATH  output dir (same names as inputs)
    -h       show this screen
"""

import vtk
import sys, os
import docopt
import ssm.tools, ssm.iovtk

################################################################################
## Main
if __name__ == "__main__":

    params = docopt.docopt(__doc__, help=True, version='0.1')
    print(params)

    # outdir
    if not os.path.exists(params["-o"]):
        os.mkdir(params["-o"])

    # fix mesh
    fix = ssm.iovtk.ReadPolyData(params["-t"])

    # transform
    transform = vtk.vtkIterativeClosestPointTransform()
    if params["rigid"]:
        transform.GetLandmarkTransform().SetModeToRigidBody()
    elif params["similarity"]:
        transform.GetLandmarkTransform().SetModeToSimilarity()
    elif params["affine"]:
        transform.GetLandmarkTransform().SetModeToAffine()
    else:
        print("No recognised deformation mode")
        sys.exit(1)

    transform.SetTarget(fix)
    transform.SetStartByMatchingCentroids(True)

    # ICP registration
    for f in params["<mesh>"]:
        print("aligning", f)
        mov = ssm.iovtk.ReadPolyData(f)
        transform.SetSource(mov)
        transform.Update()
        res = ssm.tools.apply_transform(transform, mov)
        ssm.iovtk.WritePolyData(os.path.join(params["-o"], f.split("/")[-1]), res)

    print("Done.")
