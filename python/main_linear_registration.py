#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Linear registration to a reference
Register meshes to a target template using similarity or rigid deformation

Usage:
    main_linear_registrion (rigid|similarity|affine) -t <target> -o <odir> <mesh>...
    main_linear_registrion -h

Options:
    -t PATH  target template
    -o PATH  output dir (same names as inputs)
    -h       show this screen
"""

import vtk
import sys
import docopt
import ssm_tools

################################################################################
## Main
if __name__ == "__main__":

    params = docopt.docopt(__doc__, help=True, version='0.1')
    print(params)

    # fix mesh
    fix = ssm_tools.ReadPolyData(params["-t"])

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
        mov = ssm_tools.ReadPolyData(f)
        transform.SetSource(mov)
        transform.Update()
        res = ssm_tools.apply_transform(transform, mov)
        ssm_tools.WritePolyData(params["-o"] + f.split("/")[-1], res)

    print("Done.")
