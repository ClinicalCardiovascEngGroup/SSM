#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
SSM atlas estimation

Usage:
    main_ssm [options]

Options:
    -c FILE, --config FILE      Config file
    -i PATH, --idir PATH        Input files prefix (include 'idir*.vtk')
    -o PATH, --odir PATH        Output directory [default: output/]
    -n NAME, --name NAME        Object name [default: geom]
    -t MESH, --template MESH    Initial guess, int or path [default: 0]

    --kwg FLOAT        kernel width of the geometry [default: 10.]
    --kwd FLOAT        kernel width of the diffeomorphisms [default: 20.]
    --noise FLOAT      noise [default: 10.]
"""


import subprocess as sp
import os, sys
import logging

import docopt, configparser

import ssm_pca
import ssm_atlas

################################################################################
## Parameters
def input_parameters():
    """
    parse command line parameter and config
    Warning: config overwrites command line parameters
    """
    params = docopt.docopt(__doc__, help=True, version='0.1')
    if params["--config"]:
        print("Warning: config overwrites command line parameters")
        config = configparser.ConfigParser()
        config.read(params["--config"])
        params["--idir"] = config.get("paths", "idir").strip('"')
        params["--odir"] = config.get("paths", "odir").strip('"')
        params["--name"] = config.get("paths", "name").strip('"')
        params["--template"] = config.get("paths", "initial_guess").strip('"')

        params["--kwg"]   = config.get("parameters", "kernel_width_geometry")
        params["--kwd"]   = config.get("parameters", "kernel_width_deformation")
        params["--noise"] = config.get("parameters", "noise_geometry")
    print(params)
    return params

################################################################################
## Main
if __name__ == "__main__":

    params = input_parameters()

    sp.call(["mkdir", "-p", params["--odir"]])
    ae = ssm_atlas.DeformetricaAtlasEstimation(params)
    ae.estimate()


    ao = ssm_pca.DeformetricaAtlasOutput(
        idir = ae.odir + "output/",
        odir = ae.odir + "pca/")

    ao.kw = ae.p_kernel_width_deformation


    ao.compute_pca(with_plots=True)
    ae.shooting(ao.save_eigv(0), ae.odir + "pca/shoot0/")
    ae.shooting(ao.save_eigv(1), ae.odir + "pca/shoot1/")

    ao.render_momenta_norm(ao.get_eigv(2), name=ae.id)
