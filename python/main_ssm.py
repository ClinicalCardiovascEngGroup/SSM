#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
SSM atlas estimation

Usage:
    main_ssm [options]

Options:
    -c FILE, --config FILE      Config file
    -i PATH, --idir PATH        Input files prefix (use quotes to include 'idir*.vtk')
    -o PATH, --odir PATH        Output directory [default: output/]
    -n NAME, --name NAME        Object name [default: geom]
    -t MESH, --template MESH    Initial guess, int or path [default: 0]

    --kwg FLOAT         kernel width of the geometry [default: 10.]
    --kwd FLOAT         kernel width of the diffeomorphisms [default: 20.]
    --noise FLOAT       noise [default: 10.]

    --show-pw           try to open paraview to show the results
    --no-check          skip initialisation check
    --do-pca            compute PCA modes and shoot on 0 and 1.
"""


import subprocess as sp
import os, sys
import logging

import docopt, configparser

import ssm.pca
import ssm.atlas

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
    ae = ssm.atlas.DeformetricaAtlasEstimation(
        idir=params["--idir"],
        odir=params["--odir"],
        name=params["--name"],
        initial_guess=params["--template"],
        kwd=params["--kwd"],
        kwg=params["--kwg"],
        noise=params["--noise"])

    # Atlas estimation
    ae.check_initialisation(do_quick=params["--no-check"])
    ae.estimate()
    ae.save_parameters()

    # Results
    if params["--show-pw"]:
        try:
            sp.call(["paraview", "--data=" + ae.odir + "/output/DeterministicAtlas__EstimatedParameters__Template_" + ae.id + ".vtk"])
        except FileNotFoundError:
            pass

    # PCA
    if params["--do-pca"]:
        ao = ssm.pca.DeformetricaAtlasPCA(
            idir = ae.odir + "output/",
            odir = ae.odir + "pca/")
        ao.compute_pca(with_plots=True)

        f0 = ao.save_eigv(0, with_controlpoints=True)
        f1 = ao.save_eigv(1, with_controlpoints=True)

        ae.shooting(f0 + ".txt", ae.odir + "pca/shoot0/")
        ae.shooting(f1 + ".txt", ae.odir + "pca/shoot1/")

        #ae.render_momenta_norm(ao.get_eigv(0))
        if params["--show-pw"]:
            try:
                sp.call(["paraview", "--data=" + ae.odir + "pca/shoot0/Shooting__GeodesicFlow__" + ae.id + "_tp..vtk"])
            except FileNotFoundError:
                pass



    # Registering a new subject
    #fmesh = ae.get_path_data(0)
    #ae.registration(fmesh, ae.odir + "registration/", subject_id="subj")
    #fmomenta = ae.odir + "registration/DeterministicAtlas__EstimatedParameters__Momenta.txt"
    #fig = ao.project_subject_on_pca(fmomenta)

    print("Done.")
