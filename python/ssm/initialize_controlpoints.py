#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Initialize control points for a list of vtk
dim = 3 only
"""

import numpy as np
import vtk
from vtk.util import numpy_support as nps

from . import iovtk


from deformetrico.core.models.model_functions import create_regular_grid_of_points
from deformetrico.in_out.array_readers_and_writers import write_2D_array


def get_mesh_bounding_box(pd):
    """ 3D bounding box of polydata pd, shape (D, 2) """
    apts = nps.vtk_to_numpy(pd.GetPoints().GetData())
    return np.stack((apts.min(axis=0), apts.max(axis=0)), axis=1)


def initialize_controlpoints(lvtk, spacing, margin, output_dir, name):
    """ create control points that include all meshes """
    b = np.array([get_mesh_bounding_box(iovtk.ReadPolyData(f)) for f in lvtk])
    box = np.zeros((3,2))
    box[:, 0] = np.min(b, axis=0)[:, 0] - margin
    box[:, 1] = np.max(b, axis=0)[:, 1] + margin

    p = create_regular_grid_of_points(box, spacing, dimension=3)
    write_2D_array(p, output_dir, name, fmt='%f')
    return box, p
