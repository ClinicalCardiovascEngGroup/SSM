#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Atlas estimation using deformetrica

deformetrica estimate model.xml data-set.xml -p optimization_parameters.xml

"""


import subprocess as sp
import os, sys
import glob


# to enable built in directory tab completion with raw_input():
import readline
readline.parse_and_bind("tab: complete")


import numpy as np
import scipy, scipy.linalg
import matplotlib
import matplotlib.pyplot as plt

import vtk
from vtk.util import numpy_support as nps
import torch


import deformetrica
# modules api, core, in_out and support are part of deformetrica \o/
import support.kernels as kernel_factory


import create_data_set_xml



import logging
logger = logging.getLogger("ssm_atlas")
logger.setLevel(logging.INFO)

# deformetrica messing with the verbosity level...
logging.getLogger('matplotlib.font_manager').disabled = True


################################################################################
##  Deformetrica estimation class
class DeformetricaAtlasEstimation():
    """ result of an DeterministicAtlas estimation """

    def __init__(self, idir="./", odir="output/", name="obj", initial_guess="", kwd=20., kwg=10., noise=10.):
        self.idir = idir
        self.odir = odir
        self.id = name

        self.initial_guess = initial_guess
        self.p_kernel_width_deformation = kwd
        self.p_kernel_width_geometry = kwg
        self.p_noise = noise

        self.dataset_xml = ""

        self.optimization_parameters_xml = os.path.join(os.path.dirname(__file__), "ressources/optimization_parameters.xml")
        self.model_xml = os.path.join(os.path.dirname(__file__), "ressources/model-atlas-template.xml")

    def __get_list_vtk(self, sorted=False):
        """ list of vtk using prefix in self.idir """
        lf = []
        if os.path.isdir(self.idir):
            lf = glob.glob(os.path.join(self.idir, "*.vtk"))
        else:
            lf = glob.glob(self.idir + "*.vtk")

        if sorted:
            lf.sort()

        return lf

    def check_initialisation(self):
        """ check that the input paths exist """

        def __get_input_path(s, p, pass_check=False):
            """ ask for a path using string s and default value p """
            while True:
                x = input(s + " [{}]: ".format(p))
                if x == "":
                    x = p
                if pass_check or os.path.exists(x):
                    return x
                else:
                    print("path {} does not exist".format(x))

        def __get_input_float(s, p):
            while True:
                try:
                    x = input(s + " [{}]: ".format(p))
                    if x == "":
                        x = p
                    return float(x)
                except ValueError:
                    print("not a float")

        self.idir = __get_input_path("Set input directory", self.idir, True)
        print(len(self.__get_list_vtk()), "vtk-files in input directory")

        self.odir = __get_input_path("Set output directory", self.odir)
        #sp.call(["mkdir", "-p", self.odir])
        #print("output directory created: ", self.odir)

        try:
            k = int(self.initial_guess)
            self.initial_guess = self.get_path_initial_guess(k)
        except ValueError:
            pass

        self.initial_guess = __get_input_path("Set initial mesh", self.initial_guess)
        self.optimization_parameters_xml = __get_input_path("Set optimization xml",self.optimization_parameters_xml)
        self.model_xml = __get_input_path("Set model xml",self.model_xml)

        self.p_kernel_width_geometry = __get_input_float('Set kernel width geometry: ', self.p_kernel_width_geometry)
        self.p_kernel_width_deformation =  __get_input_float('Set kernel width deformation: ', self.p_kernel_width_deformation)
        self.p_noise =  __get_input_float('Set noise level: ', self.p_noise)

    def get_path_initial_guess(self, k):
        """ path of data of subject k """
        lf = self.__get_list_vtk(sorted=True)
        return lf[k]

    def create_dataset_xml(self):
        """ with every vtk files in idir """
        lf = self.__get_list_vtk(sorted=True)
        fxml = os.path.join(self.odir, "dataset.xml")

        create_data_set_xml.create_xml_atlas(lf, fxml, self.id)
        self.dataset_xml = fxml

    def estimate(self):
        """ estimate atlas """
        # check and ask for good initialization

        self.create_dataset_xml()

        # General parameters
        xml_parameters = deformetrica.XmlParameters()
        xml_parameters._read_optimization_parameters_xml(self.optimization_parameters_xml)
        xml_parameters._read_model_xml(self.model_xml)
        xml_parameters._read_dataset_xml(self.dataset_xml)

        # Overwriting main parameters
        xml_parameters.deformation_kernel_width = self.p_kernel_width_deformation

        # Template
        template_object = xml_parameters._initialize_template_object_xml_parameters()
        template_object['deformable_object_type'] = "surfacemesh"
        template_object['attachment_type'] = "current"
        template_object['kernel_type'] = 'torch'
        template_object ['kernel_device'] = 'gpu'

        template_object['noise_std'] =  self.p_noise
        template_object['filename'] = self.initial_guess
        template_object['kernel_width'] = self.p_kernel_width_geometry
        xml_parameters.template_specifications[self.id] = template_object


        ## Estimation
        # call: deformetrica estimate model.xml dataset.xml -p optimization_parameters.xml
        odir = os.path.join(self.odir, "output/")
        Deformetrica = deformetrica.api.Deformetrica(output_dir=odir, verbosity="DEBUG")
        Deformetrica.estimate_deterministic_atlas(
            xml_parameters.template_specifications,
            deformetrica.get_dataset_specifications(xml_parameters),
            estimator_options=deformetrica.get_estimator_options(xml_parameters),
            model_options=deformetrica.get_model_options(xml_parameters))


    def shooting(self, fv, odir):
        """
        warp Atlas using momenta in fv
        interesting momenta are in DeterministicAtlas__EstimatedParameters__Momenta.txt
        or output from pca using save_eigv
        """

        # General parameters
        xml_parameters = deformetrica.XmlParameters()
        xml_parameters._read_optimization_parameters_xml(self.optimization_parameters_xml)
        xml_parameters._read_model_xml(self.model_xml)

        xml_parameters.model_type = "shooting"
        xml_parameters.initial_control_points = os.path.normpath(
            os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt"))
        xml_parameters.initial_momenta = fv

        # Template
        template_object = xml_parameters._initialize_template_object_xml_parameters()
        template_object['deformable_object_type'] = "surfacemesh"
        template_object['attachment_type'] = "current"

        template_object['filename'] = os.path.normpath(
            os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__Template_"+self.id+".vtk"))
        xml_parameters.template_specifications[self.id] = template_object

        # Deformation parameters
        xml_parameters.deformation_kernel_width = self.p_kernel_width_deformation
        xml_parameters.t0 = 0.
        xml_parameters.tmin = -5.
        xml_parameters.tmax = +5.

        ## Shooting
        # call: deformetrica compute model.xml -p optimization_parameters.xml
        Deformetrica = deformetrica.api.Deformetrica(output_dir=odir, verbosity="INFO")

        Deformetrica.compute_shooting(xml_parameters.template_specifications,
                model_options=deformetrica.get_model_options(xml_parameters))

        from ssm_pca import rename_df2pv
        rename_df2pv(odir + "Shooting__GeodesicFlow__" + self.id)


    def registration(self, fv, odir, subject_id="subj"):
        """
        register a (new) subject to the template
        do not require: idir, initial_guess
        """

        # General parameters
        xml_parameters = deformetrica.XmlParameters()
        xml_parameters._read_optimization_parameters_xml(self.optimization_parameters_xml)
        xml_parameters._read_model_xml(self.model_xml)

        xml_parameters.model_type = "registration"
        xml_parameters.initial_control_points = os.path.normpath(
        os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt"))
        xml_parameters.freeze_control_points = True

        # Template
        template_object = xml_parameters._initialize_template_object_xml_parameters()
        template_object['deformable_object_type'] = "surfacemesh"
        template_object['attachment_type'] = "current"
        template_object['kernel_type'] = 'torch'
        template_object ['kernel_device'] = 'gpu'
        template_object['noise_std'] =  self.p_noise
        template_object['kernel_width'] = self.p_kernel_width_geometry

        template_object['filename'] = os.path.normpath(
            os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__Template_"+self.id+".vtk"))
        xml_parameters.template_specifications[self.id] = template_object

        # Deformation parameters
        xml_parameters.deformation_kernel_width = self.p_kernel_width_deformation

        # Dataset
        dataset_specifications = {}
        dataset_specifications['visit_ages'] = [[]]
        dataset_specifications['dataset_filenames'] = [[{self.id:fv}]]
        dataset_specifications['subject_ids'] = [subject_id]

        ## Estimation
        Deformetrica = deformetrica.api.Deformetrica(output_dir=odir, verbosity="DEBUG")

        Deformetrica.estimate_registration(
            xml_parameters.template_specifications,
            dataset_specifications,
            estimator_options=deformetrica.get_estimator_options(xml_parameters),
            model_options=deformetrica.get_model_options(xml_parameters))


    def convolve_momentum(self, m, x):
        """
        kernel convolution of momenta at points x
        m : np.array K, 3,  K=number of controls points,
                            ex: m=self.momenta[0,:], m=get_eigv(0)
        kw: float,          kernel width
        x : np.array N, 3,  coordinates
        """

        kern = kernel_factory.factory("torch", gpu_mode=True, kernel_width=self.kwd)

        a_cp = np.loadtxt(self.odir + "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")
        assert a_cp.shape == m.shape

        t_y = torch.tensor(a_cp, device="cpu")
        t_x = torch.tensor(x, device="cpu")
        t_p = torch.tensor(m, device="cpu")

        t_z = kern.convolve(t_x, t_y, t_p)
        return np.array(t_z)

    def read_template(self):
        """ return polydata of the template """
        ft = self.odir + "output/DeterministicAtlas__EstimatedParameters__Template_{}.vtk".format(self.id)
        ft = os.path.abspath(ft).encode()

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(ft)
        reader.Update()
        v_pd = reader.GetOutput()
        return v_pd

    def save_controlpoints_vtk(self):
        """ save controlpoints (could add some momenta) as vtk point cloud """
        self.ctrlpts = np.loadtxt(self.odir + "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")
        vtkp = ssm_tools.controlpoints_to_vtkPoints(self.ctrlpts)
        fv = os.path.normpath(os.path.join(self.odir, "controlpoints.vtk"))
        ssm_tools.WritePolyData(fv, vtkp)

    def render_momenta_norm(self, moments):
        """ render the norm of the momenta on the template geometry """

        # read mesh
        v_pd = self.read_template()
        N = v_pd.GetPoints().GetNumberOfPoints()
        points = nps.vtk_to_numpy(v_pd.GetPoints().GetData()).astype('float64')

        # compute attributes
        z = self.convolve_momentum(moments, points)
        print(z.shape)
        d = np.sum(z**2, axis=1)
        assert d.size == N

        # set attributes
        scalars = vtk.vtkFloatArray()
        for i in range(N):
            scalars.InsertTuple1(i, d[i])
        v_pd.GetPointData().SetScalars(scalars)

        # render
        import visualization_tools
        visualization_tools.renderVtkPolyData(v_pd, vmin=0., vmax=d.max())
