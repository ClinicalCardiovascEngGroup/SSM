#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Atlas estimation using deformetrica

deformetrica estimate model.xml data-set.xml -p optimization_parameters.xml

"""


import subprocess as sp
import os, sys
import glob
import json

# to enable built in directory tab completion with raw_input():
import readline
readline.parse_and_bind("tab: complete")


import numpy as np
import scipy, scipy.linalg
import vtk
from vtk.util import numpy_support as nps
import torch


import deformetrica
# modules api, core, in_out and support are part of deformetrica \o/
import support.kernels as kernel_factory


import create_data_set_xml
import visualization_tools


import logging
logger = logging.getLogger("ssm_atlas")
logger.setLevel(logging.INFO)

################################################################################
##  Deformetrica estimation class
class DeformetricaAtlasEstimation():
    """ result of an DeterministicAtlas estimation """

    def __init__(self, idir="./", odir="output/", name="obj", initial_guess="", kwd=20., kwg=10., noise=10.):
        self.idir = idir
        self.lf = None
        self.odir = odir
        self.id = name

        if isinstance(initial_guess, int):
            self.initial_guess = self.get_path_data(initial_guess)
        else:
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

    def save_parameters(self):
        """ printing parameters in a json for future use if need be """
        d = {"idir":self.idir,
            "odir":self.odir,
            "name":self.id,
            "init":self.initial_guess,
            "kwd":self.p_kernel_width_deformation,
            "kwg":self.p_kernel_width_geometry,
            "noise":self.p_noise
            }
        with open(self.odir + "params.json", "w") as fd:
            json.dump(d, fd, indent=2)

    def load_parameters(self, fjson, do_load_lf=False):
        """loading parameters from a json"""
        with open(fjson, "r") as fd:
            d = json.load(fd)
            print("loading parameters: ", d)
            self.idir = d["idir"]
            self.odir = d["odir"]
            self.id = d["name"]
            self.initial_guess = d["init"]
            self.p_kernel_width_deformation = d["kwd"]
            self.p_kernel_width_geometry = d["kwg"]
            self.p_noise = d["noise"]
        if do_load_lf:
            self.lf = self.__get_list_vtk(sorted=True)


    def check_initialisation(self):
        """ check that the input paths exist """

        def __get_input_path(s, p, pass_check=False):
            """ ask for a path using string s and default value p """
            while True:
                x = input(s + " [{}]: ".format(p))
                if x == "":
                    x = p
                x = os.path.abspath(x)
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
            self.initial_guess = self.get_path_data(k)
        except ValueError:
            pass

        self.initial_guess = __get_input_path("Set initial mesh", self.initial_guess)
        self.optimization_parameters_xml = __get_input_path("Set optimization xml",self.optimization_parameters_xml)
        self.model_xml = __get_input_path("Set model xml",self.model_xml)

        self.p_kernel_width_geometry = __get_input_float('Set kernel width geometry: ', self.p_kernel_width_geometry)
        self.p_kernel_width_deformation =  __get_input_float('Set kernel width deformation: ', self.p_kernel_width_deformation)
        self.p_noise =  __get_input_float('Set noise level: ', self.p_noise)

    def get_path_data(self, k):
        """ path of data of subject k """
        if self.lf is None:
            self.lf = self.__get_list_vtk(sorted=True)
        return self.lf[k]

    def create_dataset_xml(self):
        """ with every vtk files in idir """
        if self.lf is None:
            self.lf = self.__get_list_vtk(sorted=True)
        self.dataset_xml = os.path.join(self.odir, "dataset.xml")
        create_data_set_xml.create_xml_atlas(self.lf, self.dataset_xml, self.id)

    def estimate(self, xml_params=None, do_keep_all=False):
        """
        estimate atlas
        xml_params is a dictionary used to overwrite xml_arameters attributes
        ex:
        xml_params = {"memory_length":1, "initial_step_size":1e-4, "freeze_control_points":False, "freeze_control_points":False}
        ==>
        xml_parameters.memory_length = 1
        xml_parameters.initial_step_size = 1e-4
        xml_parameters.freeze_template = True
        xml_parameters.freeze_control_points = False
         """

        sp.call(["mkdir", "-p", self.odir])
        self.create_dataset_xml()

        # General parameters
        xml_parameters = deformetrica.XmlParameters()
        xml_parameters._read_optimization_parameters_xml(self.optimization_parameters_xml)
        xml_parameters._read_model_xml(self.model_xml)
        xml_parameters._read_dataset_xml(self.dataset_xml)
        xml_parameters.deformation_kernel_width = self.p_kernel_width_deformation

        # Overwriting main parameters
        if xml_params is None:
            xml_params = dict()
        for x in xml_params:
            try:
                print("overwriting parameters: ", x, "from: ", getattr(xml_parameters, x), "to: ", xml_params[x])
                setattr(xml_parameters, x, xml_params[x])
            except AttributeError as e:
                print(e)
                print("no estimation done")
                return


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

        ## cleaning a bit the deformetrica output
        if not do_keep_all:
            sp.call("rm "+odir+"DeterministicAtlas__flow__*.vtk", shell=True)


    def shooting(self, fv, odir, tmin=-5, tmax=+5, fmesh=None, do_keep_all=False):
        """
        warp Atlas using momenta in fv
        interesting momenta are in DeterministicAtlas__EstimatedParameters__Momenta.txt
        or given by pca using save_eigv
        arguments:
            fv      momenta file
            odir    output directory
            tmin, tmax shooting parameters
            fmesh   mesh to deform (default=Template)
            do_keep_all do keep all the result files
        return:
            None
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

        if fmesh is None:
            fmesh = os.path.normpath(os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__Template_"+self.id+".vtk"))
        template_object['filename'] = fmesh
        xml_parameters.template_specifications[self.id] = template_object

        # Deformation parameters
        xml_parameters.deformation_kernel_width = self.p_kernel_width_deformation
        xml_parameters.t0 = 0.
        xml_parameters.tmin = tmin
        xml_parameters.tmax = tmax

        ## Shooting
        # call: deformetrica compute model.xml -p optimization_parameters.xml
        Deformetrica = deformetrica.api.Deformetrica(output_dir=odir, verbosity="INFO")

        Deformetrica.compute_shooting(xml_parameters.template_specifications,
                model_options=deformetrica.get_model_options(xml_parameters))

        visualization_tools.rename_df2pv(odir + "Shooting__GeodesicFlow__" + self.id)

        if not do_keep_all:
            sp.call("rm "+odir+"Shooting__GeodesicFlow__ControlPoints__tp*.txt", shell=True)
            sp.call("rm "+odir+"Shooting__GeodesicFlow__Momenta__tp*.txt", shell=True)

    def registration(self, fmesh, odir, subject_id="subj"):
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
        dataset_specifications['dataset_filenames'] = [[{self.id:fmesh}]]
        dataset_specifications['subject_ids'] = [subject_id]

        ## Estimation
        Deformetrica = deformetrica.api.Deformetrica(output_dir=odir, verbosity="DEBUG")

        Deformetrica.estimate_registration(
            xml_parameters.template_specifications,
            dataset_specifications,
            estimator_options=deformetrica.get_estimator_options(xml_parameters),
            model_options=deformetrica.get_model_options(xml_parameters))


    def momenta_from_sbj_to_atlas(self, sbj, odir, do_warpback=False):
        """
        forward-backward shooting to invert the atlas-subject deformation
        sbj         subject id (int)

        return file_moment, file_ctrlpts
        """
        sp.call(["mkdir", "-p", odir])
        ipfx = self.odir + "output/DeterministicAtlas__EstimatedParameters__"

        # moment
        m = self.read_momenta()
        np.savetxt(odir + "forward_momenta.txt", m[sbj, :, :])

        # forward to get end momenta
        template_specifications = {
            self.id: {'deformable_object_type': 'surfacemesh',
            'noise_std': self.p_noise,
            'filename': ipfx + "Template_" + self.id +".vtk"}
        }
        model_options={
                    'dimension': 3,
                    'deformation_kernel_type': 'torch',
                    'deformation_kernel_width': self.p_kernel_width_deformation,
                    'tmin':0,
                    'tmax':1,
                    "initial_control_points": ipfx + "ControlPoints.txt",
                    "initial_momenta": odir + "forward_momenta.txt"}

        Deformetrica = deformetrica.api.Deformetrica(verbosity="INFO", output_dir=odir + "forward/")
        Deformetrica.compute_shooting(template_specifications, model_options=model_options)

        # backward setting (shoot similarly)
        m = np.loadtxt(odir + "forward/Shooting__GeodesicFlow__Momenta__tp_10__age_1.00.txt")
        np.savetxt(odir + "backward_momenta.txt", -m)
        sp.call(["cp", odir + "forward/Shooting__GeodesicFlow__ControlPoints__tp_10__age_1.00.txt", odir + "backward_ctrlpts.txt"])

        # backward shooting (could be manually applied to other meshes!)
        if do_warpback:
            template_specifications = {
                self.id: {'deformable_object_type': 'surfacemesh',
                'noise_std': self.p_noise,
                'filename': self.get_path_data(sbj)}
            }
            model_options={
                'dimension': 3,
                'deformation_kernel_type': 'torch',
                'deformation_kernel_width': self.p_kernel_width_deformation,
                'tmin':0,
                'tmax':1,
                "initial_control_points": odir +  "backward_ctrlpts.txt",
                "initial_momenta": odir + "backward_momenta.txt"}

            Deformetrica = deformetrica.api.Deformetrica(verbosity="INFO", output_dir=odir + "backward/")
            Deformetrica.compute_shooting(template_specifications, model_options=model_options)

        return odir + "backward_momenta.txt", odir +  "backward_ctrlpts.txt"


    def convolve_momentum(self, m, x):
        """
        kernel convolution of momenta at points x
        m : np.array K, 3,  K=number of controls points,
                            ex: m=self.momenta[0,:], m=get_eigv(0)
        kw: float,          kernel width
        x : np.array N, 3,  coordinates
        return:
        z : np.array N, 3
        """

        kern = kernel_factory.factory("torch", gpu_mode=True, kernel_width=self.p_kernel_width_deformation)

        a_cp = self.read_ctrlpoints()
        assert a_cp.shape[0] == m.shape[0]

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

    def read_momenta(self):
        """ read the momenta file, first line contain the shape """
        a_momenta = np.loadtxt(self.odir + "output/DeterministicAtlas__EstimatedParameters__Momenta.txt")
        shape = a_momenta[0,:].astype("int")
        return a_momenta[1:, :].reshape(shape)

    def read_ctrlpoints(self):
        return np.loadtxt(self.odir + "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")

    def save_controlpoints_vtk(self, fname="controlpoints.vtk", X=None):
        """ save controlpoints (could add some momenta X (n, d)) as vtk point cloud """
        import ssm_tools
        ctrlpts = np.loadtxt(self.odir + "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")
        vtkp = ssm_tools.controlpoints_to_vtkPoints(ctrlpts, X)
        fv = os.path.normpath(os.path.join(self.odir, fname))
        ssm_tools.WritePolyData(fv, vtkp)

    def render_momenta_norm(self, moments, do_sq_norm=True, do_render=True):
        """
        render the norm of the momenta on the template geometry
        moments.shape = (ncp, k)
        """

        # read mesh
        v_pd = self.read_template()
        N = v_pd.GetPoints().GetNumberOfPoints()
        points = nps.vtk_to_numpy(v_pd.GetPoints().GetData()).astype('float64')

        # compute attributes
        z = self.convolve_momentum(moments, points)
        print(z.shape)

        if do_sq_norm:
            d = np.sum(z**2, axis=1)
        else:
            d = z[:, 0]

        # set attributes
        scalars = vtk.vtkFloatArray()
        for i in range(N):
            scalars.InsertTuple1(i, d[i])
        v_pd.GetPointData().SetScalars(scalars)

        # render
        if do_render:
            visualization_tools.renderVtkPolyData(v_pd, vmin=min(0, d.min()), vmax=d.max())

        return v_pd
