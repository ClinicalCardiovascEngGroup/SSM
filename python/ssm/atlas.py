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
try:
    import readline
    readline.parse_and_bind("tab: complete")
except:
    pass

import numpy as np
import scipy, scipy.linalg

import vtk
from vtk.util import numpy_support as nps
import torch

sys.path.append('/home/face3d/programs/deformetrica/')
import deformetrico as deformetrica

from . import iovtk
from . import data_set_xml
from . import visualization_tools

#import logging
#logger = logging.getLogger("ssm_atlas")
#logger.setLevel(logging.INFO)

################################################################################
##  Deformetrica estimation class
class DeformetricaAtlasEstimation():
    """ result of an DeterministicAtlas estimation """

    def __init__(self, idir="./", odir="output/", name="obj", initial_guess="", kwd=20., kwg=10., noise=10.):
        self.idir = idir
        self._lf = None
        self.odir = odir
        self.id = name
        self.object_type = 'surfacemesh'
        self.attachment = 'current'
        self.description = ''

        if isinstance(initial_guess, int):
            self.initial_guess = self.lf[initial_guess]
        else:
            self.initial_guess = initial_guess

        self.p_kernel_width_deformation = kwd
        self.p_kernel_width_geometry = kwg
        self.p_noise = noise

        self.dataset_xml = ""

        self.optimization_parameters_xml = os.path.join(os.path.dirname(__file__), "../ressources/optimization_parameters.xml")
        self.model_xml = os.path.join(os.path.dirname(__file__), "../ressources/model-atlas-template.xml")

    def _get_list_vtk(self, sorted=False):
        """ list of vtk using prefix in self.idir """
        if os.path.isdir(self.idir):
            lf = glob.glob(os.path.join(self.idir, "*.vtk"))
        else:
            lf = glob.glob(self.idir)

        if sorted:
            lf.sort()
        return lf

    @property
    def lf(self):
        if self._lf is None:
            self._lf = self._get_list_vtk(sorted=True)
        return self._lf
    @lf.setter
    def lf(self, l):
        self._lf = l

    @property
    def n_subjects(self):
        try:
            return len(self.lf)
        except:
            return 0

    def save_parameters(self, do_save_lvtk=False):
        """ printing parameters in a json for future use if need be """
        d = {"idir":self.idir,
            "odir":self.odir,
            "name":self.id,
            "init":self.initial_guess,
            "kwd":self.p_kernel_width_deformation,
            "kwg":self.p_kernel_width_geometry,
            "noise":self.p_noise,
            "object_type":self.object_type,
            "attachment":self.attachment,
            "description":self.description
            }
        if do_save_lvtk:
            d.update({"files":self.lf})
        sp.call(["mkdir", "-p", self.odir])
        with open(os.path.join(self.odir, "params.json"), "w") as fd:
            json.dump(d, fd, indent=2)

    def load_parameters(self, fjson, verbose=True):
        """loading parameters from a json"""
        with open(fjson, "r") as fd:
            d = json.load(fd)

            self.idir = d["idir"]
            self.odir = d["odir"]
            self.id = d["name"]
            self.initial_guess = d["init"]
            self.p_kernel_width_deformation = d["kwd"]
            self.p_kernel_width_geometry = d["kwg"]
            self.p_noise = d["noise"]

            if self.n_subjects == 0:
                print("W: idir path {} does not match any existing file".format(self.idir))
            if not os.path.exists(os.path.abspath(self.odir)):
                print("W: odir path {} does not exist".format(self.odir))
            if not os.path.exists(os.path.abspath(self.initial_guess)):
                print("W: initial_template path {} does not exist".format(self.initial_guess))

            try:
                self.lf = d["files"]
                if len(d["files"]) > 5:
                    d["files"] = "[{} listed files, ...]".format(len(d["files"]))
            except KeyError:
                pass
            try:
                self.object_type = d["object_type"]
                self.attachment = d["attachment"]
                self.description = d["description"]
            except KeyError:
                pass

            if verbose:
                print("loading parameters: ")
                for k in d:
                    print("  {:6}: {}".format(k, d[k]))
                print("n subjects: ", self.n_subjects)


    def check_initialisation(self, do_quick=False):
        """ check that the input paths exist """

        def _get_input_path(s, p, pass_check=False):
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

        def _get_input_float(s, p):
            """ ask for a float using string s and default value p """
            while True:
                try:
                    x = input(s + " [{}]: ".format(p))
                    if x == "":
                        x = p
                    return float(x)
                except ValueError:
                    print("not a float")

        if do_quick:
            try:
                k = int(self.initial_guess)
                self.initial_guess = self.lf[k]
            except ValueError:
                pass
            ae.p_kernel_width_geometry = float(ae.p_kernel_width_geometry)
            ae.p_kernel_width_deformation =  float(ae.p_kernel_width_deformation)
            ae.p_noise =  float(ae.p_noise)
        else:

            self.idir = _get_input_path("Set input directory", self.idir, True)
            print(len(self._get_list_vtk()), "vtk-files in input directory")

            self.odir = _get_input_path("Set output directory", self.odir)
            #sp.call(["mkdir", "-p", self.odir])
            #print("output directory created: ", self.odir)

            try:
                k = int(self.initial_guess)
                self.initial_guess = self.lf[k]
            except ValueError:
                pass

            self.initial_guess = _get_input_path("Set initial mesh", self.initial_guess)
            self.optimization_parameters_xml = _get_input_path("Set optimization xml",self.optimization_parameters_xml)
            self.model_xml = _get_input_path("Set model xml",self.model_xml)

            self.p_kernel_width_geometry = _get_input_float('Set kernel width geometry: ', self.p_kernel_width_geometry)
            self.p_kernel_width_deformation =  _get_input_float('Set kernel width deformation: ', self.p_kernel_width_deformation)
            self.p_noise =  _get_input_float('Set noise level: ', self.p_noise)



    def create_dataset_xml(self):
        """ with every vtk files in idir """
        self.dataset_xml = os.path.join(self.odir, "dataset.xml")
        data_set_xml.create_xml_atlas(self.lf, self.dataset_xml, self.id)

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

        if self.n_subjects == 0:
            print("no input data for estimation!")
            return

        # General parameters
        xml_parameters = deformetrica.in_out.XmlParameters()
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
        template_object['deformable_object_type'] = self.object_type
        template_object['attachment_type'] = self.attachment
        template_object['kernel_type'] = 'torch'

        template_object['noise_std'] =  self.p_noise
        template_object['filename'] = os.path.abspath(self.initial_guess)
        template_object['kernel_width'] = self.p_kernel_width_geometry
        xml_parameters.template_specifications[self.id] = template_object

        ## Estimation
        # call: deformetrica estimate model.xml dataset.xml -p optimization_parameters.xml
        odir = os.path.join(self.odir, "output/")
        Deformetrica = deformetrica.api.Deformetrica(output_dir=odir, verbosity="DEBUG")

        Deformetrica.estimate_deterministic_atlas(
            xml_parameters.template_specifications,
            deformetrica.in_out.get_dataset_specifications(xml_parameters),
            estimator_options=deformetrica.in_out.get_estimator_options(xml_parameters),
            model_options=deformetrica.in_out.get_model_options(xml_parameters))


        ## cleaning a bit the deformetrica output
        if not do_keep_all:
            if odir.find(" ") == -1:
                for f in glob.glob(odir + "DeterministicAtlas__flow__*.vtk"):
                    os.remove(f)
            else:
                print("cannot remove flow files because of ' '")


    def shooting(self, fv, odir, tmin=-5, tmax=+5, fmesh=None, fcontrolpoints=None, do_keep_all=False, concentration_of_time_points=4):
        """
        warp Atlas using momenta in fv
        interesting momenta are in DeterministicAtlas__EstimatedParameters__Momenta.txt
        or given by pca using save_eigv
        arguments:
            fv      momenta file
            odir    output directory
            tmin, tmax shooting parameters
            fmesh   mesh to deform (default=Template)
            fcontrolpoints control points (default from output dir)
            do_keep_all do keep all the result files
        return:
            None
        """

        # General parameters
        xml_parameters = deformetrica.in_out.XmlParameters()
        xml_parameters._read_optimization_parameters_xml(self.optimization_parameters_xml)
        xml_parameters._read_model_xml(self.model_xml)

        xml_parameters.model_type = "shooting"
        if fcontrolpoints is None:
            fcontrolpoints = os.path.normpath(
                os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt"))
        xml_parameters.initial_control_points = fcontrolpoints
        xml_parameters.initial_momenta = fv

        # Template
        template_object = xml_parameters._initialize_template_object_xml_parameters()
        template_object['deformable_object_type'] = self.object_type
        template_object['attachment_type'] = self.attachment

        if fmesh is None:
            fmesh = os.path.normpath(os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__Template_"+self.id+".vtk"))
        template_object['filename'] = fmesh
        xml_parameters.template_specifications[self.id] = template_object

        # Deformation parameters
        xml_parameters.concentration_of_time_points = concentration_of_time_points
        xml_parameters.deformation_kernel_width = self.p_kernel_width_deformation

        xml_parameters.t0 = 0.
        xml_parameters.tmin = tmin
        xml_parameters.tmax = tmax

        ## Shooting
        # call: deformetrica compute model.xml -p optimization_parameters.xml
        Deformetrica = deformetrica.api.Deformetrica(output_dir=odir, verbosity="INFO")

        Deformetrica.compute_shooting(xml_parameters.template_specifications,
                model_options=deformetrica.in_out.get_model_options(xml_parameters))

        if not do_keep_all:
            if odir.find(" ") == -1:
                visualization_tools.rename_df2pv(odir + "Shooting__GeodesicFlow__" + self.id)
                sp.call("rm "+odir+"Shooting__GeodesicFlow__ControlPoints__tp*.txt", shell=True)
                sp.call("rm "+odir+"Shooting__GeodesicFlow__Momenta__tp*.txt", shell=True)
            else:
                print("cannot remove flow files because of ' '")


    def registration(self, fmesh, odir, subject_id="subj", use_template=True, xml_params=None, number_of_time_points=10, ):
        """
        register a (new) subject to the template


        use_template is used to choose between:
            - the estimation results: Template and control points (do not require: idir, initial_guess)
            - the initial guess: Mesh only (used for simple registration)
        """

        # General parameters
        xml_parameters = deformetrica.in_out.XmlParameters()
        xml_parameters._read_optimization_parameters_xml(self.optimization_parameters_xml)
        xml_parameters._read_model_xml(self.model_xml)

        xml_parameters.model_type = "registration"

        if use_template:
            xml_parameters.initial_control_points = os.path.normpath(
                os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt"))
            xml_parameters.freeze_control_points = True


        # Overwriting main parameters
        xml_parameters.number_of_time_points = number_of_time_points

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
        template_object['deformable_object_type'] = self.object_type
        template_object['attachment_type'] = self.attachment
        template_object['kernel_type'] = 'torch'
        template_object['noise_std'] = self.p_noise
        template_object['kernel_width'] = self.p_kernel_width_geometry

        if use_template:
            template_object['filename'] = os.path.normpath(
                os.path.join(self.odir, "output/DeterministicAtlas__EstimatedParameters__Template_"+self.id+".vtk"))
        else:
            template_object['filename'] = self.initial_guess

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
            estimator_options=deformetrica.in_out.get_estimator_options(xml_parameters),
            model_options=deformetrica.in_out.get_model_options(xml_parameters))


    def momenta_from_sbj_to_atlas(self, sbj, odir, do_warpback=False):
        """
        special case of forward-backward shooting to invert the atlas-subject deformation
        sbj         subject id (int)

        return file_moment, file_ctrlpts
        """
        m = self.read_momenta()[sbj, :, :]
        if do_warpback:
            mesh = self.lf[sbj]
        else:
            mesh = None
        return self.apply_reverse_transform(m, mesh, odir)

    def apply_reverse_transform(self, momentum, fmesh, odir):
        """
        forward-backward shooting to invert the atlas-subject deformation
        momentum, array
        mesh, path (can be None)

        return file_moment, file_ctrlpts
        """

        sp.call(["mkdir", "-p", odir])
        ipfx = self.odir + "output/DeterministicAtlas__EstimatedParameters__"

        # moment
        np.savetxt(odir + "forward_momenta.txt", momentum)

        # forward to get end momenta
        template_specifications = {
            self.id: {'deformable_object_type': self.object_type,
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
        if os.path.exists(fmesh):
            template_specifications = {
                self.id: {'deformable_object_type': self.object_type,
                'noise_std': self.p_noise,
                'filename': fmesh}
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

        kern = deformetrica.support.kernels.factory("torch", gpu_mode=True, kernel_width=self.p_kernel_width_deformation)

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
        ft = os.path.abspath(ft)
        return iovtk.ReadPolyData(ft)

    def read_momenta(self):
        """ read the momenta file, first line contain the shape """
        a_momenta = np.loadtxt(self.odir + "output/DeterministicAtlas__EstimatedParameters__Momenta.txt")
        shape = a_momenta[0,:].astype("int")
        return a_momenta[1:, :].reshape(shape)

    def read_residuals(self):
        return np.loadtxt(self.odir + "output/DeterministicAtlas__EstimatedParameters__Residuals.txt")

    def read_ctrlpoints(self):
        return np.loadtxt(self.odir + "output/DeterministicAtlas__EstimatedParameters__ControlPoints.txt")

    def save_controlpoints_vtk(self, fname="controlpoints.vtk", X=None):
        """ save controlpoints (could add some momenta X (n, d)) as vtk point cloud """
        if X is not None and X.ndim == 1:
            X = X.reshape((-1, 1)).astype("float")
        ctrlpts = self.read_ctrlpoints()
        vtkp = iovtk.controlpoints_to_vtkPoints(ctrlpts, X)
        iovtk.WritePolyData(os.path.normpath(os.path.join(self.odir, fname)), vtkp)


    def render_momenta_norm(self, moments, do_weight=False, set_xmax=None, do_sq_norm=True, do_render=False, fname=""):
        """
        render the norm of the momenta on the template geometry
        moments.shape = (ncp, k)

        do_weight   normalize the result such that constant moments lead to a constant map
        do_sq_norm  L2 norm of the momenta (otherwise show x[0])
        do_render   vtk live render
        fname       if set, save the resulting polydata
        """
        v_pd = self.read_template()
        N = v_pd.GetPoints().GetNumberOfPoints()
        points = nps.vtk_to_numpy(v_pd.GetPoints().GetData()).astype('float64')

        # compute attributes
        z = self.convolve_momentum(moments, points)
        print(z.shape)

        if do_weight:
            w = self.convolve_momentum(np.ones((moments.shape[0], 1)), points)
            z = z/w

        if do_sq_norm:
            d = np.sum(z**2, axis=1)
        else:
            d = z[:, 0]

        if set_xmax is not None:
            d *= set_xmax / d.max()

        # set attributes
        scalars = vtk.vtkFloatArray()
        for i in range(N):
            scalars.InsertTuple1(i, d[i])
        v_pd.GetPointData().SetScalars(scalars)

        # render
        if do_render:
            visualization_tools.renderVtkPolyData(v_pd, vmin=min(0, d.min()), vmax=d.max())
        if len(fname): #isinstance(do_save, str)
            iovtk.WritePolyData(self.odir + fname, v_pd)

        return v_pd

    def plot_loglikelihood(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        flogs = glob.glob(self.odir + "output/*.log")
        if len(flogs) == 0:
            print("no log file in:", self.odir + "output/")
            return fig, []
        elif len(flogs) > 1:
            print("several log files: ", flogs)

        ll = visualization_tools.read_loglikelihood_from_log(flogs[0])
        lll = np.log(-1 * np.array(ll))
        ax.plot(lll, ".");

        print("last =", "{:.4f}".format(lll[-2] - lll[-1])) # 1 - exp(-x) ~ x
        return fig, ll
