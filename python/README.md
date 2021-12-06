# Python SSM

More flexible implementation in python.

The main advantage is to be able to directly call deformetrica API.


--------------------------------------------------------------------------------
### TODO:

work in progress...

* recursive atlas estimation
* organize output
* update for 4.3.0, in particular packaging was improved in 4.3.0. No more modules api, core, in_out and support at the same level.


--------------------------------------------------------------------------------

### Dependencies of the code:

* Deformetrica 4.3
* VTK 8.2.0

--------------------------------------------------------------------------------
### How to run - Manual

* `main_ssm.py` is an example of possible use. It can be called using a config file
or command line parameters. Run `python main_ssm.py -h`

* `ssm/` is the python module useful to run everything, in particular:
  * `atlas.py` implements the `DeformetricaAtlasEstimation` class to interface
  with Deformetrica API
  * `pca.py` implements the `DeformetricaAtlasOutput` class for PCA estimation
  * `iovtk.py` gives simpler functions to read/write the meshes and momenta
  * `tools.py` gives simpler functions to pre-process the meshes
  * `stats/` submodule for multivariate statistics

* `ressources/` includes default config files



--------------------------------------------------------------------------------
### Deformetrica tricks and tips

- Memory (RAM). In general you would want to reduce the number of nodes in your template mesh (you can for example smooth it/remeshed it with meshmixer) : 5 to 20k points should be ok.  You can also:
  - increase the deformation kernel width,
  - reduce the number of subjects (of course not optimal),
  - remesh every subject (more work, for relatively less effect),
  - example: 12 subjects, 1200 points/mesh, 400 controls points = 1.7GB.

- STLs files are not fully supported: use vtk

- Check if the normals of the meshes are consistently oriented, ie if the inside/outside are properly defined (in Paraview, Filter:Generate Surface Normals, then Display (using Coloring) one component x,y or z)

- If the atlas estimation do not fit (same or similar reconstruction for each subject, same as initialization):
  - change the optimization parameters (more iterations, scipyBFGS/GradientAscent, convergence=1e-4)
  - decrease the deformation kernel width (if too few controlpoints)
  - lower the noise level (noise-std in model.xml)
  - check if there is any 'NaNs' in the momenta file.


- Can run deformetrica using:  `deformetrica estimate model.xml data_set.xml -p optimization_parameters.xml -v INFO`


--------------------------------------------------------------------------------
#### To go further:

* The Deformetrica API is well documented https://gitlab.com/icm-institute/aramislab/deformetrica/-/wikis/3_user_manual/3.7_api
