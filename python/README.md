# Python SSM

More flexible implementation in python.

The main advantage is to be able to directly call deformetrica API.


--------------------------------------------------------------------------------
### TODO:

work in progress...

* recursive atlas estimation
* organize output


--------------------------------------------------------------------------------

### Dependencies of the code:

* Deformetrica 4.2
* VTK 8.2.0

--------------------------------------------------------------------------------
### How to run - Manual

* `main_ssm.py` is an example of possible use. It can be called using a config file
or command line parameters. Run `python main_ssm.py -h`

* `ssm_atlas.py` implements the `DeformetricaAtlasEstimation` class to interface
with Deformetrica API

* `ssm_pca.py` implements the `DeformetricaAtlasOutput` class for visualization purposes

* `ssm_tools.py` gives simpler functions to pre-process the meshes

* `ressources/` includes default config files

--------------------------------------------------------------------------------
#### To go further:

* The Deformetrica API is well documented https://gitlab.com/icm-institute/aramislab/deformetrica/-/wikis/3_user_manual/3.7_api
