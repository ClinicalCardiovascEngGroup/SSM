# Statistical Shape Modeling (SSM)

This Matlab code was created by Dr. Jan Bruse and Dr. Benedetta Biffi in 2017. It consists of a Matlab GUI interface calling
two other free softwares responsible for the PCA and the shape registration: Deformetrica 4.2 (http://www.deformetrica.org/) and VMTK 1.4 (http://www.vmtk.org/)
to allow a smooth experience of the user with statistical shape modelling (SSM).

SSM is a tool to compute an average shape and its variations within a coherent family of geometries (or training set). It is based on the assumption that each shape of the family is a deformed version of a reference shape. It uses Principal Component Analysis (PCA) on a set of landmark points to describe the shape of an object.

----------------------------------------------------------------------------------------
Literature references:

* Durrleman, Stanley, et al. "Morphometry of anatomical shape complexes with dense deformations and sparse parameters." NeuroImage 101 (2014): 35-49.
* Bône, Alexandre, et al. "Deformetrica 4: an open-source software for statistical shape analysis." International Workshop on Shape in Medical Imaging. Springer, Cham, 2018.
* Antiga, Luca, et al. "An image-based modeling framework for patient-specific computational hemodynamics." Medical & biological engineering & computing 46.11 (2008): 1097.

----------------------------------------------------------------------------------------

Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com, All rights reserved.                     
Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com, All rights reserved.        

If you are using this software, please cite our works:                                      

* Biffi, Benedetta, Bruse, Jan L., et al. (2017). “Investigating cardiac Motion Patterns Using
synthetic high-resolution 3D cardiovascular Magnetic resonance images and statistical        
shape analysis". Frontiers in Pediatrics 5.                                                  

* Bruse, Jan L., et al. (2016). "A Statistical Shape Modelling Framework to Extract 3D Shape   
Biomarkers from Medical Imaging Data: Assessing Arch Morphology of Repaired Coarctation      
of the Aorta." BMC Medical Imaging 16, no. 1.                                                

This code has been updated by Emilie Sauvage (July 2019) in order                        
use solely the Principal Component Analysis part using the latest Deformetrica version (v4.2)

----------------------------------------------------------------------------------------

### Dependencies of the code:

* MATLAB (not older than R2018b)
* Deformetrica 4.2
* VMTK 1.4


----------------------------------------------------------------------------------------
### How to run - Manual

Please clone the Git repository on your machine:
`git clone https://github.com/ClinicalCardiovascEngGroup/SSM.git`

Alternatively, if you are NOT planning to modify the code, you can also download the content of the repository on your machine.

What you will get:
* a main file `ShapeAnalysis_OnlyPCA_2020.m`
* a folder containing all MATLAB functions called by the main
* a folder containing several screenshots (PNG files) from the interface
* a PDF document written by Dr. Bruse and Dr. Biffi giving more details on the present Matlab code
* this README.md

Before you start, make sure all dependencies are correctly installed and tested.
For Linux users, you are advised to create the following environment in your .bashrc:

`function condaenv {
  __conda_setup="$('/opt/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
  if [ $? -eq 0 ]; then
      eval "$__conda_setup"
  else
      if [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
          . "/opt/anaconda3/etc/profile.d/conda.sh"
      else
          export PATH="/opt/anaconda3/bin:$PATH"
      fi
  fi
  unset __conda_setup

  export PATH=/opt/anaconda3/bin:$PATH
}`

`function deformenv {
  condaenv
  conda activate deformetrica
}`

`function vmtkenv
{
  source /opt/vmtk/Install/vmtk_env.sh
}`


`function shapenv {
  vmtkenv
  deformenv
}`


**************
 **Overview structure of the folders** created by the SSM code during a run:


![Folders architecture](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/GPA_IterationFolders.png)

**************

To **run** the code please follow these instructions:

* *Step 0*: Create a folder 'Test_Folder' that ONLY contains the STLs files and nothing else (no subfolder named "RegistrationAtlasConstruction" for example)
* *Step 1*: Go to source "SSM" directory which contains the "main" file (ShapeAnalysis_OnlyPCA_2019.m)
* *Step 2*: Type in a terminal `shapenv` --> this will call VMTK and Deformetrica environments
* *Step 3*: Then type the command: `matlab ShapeAnalysis_OnlyPCA_2020.m`
* *Step 4*: Click in the middle of the text editor where the main code appears (green play button on top panel). A window "MATLAB Editor" appears asking to change the current folder path. Click on "Change Folder" button and select your 'Test_Folder'
* *Step 5*: A first menu with 2 choices appears:

![Menu 1](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/Screenshot_Menu1.png)

If you choose "Registration and Atlas construction":

* *Step 0*: A window appears asking to select the input mesh files for Registration. Go to Test_Folder ie. "/SSM/Test_Folder" and select (highlight) ALL STL files presents. You should also see a folder named :"RegistrationAtlasConstruction". Don't worry about this folder - it was created after step nr. 4.
This 5th step will first create an Input folder (See the overview structure of the folders in the picture above  - GPA_IterationFolders.png). During this step all initial STLs files are copied into a folder called "Input".

To perform the Generalised Procustes Analysis (GPA analysis) on the registered geometries:

* *Step 1*: After iteration 0, you should be prompted with another menu (of 3 entries) asking you "What would you like to compute?". Click on "Registration - Generalized Procustes Analysis". This step performs a registration of all geometries with VMTK. A new folder "iteration_1" is created that contains all registered geometries in VTK format along with other distance information files.

![Menu 2](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/Screenshot_Menu2.png)

This step will perform the next iteration (=iteration nr.1) and prompted you with a small "End session" window asking you: "Would you like to continue with another registration and atlas construction step?". Click on "Yes" to perform the next iteration.

![End Session 1](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/Screenshot_EndSession1.png)

* *Step 2*: The same menu (of 3 entries) asking you "What would you like to compute?" will appear. Click one more time on "Registration - Generalized Procustes Analysis". First a new registration is performed with VMTK and a new folder appears "iteration_2".

An additional window with all parameters needed for the computation of the averaged model from Deformetrica will appear. Modify those parameters as you wish (default parameters are provided). Press on "OK" button. We are now at iteration nr. 2. In the folder "RegistrationAtlasConstruction", you should now see 3 sub-folders: Input, iteration_1, iteration_2

![Deformetrica Parameters](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/Screenshot_DeformetricaParameters.png)

Once you have validated the parameters, Deformetrica will operate the computation of the average shape from the previously registered geometries. This might take a few minutes.

At the end of this step you should see a graph window displaying distance from previous template vs. number of iterations and another small "end session" window asking: "Would you like to continue with another registration and atlas construction step?". Click on "Yes" to perform the next iteration.

![DistIteration Graph 1](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/DistIteration1.png)

![End Session 2](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/Screenshot_EndSession2.png)

* *Step 3*: After iteration 2, you should be prompted with another menu (of 3 entries) asking you "What would you like to compute?". Click on "Registration - Generalized Procustes Analysis". This step will perform the next iteration (=iteration nr.3)
Continue the iterations until you are satisfied with the distance from the previous template - the curve should reach a plateau after a few iteration - this plateau is a good moment to stop iterating.

![DistIteration Graph 2](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/DistIteration2.png)


**************
### For the developers
For those of you who wish to work on the code directly, here is a small overview of the code architecture:

![Code Call Graph](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/Illustrations/CodeArchitecture.png)

----------------------------------------------------------------------------------------
#### To go further:

* Online course on statistical shape modelling from the University of Basel: http://shapemodelling.cs.unibas.ch/
