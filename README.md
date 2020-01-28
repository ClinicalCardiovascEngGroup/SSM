# Statistical Shape Modeling (SSM)

This Matlab code was created by Dr. Jan Bruse and Dr. Benedetta Biffi in 2017. It consists of a Matlab GUI interface calling
two other free softwares responsible for the PCA and the shapes registration: Deformetrica 4.2 (http://www.deformetrica.org/) and VMTK 1.4 (http://www.vmtk.org/)
to allow a smooth experience of the user with statitiscal shape modelling. 

SSM is a tool to compute averaged shape and its variations within a coherent family of geometries (or training set). It is based on the assumption that each shape of the family is a deformed version of a reference shape. It uses Principal Component Analysis (PCA) on a set of landmark points to describe the shape of an object.

----------------------------------------------------------------------------------------
Literature references:

* Durrleman, Stanley, et al. "Morphometry of anatomical shape complexes with dense deformations and sparse parameters." NeuroImage 101 (2014): 35-49.

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
use solely the Prinicpal Component Analysis part using the latest Deformatrica version (v4.2)

----------------------------------------------------------------------------------------

### Dependencies of the code:

* MATLAB (not older than R2018b)
* Deformetrica 4.2
* VMTK 1.4


----------------------------------------------------------------------------------------
### How to run - Manual

Please clone the Git repository on your machine:
`git clone https://github.com/ClinicalCardiovascEngGroup/SSM.git`

Alternatively, if you are NOT planning to modify the code and use the sources as such, you can also download the content of the repository on your machine.

What you will get:
* a main file `ShapeAnalysis.m` 
* a folder containing all MATLAB functions called by the main
* a PDF document written by Dr. Bruse and Dr. Biffi giving more details on the present Matlab code
* a README.md 

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
}

function vmtkenv
{
  source /opt/vmtk/Install/vmtk_env.sh
}`


`function shapenv {
  vmtkenv
  deformenv
}`


**************
#### Note for new comers:
before you start running the code please read the code and folder structure below:

This is **an overview structure of the folders** created by the SSM code during a run:


![Folders architecture](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/GPA_IterationFolders.png)

To **run** the code please folow these instructions:

* *Step 0*: Make sure that Test_Folder ONLY contains the STLs files and nothing else (no folder named "RegistrationAtlasConstruction" for example)
* *Step 1*: Go to "SSM" directory which contains the "main" file (ShapeAnalysis_OnlyPCA_2019.m)
* *Step 2*: Type in a terminal "shapenv" --> this will call VMTK and Deformetrica environments
* *Step 3*: Then type the command: "matlab ShapeAnalysis_OnlyPCA_2019.m"
* *Step 4*: Click in the middle of the text editor where the main code appears (green play botton on top panel). A window "MATLAB Editor" appears asking to change the current folder path. Click on "Change Folder" button and go to your Test_Folder
* *Step 5*: A first menu with 2 choices appears. Click on the top button "Registration and Atlas construction"

If you choose "Registration and Atlas construction":

* *Step 6*: A window appears asking to select the input mesh files for Registration. Go to Test_Folder ie. "/SSM/Test_Folder" and select (highlight) ALL STL files presents. You should also see a folder named :"RegistrationAtlasConstruction". Don't worry about this folder - it was created after step nr. 4
This 5th step will first create an Input folder (See the overview structure of the folders in the picture above  - GPA_IterationFolders.png).


For those of you who which to work on the code directly, here is a small overview of the code architecture:


----------------------------------------------------------------------------------------
### More info on the details of the code, here:

