# Statistical Shape Modeling (SSM)

This Matlab code was created by Dr. Jan Bruse and Dr. Benedetta Biffi in 2017. It consists of a Matlab GUI interface calling
two other free softwares responsible for the PCA and the shapes registration: Deformetrica 4.2 (http://www.deformetrica.org/) and VMTK 1.4 (http://www.vmtk.org/)
to allow a smooth experience of the user with statitiscal shape modelling. 

SSM is a tool to compute averaged shape and its variations within a coherent family of geometries (or training set). It is based on the assumption that each shape of the family is a deformed version of a reference shape. It used Principal Component Analysis (PCA) on a set of landmark points to describe the shape of an object.

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



This is an overview of the folders structure created by the SSM code:


![Folders architecture](https://github.com/ClinicalCardiovascEngGroup/SSM/blob/master/GPA_IterationFolders.png)
