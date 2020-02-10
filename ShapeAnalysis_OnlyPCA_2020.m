%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%                                                                                              %%%%%%%%%
%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com, All rights reserved.                     %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com, All rights reserved.        %%%%%%%%%
%%%%%%%%                                                                                              %%%%%%%%%
%%%%%%%% If you are using this software, please cite our works!                                       %%%%%%%%%
%%%%%%%%                                                                                              %%%%%%%%%
%%%%%%%% Biffi, Benedetta, Bruse, Jan L., et al. (2017). “Investigating cardiac Motion Patterns Using %%%%%%%%%
%%%%%%%% synthetic high-resolution 3D cardiovascular Magnetic resonance images and statistical        %%%%%%%%%
%%%%%%%% shape analysis". Frontiers in Pediatrics 5.                                                  %%%%%%%%%
%%%%%%%%                                                                                              %%%%%%%%%
%%%%%%%% Bruse, Jan L., et al. (2016). "A Statistical Shape Modelling Framework to Extract 3D Shape   %%%%%%%%%
%%%%%%%% Biomarkers from Medical Imaging Data: Assessing Arch Morphology of Repaired Coarctation      %%%%%%%%%
%%%%%%%% of the Aorta." BMC Medical Imaging 16, no. 1.                                                %%%%%%%%%
%%%%%%%%                                                                                              %%%%%%%%%
%%%%%%%% This code has been updated by Emilie Sauvage (July 2019) in order                            %%%%%%%%%
%%%%%%%% use solely the Prinicpal Component Analysis part have using the                              %%%%%%%%%
%%%%%%%% latest Deformatrica version (v4.2)                                                           %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all; close all; clc;

%addpath(genpath('../EMI_SSAtoolbox'));
%cd ('/home/emilie/Benni_Handover/EMI_SSAtoolbox/EMI_MatlabScripts');
cd ('/home/emilie/Work/MyLittleCodes/SSM_Matlab.git/MatlabScripts');

%% Ask where to store analysis results and add to path

fullanalysis_folder_name = uigetdir('../','Select folder where you want to save your analysis results');
%addpath(genpath(fullanalysis_folder_name));

%% Display Menu with pipeline options

finish=0;
while finish == 0
     choice_menu = menu('What would you like to do?', 'Registration and Atlas construction','Postprocessing with PCA');
     option.WindowStyle = 'normal';
     switch choice_menu
                   
         case 1 %'Registration and Atlas Construction'
               display('--------- Loading Registration ---------');
               %RegistrationAtlasConstruction( fullanalysis_folder_name, deformetrica_path );
               RegistrationAtlasConstruction( fullanalysis_folder_name );

         
               
          case 2 %'Postprocessing with PCA'
               display('--------- Loading Postprocessing with PCA ---------');
               %PostprocessingWithPCA( fullanalysis_folder_name, deformetrica_path );
               PostprocessingWithPCA( fullanalysis_folder_name );
     end
     
     % Ask whether to continue or not
     choice_to_finish = questdlg('Would you like to continue with another step of the shape analysis?', 'End session', 'Yes','No','Yes');
     % Handle response
     switch choice_to_finish
         case 'Yes'
              finish=0;
         case 'No'
              finish=1;
     end
end
