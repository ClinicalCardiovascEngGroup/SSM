%function [  ] = RegistrationAtlasConstruction( fullanalysis_folder_name, deformetrica_path )
function [  ] = RegistrationAtlasConstruction( fullanalysis_folder_name)

%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%

%RegistrationAtlasConstruction
%   Function implementing the menu to perform Generalised Procustes
%   Analysis for input shape registration, Atlas construction and atlas
%   registration error computation

     %% Create working folder for this script and check if it already exists
     % Main working folder
     BasePath = strcat(fullanalysis_folder_name, '/RegistrationAtlasConstruction');
     mkdir(BasePath);
     fprintf('The BASE PATH is: %s\n',BasePath);
     
     % Folder with initial input files
     registrationInputfolder = strcat(fullanalysis_folder_name, '/RegistrationAtlasConstruction/Input');
     mkdir(registrationInputfolder);
     fprintf('The registration Input Folder PATH is: %s\n',registrationInputfolder);

     
     %% If input folder is empty, ask to manually select vtk or stl files to be used as input

     [InitialGeomNames, Name, ext] = readInFilenames(registrationInputfolder);
     if ext == 0
          [InitialGeomNames,PathName,FilterIndex] = uigetfile({'*.vtk;*.stl','Mesh Files'},'Select input mesh files for Registration',...
          '../../', 'MultiSelect', 'On');

          % Copy selected files into input folder 
          for i=1:size(InitialGeomNames, 2)
               system(['cp ', strcat(PathName, '/', InitialGeomNames{i}), ' ', registrationInputfolder]);
          end
     
          % Convert binary into ASCII
          [InitialGeomNames, Name, ext] = readInFilenames(registrationInputfolder);
          WriteASCII(registrationInputfolder, InitialGeomNames, Name, ext);
          
     end
     

          
%% Menu with options for registration and atlas construction
% Menu keep asking action to do until one says no
finish=0;

while finish == 0
     ListIterFolder = dir(strcat(BasePath, '/iteration*'));
     if isempty(ListIterFolder)
         IterationNumber = 0;
   
     else (size(ListIterFolder,1) == 1)
         IterationNumber = size(ListIterFolder,1);
         
     end
     disp('**************DEBUG***************')
     fprintf('Our iteration number is: %d\n', IterationNumber);
     disp('**************DEBUG***************')

     choice_menu = menu('What would you like to compute?','Registration - Generalised Procustes Analysis','Atlas construction error - check for lambda parameters','Atlas Construction');
     option.WindowStyle = 'normal';
     
     switch choice_menu
          
          case 1 %Registration - Generalised Procustes Analysis
               display('--------- Loading Registration - Generalised Procustes Analysis ---------');         
               % Function to perform one registration iteration - returns
               % the distance to the previous iteration template as
               % information towards registration convergence (i.e. if the
               % template "does not move" from one iteration to the other,
               % it means that the registration has converged)
               
               %distanceToPreviousTemplate = registration( Mainfolder, FileNames, Name, deformetrica_path );
               distanceToPreviousTemplate = registration( BasePath, IterationNumber, InitialGeomNames, Name  );
               if (IterationNumber > 0)
                   distanceToPreviousTemplate = [distanceToPreviousTemplate.D_Average];
                   iterationsTemplateDistance = plotTemplateDistance( BasePath );
                   display(strcat('Distance to previous tempate is: ', num2str(mean(distanceToPreviousTemplate)), ' mm'));
                   display(strcat('Template distance trend is: ', num2str(iterationsTemplateDistance), ' mm'));
               end
               
          case 2 %'Atlas construction error - check for lambda parameters'
               display('--------- Loading Atlas construction error - check for lambda parameters ---------');    
               % Function to compute the average surface distance
               % between each shape and the ones computed by the atlas,
               % averaged on the whole set of input shapes. Together with
               % visual inspection, it can help in the process of
               % atlas parameter tuning
               
               surfaceDistanceData = atlasRegistrationError( BasePath );
               Registration GPA;
               averageDistanceAtlas = [averageDistanceAtlas.D_Average];
               display(strcat('Atlas construction Error is: ', num2str(mean(averageDistanceAtlas)), ' mm'));
               
          case 3 %'Atlas Construction'
               display('--------- Loading Atlas Construction ---------');               
               % Function to perform atalas construction with a set of
               % registered input shapes. Parameters for the atlas are set
               % by the user. 
               
               % Ask at which registration iteration one wants to get the
               % input data from
               iteration_folder_name = uigetdir(BasePath,'Select iteration folder where you want to compute Atlas');
               AtlasConstruction( iteration_folder_name, deformetrica_path );              
               
     end
     
     % Ask whether to continue or not
     choice_to_finish = questdlg(' Would you like to continue with another registration and atlas construction step?', 'End session', 'Yes','No','Yes');
     % Handle response
     switch choice_to_finish
         case 'Yes'
              finish=0;
         case 'No'
              finish=1;
     end
end
     
end
