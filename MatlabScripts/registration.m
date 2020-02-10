%function [ distanceToPreviousTemplate ] = registration( folder, Filenames, Names, deformetrica_path )
%function [ distanceToPreviousTemplate ] = registration( folder, Filenames, Names )
function [ distanceToPreviousTemplate ] = registration( BasePath, IterationNumber, InitialGeomNames, Names )
%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%

% Function to perform one registration iteration - returns the distance to 
% the previous iteration template as information towards registration convergence 
% (i.e. if the template "does not move" from one iteration to the other,
% it means that the registration has converged)

        %% Define REFERENCE folder for registration
          % If a iteration folder already exists , chose the last one
          % If not, start from the input folder and do first iteration
        
          %%%%EMI : I need to copy Prototype.vtk from one iteration folder
          %%%%to another !!!
disp('******************************************');
fprintf('The BASE PATH is: %s\n',BasePath);
disp('******************************************');
disp('History of the functions called at runtime');
disp([ 9 'registration.m']);
          
%%%% EMI: ReferenceFolderInputREG should be renamed IterationFolder to be
%%%% consistent with the rest of the code
          if (IterationNumber == 0)
               %disp('**************DEBUG***************')
               %fprintf('Our iteration number is: %d\n', IterationNumber);
               %disp('**************DEBUG***************')
               ReferenceFolderInputREG = strcat(BasePath, '/Input');
               FloatingFiles = InitialGeomNames;
          elseif (IterationNumber == 1)
               %disp('registration.m: I am in the case where there is only one iteration folder: iter_1'); 
               ReferenceFolderInputREG = strcat(BasePath, filesep,'iteration_',num2str(IterationNumber));

               %disp('**************DEBUG***************')
               %fprintf('Our iteration number is: %d\n', IterationNumber);
               %fprintf('We are setting up the ReferenceFolderInputREG\n');
               %fprintf('The name of the ReferenceFolderInputREG is : %s\n', ReferenceFolderInputREG);
               %fprintf('It is a path!!!!\n');               
               %disp('**************DEBUG***************')
  
               for i=1:length(Names)
                    FloatingFiles{i,1} = strcat(Names{i}, '_ICPreg.vtk');
               end
               ReferenceFileName = 'Prototype.vtk';
               ReferenceFilePath = strcat(ReferenceFolderInputREG,filesep,ReferenceFileName);
               copyfile ( fullfile(ReferenceFolderInputREG, filesep, FloatingFiles{1,1}), ReferenceFilePath );

          else
               %disp('registration.m: I am in the case where there at least 2 iteration folders');
               %ReferenceFolderInputREG = ListIterFolder(end).name;
               ReferenceFolderInputREG = strcat(BasePath, filesep,'iteration_',num2str(IterationNumber));
               
               %disp('**********************************')
               %fprintf('Our iterartion number is: %d\n', IterationNumber);
               %fprintf('We are setting up the ReferenceFolderInputREG\n');
               %fprintf('The name of the ReferenceFolderInputREG is : %s\n', ReferenceFolderInputREG);
               %fprintf('It is a path!!!!\n');               
               %disp('**********************************')
               for i=1:length(Names)
                    FloatingFiles{i,1} = strcat(Names{i}, '_ICPreg.vtk');
               end
               data = strcat(ReferenceFolderInputREG, '/data');
               ReferenceFileName = 'Prototype.vtk';              
               ReferenceFilePath = strcat(ReferenceFolderInputREG,filesep,data,filesep,ReferenceFileName);
               
               %fprintf('The name of the ReferenceFileREG is : %s\n', ReferenceFileREG.name);

          end
        %% Define REFERENCE file for registration
          % If in the reference folder a template already exists , chose one of
          % the templates
          % If there are no template shapes, just chose one random shape
          
          A = exist('ReferenceFileName', 'var');
          if (A==0)              
               ReferenceFileName = InitialGeomNames{1};
               %%EMI: fprintf('My reference file REG is: %s\n', ReferenceFileREG);
               %%% EMI: the struct is empty!!!!
               %disp('**************DEBUG***************')
               %fprintf('We are in the case where struct is empty\n');
               %fprintf('Assigning a new name to Reference file REF = Filenames{1}\n');
               %fprintf('The name of the ReferenceFileREG is : %s\n', ReferenceFileName);
               %disp('**************DEBUG***************')
%{
          else
              disp('**************DEBUG***************')
              fprintf('We are in the case where struct has exactly 1 entry\n');                   
              fprintf('Assigning a new name to Reference file\n');
              fprintf('The name of the ReferenceFileREG is : %s\n', ReferenceFileName);
              disp('**************DEBUG***************')
              %{
               if length(ReferenceFileName) == 1 
                   ReferenceFileREG = ReferenceFileREG.name;
                   disp('**************DEBUG***************')
                   fprintf('We are in the case where struct has exactly 1 entry\n');                   
                   fprintf('Assigning a new name to Reference file\n');
                   fprintf('The name of the ReferenceFileREG is : %s\n', ReferenceFileName);
                   disp('**************DEBUG***************')
               else % Ask which template one wants to use
                    %[ReferenceFileREG,~,~] = uigetfile({'*_template.vtk','Mesh File'},'Select template mesh files for Registration',...
                    [ReferenceFileName,~,~] = uigetfile({'Prototype.vtk','Mesh File'},'Select template mesh files for Registration',...    
                         ReferenceFolderInputREG, 'MultiSelect', 'Off');
               end
              %}
      %}              
          end
          

        %% Define and create OUTPUT folder for this registration step
        % If a iteration folder already exists , increment of 1
        % If not, start do first iteration
        
        %%%EMI: This is the loop that creates the IterationFolder 
        %%% and increment by one
        %%% EMI: This could be joined with the previous bloc of
        %%% if-conditions
        listing = dir(strcat(BasePath, '/iteration*'));
        if isempty(listing)
             IterationFolder = strcat(BasePath, '/iteration_1');
        else
             noOfIterations = size(listing,1);
             %disp('**********************************')
             %disp('**********************************')
             %fprintf('We are now creating iteration_%s\n folder\n', int2str(noOfIterations+1));
             %disp('**********************************')
             %disp('**********************************')

             % This will create the folders: "iteration_2", "iteration_3",
             % "iteration_4", etc.
             IterationFolder = strcat(BasePath, '/iteration_', int2str(noOfIterations+1));
        end
        mkdir(IterationFolder);
        
        %% REGISTER floating meshes to reference file using VMTK iterative closest point (ICP) algorithm
         %%%EMI: STEP Nr 1: ICP registration from VMTK
         disp('**********************************')
         fprintf('STEP 1: Now entering ICPregistration\n');
         %fprintf('The name of the ReferenceFileREG is : %s\n', ReferenceFileName);
         disp('***********************************')

          %ICPregistrationMATLAB( ReferenceFolderInputREG, FloatingFiles, ReferenceFileREG, IterationFolder, Names ); 
          %ICPregistrationMATLAB( BasePath, ReferenceFolderInputREG, IterationNumber, FloatingFiles, ReferenceFileREG, IterationFolder, Names ); 
          

         ICPregistrationMATLAB( BasePath, ReferenceFolderInputREG, IterationNumber, InitialGeomNames, ReferenceFileName, IterationFolder, Names ); 
          
          

        %% Run Atlas construction to compute template at the end of each iteration
          % Function to perform atalas construction with a set of
          % registered input shapes. Parameters for the atlas are set
          % by the user. 
          
        %%%EMI: STEP Nr 2: Average shape computation using Deformetrica
        %atlasFolder = AtlasConstruction( IterationFolder ); % It returns the folder where the atlas files have been saved
        %[atlasFolder, OutputTemplateFile] = AtlasConstruction( IterationFolder, BasePath, ReferenceFileREG); % It returns the folder where the atlas files have been saved
        if (IterationNumber == 0)
            distanceToPreviousTemplate = surfaceDistance( IterationFolder, ReferenceFileName, ReferenceFileName );
            save(strcat(IterationFolder, '/Registration_DistanceInfo_to_', ReferenceFileName(1:end-4)), 'distanceToPreviousTemplate');
            
        else
            disp('**********************************')
            fprintf('STEP 2: Now entering AtlasConstruction\n');
            %fprintf('The name of the ReferenceFileName is : %s\n', ReferenceFileName);
            disp('**********************************')
            [OutputTemplateFile] = AtlasConstruction( IterationFolder, IterationNumber, InitialGeomNames, BasePath, ReferenceFileName); % It returns the folder where the atlas files have been saved

         
            
            
            %% Compute distance to previous iteration template, i.e. the refrence file used for registration
            % The distance is saved in each atlas folder
            % It can be used to understand registration convergence (i.e. if the
            % template "does not move" from one iteration to the other,
            % it means that the registration has converged)
        
        
            %%%EMI: STEP Nr 3: Distance between 2 shapes using VMTK
            %distanceToPreviousTemplate = surfaceDistance( ReferenceFolderInputREG, ReferenceFileREG, atlasFolder, OutputTemplateFile );
            disp('**********************************')
            fprintf('STEP 3: Now entering surfaceDistance\n');
            %fprintf('The name of the ReferenceFileREG is : %s\n', ReferenceFileName);
            disp('***********************************')
            distanceToPreviousTemplate = surfaceDistance( IterationFolder, ReferenceFileName, OutputTemplateFile );
            save(strcat(IterationFolder, '/Registration_DistanceInfo_to_', ReferenceFileName(1:end-4)), 'distanceToPreviousTemplate');
        end    
        
               
end

