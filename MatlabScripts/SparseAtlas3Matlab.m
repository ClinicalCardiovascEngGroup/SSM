%function [  ] = SparseAtlas3Matlab( Folder, FileNames, paramfolder, deformetrica_path )
%function [  ] = SparseAtlas3Matlab( Folder, FileNames, param_folder )
%function [  ] = SparseAtlas3Matlab( folder, FileNames )
function [ OutputTemplateFile ] = SparseAtlas3Matlab( IterationFolder, IterationNumber, FileNames, BasePath, ReferenceFileName)
disp([ 9 9 9 'SparseAtlas3Matlab.m']);
%%EMI: fprintf('The iteration folder is: %s\n',IterationFolder);
%%EMI: fprintf('The path in base is: %s\n',BasePath);
%%EMI: fprintf('The ref file is: %s\n',ReferenceFileName); 

%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% NEW VERSION with Deformertrica 4.2 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% EMI : The sole purpose of this function is to run Deformetrica 4.2
disp('***************************');
disp('Running DEFORMETRICA 4.2');
disp('***************************');


StartFolder = pwd;
cd(IterationFolder);


shape_list = [strcat(IterationFolder, '/', FileNames{1}), ' '];

for j=1:length(FileNames)
     shape_list = [shape_list, strcat(IterationFolder, '/', FileNames{j}), ' '];
     copyfile *.vtk data
end



ListIterFolder = dir(strcat(BasePath, '/iteration*'));
    if isempty(ListIterFolder)
        disp('There are no interation_{i} folder');
    % If the array of structures has only one row ...
    elseif (size(ListIterFolder,1) == 1)
        %disp('I am in the case where there is only one iteration folder: iter_1'); 
        geom = dir(strcat(IterationFolder,filesep,'*vtk'));
        copyfile( geom(1).name, ReferenceFileName) ;

      
    elseif (size(ListIterFolder,1) == 2)
        %%%% EMI : This case works!!!!
        %disp('I am in the case where there are 2 iterations folders: iter_1 and iter_2');
        noOfIterations = size(ListIterFolder,1);
        PreviousIterFolderPath = strcat(BasePath, filesep,'iteration_',int2str(noOfIterations-1)');
        copyfile(fullfile(PreviousIterFolderPath, ReferenceFileName), fullfile(IterationFolder, filesep, 'data', filesep, ReferenceFileName));
        
        
        command1 = 'unset MKL_NUM_THREADS; deformetrica estimate model.xml data_set.xml -p optimization_parameters.xml >& deformetrica_run.log';
        [status,cmdout] = system(command1,'-echo');

        ListTemplateCurrentOutput = dir(strcat(IterationFolder,filesep,'output',filesep,'*Template*'));
        OutputTemplateFile = ListTemplateCurrentOutput.name;
        
        
    else
        %disp('I am in the case where there are several iteration folders');
        noOfIterations = size(ListIterFolder,1);
        PreviousDeformetricaOutputPath = strcat(BasePath, filesep,'iteration_',int2str(noOfIterations-1),filesep,'output');

        ListTemplatePrevOutput = dir(strcat(PreviousDeformetricaOutputPath,filesep,'*Template*'));
        if isempty(ListTemplatePrevOutput)
          fprintf('There are no *Template* file in %s\n',PreviousDeformetricaOutputPath);
          return
        end

        copyfile(fullfile(PreviousDeformetricaOutputPath,ListTemplatePrevOutput.name), fullfile(IterationFolder, filesep, 'data') );
        List = dir(fullfile(IterationFolder, filesep, 'data'));
        List.name;
                
        %EMI::: To test this:   movefile(fullfile(data, ListTemplatePrevOutput.name), fullfile(data, ReferenceFileREG));
        movefile(fullfile(IterationFolder, filesep, 'data',filesep, ListTemplatePrevOutput.name), fullfile(IterationFolder, filesep, 'data',filesep,'Prototype.vtk'));
        List = dir(fullfile(IterationFolder, filesep, 'data'));
        List.name;
       
        %command1 = 'unset MKL_NUM_THREADS; deformetrica estimate model.xml data_set.xml -p optimization_parameters.xml';
        command1 = 'unset MKL_NUM_THREADS; deformetrica estimate model.xml data_set.xml -p optimization_parameters.xml >& deformetrica_run.log';
        [status,cmdout] = system(command1,'-echo');
        
        ListTemplateCurrentOutput = dir(strcat(IterationFolder,filesep,'output',filesep,'*Template*'));
        if isempty(ListTemplateCurrentOutput)
          fprintf('There are no *Template* file in the current output folder');
          return
        end
        OutputTemplateFile = ListTemplateCurrentOutput.name;
        
     end


cd(StartFolder);

end

