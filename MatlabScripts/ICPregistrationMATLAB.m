%function [  ] = ICPregistrationMATLAB( inputFolder, Filenames, ReferenceFileREG, outputFolder, Names )
function [  ] = ICPregistrationMATLAB( BasePath, IterationFolder, IterationNumber, InitialGeomNames, ReferenceFileName, NextIterationFolder, Names )

disp([ 9 9 'ICPregistrationMATLAB.m']);

%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%
% 
%         manually PRE-ALIGN meshes! 
% 
%         -rfile: reference file for registration
%         -farthreshold: threshold distance beyond which points are discarded during optimization
%         -landmarks: maximum number of landmarks sampled from the two surfaces for evaluation of the registration metric
%         -iterations: maximum number of iterations for the optimization problems
%         -maxmeandistance: convergence threshold based on the maximum mean distance between the two surfaces


if (IterationNumber == 0)
    FileNames = InitialGeomNames;
else
    FileNames = cell(length(InitialGeomNames),1);

    for j = 1:length(InitialGeomNames)
        C = strsplit(InitialGeomNames{j}, '.');
        FileNames{j} = strcat(C{1}, '_ICPreg.vtk');
    end
end


if (IterationNumber >= 1)
    shape_list = [strcat(IterationFolder, '/', FileNames{1}), ' '];

    for j=1:length(FileNames)
        %fprintf('The filenames are: %s\n', FileNames{j});
        shape_list = [shape_list, strcat(IterationFolder, '/', FileNames{j}), ' '];
        copyfile *.vtk data
    end
end

if (IterationNumber == 0)
    %disp('I am in the case where there is only Input folder');
    %fprintf('The current folder is: %s\n',IterationFolder);
    geom = dir(strcat(IterationFolder,filesep,'*stl'));

   %%%%EMI: copy the first patient geometry into Prototype.stl:
   copyfile( fullfile(IterationFolder, filesep, geom(1).name), fullfile(IterationFolder, filesep, 'Prototype.stl') ) ;
   for j = 1:length(FileNames) 
       system(['vmtkicpregistration -ifile ', strcat(IterationFolder, '/', FileNames{j}), ' -rfile ',...
       strcat(IterationFolder, '/', ReferenceFileName),...
       ' -flipnormals 0 -farthreshold 1000.0 -landmarks 1000 -iterations 100 -maxmeandistance 0.01 -ofile ', ...
       strcat(NextIterationFolder, '/', Names{j}, '_ICPreg.vtk'),...
       ' --pipe vmtksurfacewriter -ifile ',strcat(NextIterationFolder, '/', Names{j}, '_ICPreg.vtk'), ' -mode ascii -ofile ',...
       strcat(NextIterationFolder, '/', Names{j}, '_ICPreg.vtk') ])
   end

    
elseif (IterationNumber == 1)
    disp('I am in the case where there are 2 iterations folders: iter_1 and iter_2');

    mkdir(fullfile(NextIterationFolder, filesep, 'data'));

    copyfile(fullfile(IterationFolder,filesep, ReferenceFileName), fullfile(NextIterationFolder, filesep, 'data', filesep, ReferenceFileName) );
    
    for j = 1:length(FileNames) 
        system(['vmtkicpregistration -ifile ', strcat(IterationFolder,filesep, FileNames{j}), ' -rfile ',...
        strcat(NextIterationFolder,'/','data', '/', ReferenceFileName),...
        ' -flipnormals 0 -farthreshold 1000.0 -landmarks 1000 -iterations 100 -maxmeandistance 0.01 -ofile ', ...
        strcat(NextIterationFolder, filesep, Names{j}, '_ICPreg.vtk'),...
        ' --pipe vmtksurfacewriter -ifile ',strcat(NextIterationFolder, '/', Names{j}, '_ICPreg.vtk'), ' -mode ascii -ofile ',...
       strcat(NextIterationFolder, '/', Names{j}, '_ICPreg.vtk') ])
    end

    
else
    disp('I am in the case where there more than 2 iterations folders');

    mkdir(fullfile(NextIterationFolder, filesep, 'data'));
       
     ListTemplatePrevOutput = dir(strcat(IterationFolder,filesep,'output',filesep,'*Template*'));
    if isempty(ListTemplatePrevOutput)
        fprintf('There are no *Template* file in %s\n',PreviousDeformetricaOutputPath);
        return
    end

    copyfile(fullfile(IterationFolder,filesep, 'output', filesep, ListTemplatePrevOutput.name),...
        fullfile(NextIterationFolder, filesep, 'data', filesep, ReferenceFileName) );

    %List = dir(fullfile(NextIterationFolder, filesep, 'data', filesep, ReferenceFileName));
    %List.name;
    

    for j = 1:length(FileNames) 
        system(['vmtkicpregistration -ifile ', strcat(IterationFolder,filesep, FileNames{j}), ' -rfile ',...
        strcat(NextIterationFolder,'/','data', '/', ReferenceFileName),...
        ' -flipnormals 0 -farthreshold 1000.0 -landmarks 1000 -iterations 100 -maxmeandistance 0.01 -ofile ', ...
        strcat(NextIterationFolder, filesep, Names{j}, '_ICPreg.vtk'),...
        ' --pipe vmtksurfacewriter -ifile ',strcat(NextIterationFolder, '/', Names{j}, '_ICPreg.vtk'), ' -mode ascii -ofile ',...
        strcat(NextIterationFolder, '/', Names{j}, '_ICPreg.vtk') ])
    end
    

end

end

