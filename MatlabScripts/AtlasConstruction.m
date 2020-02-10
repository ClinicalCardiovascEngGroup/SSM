%function [ atlasFolder ] = AtlasConstruction( folder, deformetrica_path )
%function [ atlasFolder, OutputTemplateFile ] = AtlasConstruction( IterationFolder, BasePath, ReferenceFileREG)
function [ OutputTemplateFile ] = AtlasConstruction( IterationFolder, IterationNumber, InitialGeomNames, BasePath, ReferenceFileName)

disp([ 9 9 'AtlasConstruction.m']);


%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%

% Function to perform atlas construction with the capability of 
% Deformetrica (www.deformetrica.org) with a set of
% registered input shapes. Parameters for the atlas are set
% by the user. 


     
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
     % Ask user to set atlas parameters 
     %disp('**************DEBUG***************')
     %fprintf('STEP 2 - a : Now entering setParamDiffeosSurface\n');
     %fprintf('The name of the ReferenceFileREG is : %s\n', ReferenceFileName);
     %disp('**************DEBUG***************')
     setParamDiffeosSurface( IterationFolder , FileNames);
end
     % Run sparse atlas 3
     %disp('**************DEBUG***************')
     %fprintf('STEP 2 - b : Now entering SparseAtlas3Matlab\n');
     %fprintf('The name of the ReferenceFileREG is : %s\n', ReferenceFileName);
     %disp('**************DEBUG***************')
     [OutputTemplateFile] = SparseAtlas3Matlab( IterationFolder, IterationNumber, FileNames, BasePath, ReferenceFileName); 
    
    
     
end
