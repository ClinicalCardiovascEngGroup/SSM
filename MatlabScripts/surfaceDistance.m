%function [ Distance ] = surfaceDistance( referenceFolder, referenceFilename, floatingFolder, floatingFilename )
%function [ Distance ] = surfaceDistance( IterationFolder, ReferenceFileREG, floatingFolder, floatingFilename )
function [ Distance ] = surfaceDistance( IterationFolder, ReferenceFileREG, OutputTemplateFile )
%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%
disp([ 9 9 'surfaceDistance.m']);

% Compute distance if dat file are not already there
if isempty(dir(strcat(IterationFolder,filesep,'TemplatePrototype_distance.vtk')))
    %{
     system(['vmtk vmtksurfacedistance', ' -ifile ', strcat(IterationFolder,filesep,'output',filesep, OutputTemplateFile), ...
          ' -rfile ', strcat(IterationFolder,filesep,'data',filesep, ReferenceFileREG), ' -distancearray ', ...
          'distancearray' , ' -distancevectorsarray ' , 'distancevectorsarray' , ...
          ' --pipe vmtksurfacewriter -mode ascii -ofile ', ...
          strcat(IterationFolder,filesep,'TemplatePrototype_distance.vtk')]);
      %}
    
      %%% EMI: I don't see the point of saving the Output.vtk - I should
      %%% probably remove this argument
       command1 =(['vmtksurfacedistance -ifile ',strcat(IterationFolder,filesep,'output',filesep, OutputTemplateFile),...
           ' -rfile ',strcat(IterationFolder,filesep,'data',filesep, ReferenceFileREG),...
           ' -distancearray DistanceMagn -distancevectorsarray DistanceVector'...
           ' -ofile ',strcat(IterationFolder,filesep,'Distance.dat'),...
           ' --pipe vmtksurfacewriter -mode ascii -ofile ',strcat(IterationFolder,filesep,'DistanceGeom.vtk')]);
       [status,cmdout] = system(command1,'-echo');
end

% Read dat files and compute distances
VMTKDistanceInfoFile = strcat(IterationFolder,filesep,'Distance.dat');
[Distance] =  importDATdistances(VMTKDistanceInfoFile);

end

