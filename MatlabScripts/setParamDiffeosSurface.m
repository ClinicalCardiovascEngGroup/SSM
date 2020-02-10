%function [ param_folder, answer ] = setParamDiffeosSurface( folder )
function setParamDiffeosSurface( IterationFolder , FileNames)
disp([ 9 9 9 'setParamDiffeosSurface.m']);

%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%
%%%%%%%% Adapted by Emilie Sauvage 2019 for the new Deformetrica 4.1

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%% NEW VERSION with Deformertrica 4.1 %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ask parameters to be set
prompt = {'model-type:','noise-std:','kernel-width_geom:', 'kernel-width_template', 'save-every-n-iters (optional):'};
dlg_title = 'Enter parameters for Atlas Computation';
num_lines = 1;
def = {'DeterministicAtlas','10', '10', '40','5'};
answer = inputdlg(prompt,dlg_title,num_lines,def);


%%% User defined parameters %%%
%%********************************
Model_Type = answer{1};
Noise_Std = answer{2};
Kernel_Width_Geom = answer{3};
Kernel_Width_Template = answer{4};
Save_every_N_iters = answer{5};
%%********************************
StartFolder = pwd;
cd(IterationFolder);

%%% EMI: Encoding of the xml files
slCharacterEncoding('UTF-8');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Writing the file "model.xml" 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%%% EMI write the file in a particular folder!!!! %%%%
Model_file = fopen('model.xml','w','n','UTF-8');
fprintf(Model_file,'<?xml version="1.0"?>\n');
fprintf(Model_file,'<model>\n');
%fprintf('\r')
fprintf(Model_file,strcat('\t','<model-type>',Model_Type,'</model-type>\n'));
fprintf(Model_file,strcat('\t','<dimension>3</dimension>\n'));
fprintf(Model_file,strcat('\t','<template>\n'));
%%
fprintf(Model_file,strcat('\t','\t','<object id="Geom">\n'));
%%%
fprintf(Model_file,strcat('\t','\t','\t','<deformable-object-type>SurfaceMesh</deformable-object-type>\n'));
fprintf(Model_file,strcat('\t','\t','\t','<attachment-type>Varifold</attachment-type>\n'));
fprintf(Model_file,strcat('\t','\t','\t','<noise-std>',Noise_Std,'</noise-std>\n'));
fprintf(Model_file,strcat('\t','\t','\t','<kernel-type>keops</kernel-type>\n'));
fprintf(Model_file,strcat('\t','\t','\t','<kernel-width>',Kernel_Width_Geom,'</kernel-width>\n'));
fprintf(Model_file,strcat('\t','\t','\t','<filename>data/Prototype.vtk</filename>\n'));
%%
fprintf(Model_file,strcat('\t','\t','</object>\n'));
%
fprintf(Model_file,strcat('\t','</template>\n'));
fprintf(Model_file,strcat('\t','<deformation-parameters>\n'));
%%
fprintf(Model_file,strcat('\t','\t','<kernel-width>',Kernel_Width_Template,'</kernel-width>\n'));
fprintf(Model_file,strcat('\t','\t','<kernel-type>keops</kernel-type>\n'));
%%
fprintf(Model_file,strcat('\t','</deformation-parameters>\n'));
fprintf(Model_file,'</model>\n');
fclose(Model_file);
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Writing the file "optimization_parameters.xml" 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Opti_file = fopen('optimization_parameters.xml','w','n','UTF-8');
fprintf(Opti_file,'<?xml version="1.0"?>\n');
fprintf(Opti_file,'<optimization-parameters>\n');
%
fprintf(Opti_file,strcat('\t','<optimization-method-type>GradientAscent</optimization-method-type>\n'));
fprintf(Opti_file,strcat('\t','<initial-step-size>0.1</initial-step-size>\n'));
fprintf(Opti_file,strcat('\t','<max-iterations>100</max-iterations>\n'));
fprintf(Opti_file,strcat('\t','<max-line-search-iterations>30</max-line-search-iterations>\n'));
fprintf(Opti_file,strcat('\t','<convergence-tolerance>1e-4</convergence-tolerance>\n'));
fprintf(Opti_file,strcat('\t','<freeze-template>Off</freeze-template>\n'));
fprintf(Opti_file,strcat('\t','<freeze-control-points>On</freeze-control-points>\n'));
fprintf(Opti_file,strcat('\t','<save-every-n-iters>',Save_every_N_iters,'</save-every-n-iters>\n'));
fprintf(Opti_file,strcat('\t','<memory-length>10</memory-length>\n'));
fprintf(Opti_file,strcat('\t','<use-cuda>On</use-cuda>\n'));
%
fprintf(Opti_file,'</optimization-parameters>\n');
fclose(Opti_file);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%% Writing the file "data_set.xml" 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Data_file = fopen('data_set.xml','w','n','UTF-8');
fprintf(Data_file,'<?xml version="1.0"?>\n');
fprintf(Data_file,'<data-set deformetrica-min-version="4.1">\n');
%
for j = 1:length(FileNames)
    fprintf(Data_file,strcat('\t','<subject id="0',num2str(j),'_ICPreg">\n'));
    fprintf(Data_file,strcat('\t','\t','<visit id="experiment">\n'));
    fprintf(Data_file,strcat('\t','\t','\t','<filename object_id="Geom">data/',FileNames{j},'</filename>\n'));
    fprintf(Data_file,strcat('\t','\t','</visit>\n'));
    fprintf(Data_file,strcat('\t','</subject>\n'));
end
%{
fprintf(Opti_file,strcat('\t','<subject id="02_ICPreg">\n'));
fprintf(Opti_file,strcat('\t','\t','<visit id="experiment">\n'));
fprintf(Opti_file,strcat('\t','\t','\t','<filename object_id="Geom">data/CTRL_02_R_ICPreg.vtk</filename>\n'));
fprintf(Opti_file,strcat('\t','\t','</visit>\n'));
fprintf(Opti_file,strcat('\t','</subject>\n'));
%
fprintf(Opti_file,strcat('\t','<subject id="03_ICPreg">\n'));
fprintf(Opti_file,strcat('\t','\t','<visit id="experiment">\n'));
fprintf(Opti_file,strcat('\t','\t','\t','<filename object_id="Geom">data/CTRL_03_R_ICPreg.vtk</filename>\n'));
fprintf(Opti_file,strcat('\t','\t','</visit>\n'));
fprintf(Opti_file,strcat('\t','</subject>\n'));
%}
fprintf(Data_file,'</data-set>\n');
fclose(Data_file);

cd(StartFolder);


end

