function [ param_folder, answer ] = setMomInXml( atlasfolder, folder, mom_file )
%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%


% Copy parameter files templates
system(['cp -r ', atlasfolder, '/param ', folder, '/']);

% Modify templates with new parameters
param_folder = strcat(folder, '/param');
model_xml = strcat(param_folder, '/model.xml');

% MODEL XML
% Read txt into cell A
fid = fopen(model_xml,'r');
i = 1;
tline = fgetl(fid);
A{i} = tline;
while ischar(tline)
    i = i+1;
    tline = fgetl(fid);
    A{i} = tline;
end
fclose(fid);

% momenta file
cell_to_change = A{23};
cell_to_change = strrep(cell_to_change, 'MOM', mom_file) ; %
A{23} = cell_to_change;

% Write cell A into txt
fid = fopen(model_xml, 'w');
for i = 1:numel(A)
    if A{i+1} == -1
        fprintf(fid,'%s', A{i});
        break
    else
        fprintf(fid,'%s\n', A{i});
    end
end
fclose(fid);


end
