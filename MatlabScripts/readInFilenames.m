
function [Filenames, Name, ext] = readInFilenames(folder)
%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%

     Filenames = {};
     Name = {};
     ext = 0;
%read in filenames  
     listing = dir(strcat(folder, '/*.vtk')); % creates structure
     if isempty(listing)          
        listing = dir(strcat(folder, '/*.stl')); % creates structure
        if isempty(listing)
             return
        end
     end
   

    Filenames = {listing.name}'; % create cell with all available filenames

    [noOfFiles,n] = size(Filenames);
    
%     % create blank space for VMTK    
%     b = blanks(1);
%     
%     Filenames = filenames;
    
    
for k = 1:length(Filenames)

    [pathstr, name, ext] = fileparts(Filenames{k});
    
    Name{k} = name;
    
%%%EMI: fprintf('I have detected this extension of file: %s\n', ext)    
       
    % create blanks b before filenames for VMTK input
%     Name{k} = [b, Name{k}];
%     
%     Filenames{k} = [b, Filenames{k}];
      
    
end

  
   
   % check size
    [m n] = size(Name);

if m == 1
    
    Name = Name';   
    
end    

%%%%%%%% Copyright (C) Jan Bruse 2015 - jan.bruse@gmail.com %%%%%%%%%    
end