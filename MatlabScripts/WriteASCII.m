function [] = WriteASCII(folder, Filenames, Name, ext)
%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%

%read in filenames  
   
if strcmp(ext, '.vtk')
    disp('VTK have been detected');
    FileType = 1;
else
   disp('The files detected here are not VTK files - but have a different extension');
   if strcmp(ext, '.stl')
       FileType = 2;
   else 
       disp ('Error - enter either stl or vtk!')
   end
end
    
    switch FileType
        
        %%% EMI: case with VTK files
                
        case 1
            disp('I am entering the case 1 - where VTK have been detected');

        for j = 1:length(Filenames) 

        system(['vmtksurfacewriter -ifile ', strcat(folder, '/', Filenames{j}), ' -mode ascii -ofile ', strcat(folder, '/', Name{j}), '.vtk'])

        end
        
       %%% EMI: case with STL files

        case 2
            disp('I am entering the case 2 - where STL have been detected');
        for j = 1:length(Filenames) 

        system(['vmtksurfacewriter -ifile ', strcat(folder, '/', Filenames{j}), ' -mode ascii -ofile ', strcat(folder, '/', Name{j}), '.stl'])

        end                       
        
    end
    

%disp ('%%%% ----------------  writing as ASCII done  --------------- %%%%')


%%%%%%%% Copyright (C) Jan Bruse 2015 - jan.bruse@gmail.com %%%%%%%%%    
end