
%function [DATdata, Distance] =  importDATdistances(VMTKDistanceInfoFile)
function [Distance] =  importDATdistances(VMTKDistanceInfoFile)
disp([ 9 9 9 'importDATdistances.m']);
%%%%%%%% Copyright (C) Jan Bruse 2017 - jan.bruse@gmail.com %%%%%%%%%
%%%%%%%% Copyright (C) Benedetta Biffi 2017- benedetta.biffi89@gmail.com %%%%%%%%%


% Import data from text file.

%%% EMI: the delimiter between the columns is a space
delimiter = ' ';
%%% EMI: There is 1 line of header to skip in the *.dat file
startRow = 1;

% Format string for each line of text:
%%% EMI: The *.dat file has 7 columns
formatSpec = '%f %f %f %f %f %f %f';


% initialisation

DATdata = {};
Distance.D_Maximum = {};
Distance.D_Minimum = {};
Distance.D_Average = {};
Distance.D_Median = {};
Distance.D_StandardDev = {};
%%% EMI: Interquartile range
Distance.D_IQR = {};
%%% EMI: Coefficient of variations
Distance.D_CV = {};

%% Open the text file.

%%% EMI: Count the number of lines
%%% EMI: This is not an essential function
%%% EMI: Counting the lines doesn't bring anything
fid = fopen(VMTKDistanceInfoFile);
disp(' I manage to open the file');
Nlines = 0;
tline = fgetl(fid);
while ischar(tline)
  tline = fgetl(fid);
  Nlines = Nlines+1;
end
fprintf('My file contains : %d lines \n', Nlines);
fclose(fid); 

fid = fopen(VMTKDistanceInfoFile);
C = textscan(fid,formatSpec,'Delimiter', delimiter, 'Headerlines',1);
%DATdata = cell2mat(textscan(fid, formatSpec, Nlines-1, 'HeaderLines', 1))
fclose(fid);
X = C{1};
Y = C{2};
Z = C{3};
DistanceMagn = C{4};
DistanceVector0 = C{5};
DistanceVector0 = C{6};
DistanceVector0 = C{7};


%% calculate relevant geometric data
%%% EMI: here are computed common statistical parameters on the distance
%%% magnitude field
Distance.D_Maximum = max(DistanceMagn);
Distance.D_Minimum = min(DistanceMagn);
Distance.D_Average = mean(DistanceMagn);
Distance.D_Median = median(DistanceMagn);
Distance.D_StandardDev = std(DistanceMagn);
%%% EMI: Interquartile range
Distance.D_InterQuartRange = iqr(DistanceMagn);
%%% EMI: Coefficient of variations
Distance.D_CoefVariation = (Distance.D_StandardDev)/(Distance.D_Average);
     

%% Clear temporary variables
% clearvars filename delimiter startRow formatSpec fid dataArray ans i j;

%%%%%%%% Copyright (C) Jan Bruse 2015 - jan.bruse@gmail.com %%%%%%%%%

%}
end