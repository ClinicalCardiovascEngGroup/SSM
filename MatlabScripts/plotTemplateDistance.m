function [ templateDistance ] = plotTemplateDistance( BasePath )
%function [ templateDistance ] = plotTemplateDistance( BasePath, IterationFolder )


%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

   %% Get list of iteration folders
     listing = dir(strcat(BasePath, '/iteration*'));
     if isempty(listing)
          return
     end
     templateDistance = [];
     
     StartFolder = pwd;
     %% Get value of distance for each iteration
     noOfIterations = 1:length(listing);
     
     %for i=1:length(listing)
     for i=2:length(listing)
          %this_iteration_folder = listing(i).name;
          %%%current iteration folder
          this_iteration_folder = strcat(BasePath, filesep, 'iteration_',num2str(i));
          cd(this_iteration_folder);
          %command0 = 'pwd';
          %[status,cmdout] = system(command0,'-echo');

          this_itration_filedistance = dir(strcat(this_iteration_folder,filesep,'Registration_DistanceInfo_to*'));
          this_itration_filedistance = this_itration_filedistance.name;
          load(this_itration_filedistance);
          templateDistance(i) = distanceToPreviousTemplate.D_Average;
     end
    cd(StartFolder);
    
     %% Plot graph
     f = figure();
     plot(noOfIterations, templateDistance)
     xlabel('Number of iterations')
     ylabel('Distance to previous template [mm]')
     grid on
     ax = gca;
     ax.FontSize = 16;
     saveas(f, strcat(BasePath, '/GPAconvergence_templateDistance.png'), 'png');
end

