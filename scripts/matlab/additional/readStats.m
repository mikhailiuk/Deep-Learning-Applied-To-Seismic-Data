function [stats,weightLayersSize]=readStats(path)
% A function to read the statistics about feature usage

    %open the file
    fileID = fopen(path,'r');
    
    %read the first line of the file (contains the sizes of the layers)
    tline = fgetl(fileID);
    
    %split the line by the spaces
    weightLayersSize = strsplit(tline,' ');
    
    %convert the str to double format
    weightLayersSize = str2double(weightLayersSize);
    
    %find the size of the first line (number of layers)
    [m,n] = size(weightLayersSize);
    
    %alocate space for the statistics
    stats = zeros(n,max(weightLayersSize));
    
    %run through all weight layers 
    for i=1:n
        
         %get line, split, convert and write to a matrix 
         tline = fgetl(fileID);
         line= strsplit(tline,' ');
         line = str2double(line);
         stats(i,1:weightLayersSize(1,i)) = line(1,1:weightLayersSize(1,i));
    end
       
    %close the file
    fclose(fileID);
end