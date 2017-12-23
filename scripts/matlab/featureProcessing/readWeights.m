function [weights,sizesWeights]=readWeights(path)
%function to read the weights file

    %open the file with weights
    fileID = fopen(path,'r');
    
    %read the first line, split and convert to double
    tline = fgetl(fileID);
    sizesWeights = strsplit(tline,' ');
    sizesWeights = str2double(sizesWeights);
                
    %alocate the memory for the weights
    %sizesWeights(1,1) - number of values in a line (number of connections)
    %sizesWeights(1,2) - number of line in the document (number of features)
    weights = zeros(sizesWeights(1,2),sizesWeights(1,1));
    
    %read the file line by line
    for i=1:sizesWeights(1,2)
         tline = fgetl(fileID);
         lineOfWeights = strsplit(tline,' ');
         lineOfWeights = str2double(lineOfWeights);
         weights(i,:) = lineOfWeights(1,1:sizesWeights(1,1));
    end
    
    %close the file            
    fclose(fileID);