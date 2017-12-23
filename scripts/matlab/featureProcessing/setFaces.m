function [faceZt,faceHt,faceWt]=setFaces(counterWeights, condition,sizesFeatures)
%function to set the sizes of plot dimentions for features

    %condition: first layer, last layer, layer in the middle
    if counterWeights == condition
        %satisfied if the first or the last layer
        
        %set first from the weights.dat file
        faceZt = sizesFeatures(1,3);
        faceHt = sizesFeatures(1,4);
        faceWt = sizesFeatures(1,5);
        
    else 
        %satisfied if layer in the middle
        
        %make sure that plots are square
        faceHt=floor(sqrt(sizesFeatures(1,1)-1));
        faceWt=floor(sqrt(sizesFeatures(1,1)-1));            
        faceZt=1;
	end
end