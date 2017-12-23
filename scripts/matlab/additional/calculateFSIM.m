function [FSIMShot,FSIMSlice,zeroShot,zeroSlice]= calculateFSIM(idealSlice,predictedSlice,predictedShot,idealShot,zeroShot,zeroSlice)
%function to get the FSIM for a slice and a shot and update the counters

    %if at least one element is not 0 (completely empty slices can be iftesting not on the full data set)
    if (any(any(idealSlice)) > 0 )
        
        %calculate the FSIM of the slice
        [FSIMSlice,~] =FeatureSIM(mat2gray(idealSlice),mat2gray(predictedSlice));
    else
        %set FSIM to 0
        FSIMSlice = 0;
        
        %update counter of completele 0 slices
        zeroSlice = zeroSlice + 1;
    end
    
    %if at least one element is not 0
    if (any(any(idealShot)) >  0)
        
        %calculate FSIM for the shot
        [FSIMShot,~] =FeatureSIM(mat2gray(idealShot),mat2gray(predictedShot));
    else
        FSIMShot = 0;
        zeroShot = zeroShot+1;
    end
                       
end