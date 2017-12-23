function [] = plotOneSubDir(nameDir) 
%get all subdirectories of the current directory (different mini-batches)
    d = dir(nameDir);
    isub = [d(:).isdir];
    nameFoldsMB = {d(isub).name}';
    %run through all directories with minibatches
    for jj = 3:size(nameFoldsMB)
        
        %get names of the folders containing learning rates
        d = dir(char(strcat(nameDir,nameFoldsMB(jj))));
        isub = [d(:).isdir];
        nameFoldsDA = {d(isub).name}';
                for tt = 3:size(nameFoldsDA)
                    
                    nameResDir = char(strcat(nameDir,nameFoldsMB(jj),'/',nameFoldsDA(tt))) 
                    plotSequentialData(nameResDir);
                    plotOutputTest(nameResDir);
                    plotWeightsOut(nameResDir);
                    plotStats(nameResDir);
                    
                    %closing all figures - including those that are hidden
                    close all hidden
                end
    end
end