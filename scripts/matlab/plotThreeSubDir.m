function [] = plotThreeSubDir(nameDir)
    %get all subdirectories of the current directory (different mini-batches)
    d = dir(nameDir);
    isub = [d(:).isdir];
    nameFoldsMB = {d(isub).name}';
    %run through all directories with minibatches
    for jj = 3:size(nameFoldsMB)
        
        %get names of the folders containing learning rates
        d = dir(char(strcat(nameDir,nameFoldsMB(jj))));
        isub = [d(:).isdir];
        nameFoldsLR = {d(isub).name}';
        
        %run through all directories containing learning rates
        for kk = 3:size(nameFoldsLR)
            
            d = dir(char(strcat(nameDir,nameFoldsMB(jj),'/',nameFoldsLR(kk))));
            isub = [d(:).isdir];
            nameFoldsHU = {d(isub).name}';
            
            %run through all directories containing hidden units
            for ii = 3:size(nameFoldsHU)
                
                %get path to the sub-directories with early stop and fully
                %trainined
                d = dir(char(strcat(nameDir,nameFoldsMB(jj),'/',nameFoldsLR(kk),'/',nameFoldsHU(ii))));
                isub = [d(:).isdir];
                nameFoldsDA = {d(isub).name}';
                
                %run through fully trained and early stop directories
                for tt = 3:size(nameFoldsDA)
                    
                    nameResDir = char(strcat(nameDir,nameFoldsMB(jj),'/',nameFoldsLR(kk),'/',nameFoldsHU(ii),'/',nameFoldsDA(tt))) 
                    plotSequentialData(nameResDir);
                    plotOutputTest(nameResDir);
                    plotWeightsOut(nameResDir);
                    plotStats(nameResDir);
                    
                    %closing all figures - including those that are hidden
                    close all hidden
                end
            end
        end
    end
end