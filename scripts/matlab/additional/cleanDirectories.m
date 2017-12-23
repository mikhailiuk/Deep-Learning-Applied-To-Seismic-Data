    %clear all the variables before executing the code
    clear
    
    %name the directory to clean
    nameDir = '/local/data/public/am2442/seismic/output/results/shot/tanh/';
    
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
                    
                    %get path to the current folder containing results
                    nameResultFold = char(strcat(nameDir,nameFoldsMB(jj),'/',nameFoldsLR(kk),'/',nameFoldsHU(ii),'/',nameFoldsDA(tt)));
                    
                    %name of the folder with weights
                    nameWeights = char(strcat(nameResultFold,'/weights'));
                    
                    %name of the folder with output
                    nameOutputTest = char(strcat(nameResultFold,'/outputtest'));
                    
                    %all PNG files
                    namePNG = char(strcat(nameResultFold,'/*.png'));
                    
                    %name of the file containing correlation of the encoder
                    %and decoder
                    nameCorrEncDec = char(strcat(nameResultFold,'/corrEncDec'));
                    
                    %name of the FSIM file
                    nameFSIM = char(strcat(nameResultFold,'/fsim.txt'));
                    
                    %delete everything created by matlab
                    if exist(nameWeights)
                        rmdir(nameWeights, 's');
                    end
                    if exist(nameOutputTest)
                        rmdir(nameOutputTest, 's');
                    end
                    delete(namePNG);
                    delete(nameCorrEncDec);
                    delete(nameFSIM);
                    
                end
            end
        end
    end