%Function to plot the weights
 
function [] = plotWeightsOut (nameFolder)

        %Directory with weights
        pathDirWeights = char(strcat(nameFolder,'/weights/'));
        
        %Derictory for Encoder weights
        pathEncoderWeights = char(strcat(pathDirWeights,'encoder/'));    
        
        %Directory for Decoder weigths
        pathDecoderWeights = char(strcat(pathDirWeights,'decoder/'));    

        
        %check whether the directory with weights exist
        if ~exist(pathDirWeights,'dir')

            %create required sub-directories
            mkdir(pathDirWeights);
            mkdir(pathEncoderWeights);
            mkdir(pathDecoderWeights);

            counterWeights = 0;
            totalWeights = 0;
            
            %read the first weights file (required to identify number of weights in total)
            path = char(strcat(nameFolder,'/weights',int2str(counterWeights),'.dat'));

            %count number of files
            while exist(path,'file')
                totalWeights = totalWeights+1;
                path = char(strcat(nameFolder,'/weights',int2str(totalWeights),'.dat'));
            end

            decoderNumberOfWeights = totalWeights/2;

            %plot the encoder weights
            [counterWeights,averageCorrEnc,featureStackEncoder] = plotWeightsEncoder(counterWeights,decoderNumberOfWeights,nameFolder);

            %plot the decoder weights
            [averageCorrDec,featureStackDecoder] = plotWeightsDecoder(counterWeights,totalWeights,nameFolder);

            %calculate the cross correlation between encoder and decoder
            crossCorrEncDec = plotCrossCorrelation(featureStackEncoder,featureStackDecoder,pathDirWeights,totalWeights);
            
            %set the values of correlation to write to a file
            avCor = [averageCorrEnc, averageCorrDec, crossCorrEncDec];
            pathCorSave = char(strcat(nameFolder,'/corrEncDec'));
            dlmwrite(pathCorSave,avCor,'delimiter', ' ' ,'precision', '%.4f');
        
        end

end

            
