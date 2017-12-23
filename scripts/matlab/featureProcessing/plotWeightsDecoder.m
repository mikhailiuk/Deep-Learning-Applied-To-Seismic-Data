function [averageCorrDec,featuresDecoder] = plotWeightsDecoder(counterWeights,totalWeights,nameFolder) 
    averageCorrDec = 0;
    
	%Directory for Decoder weigths
	pathDecoderWeights = char(strcat(nameFolder,'/weights/decoder/'));        
    
    %run through all layers of weights
    while counterWeights < totalWeights
        
        %get the path to weights file
        path = char(strcat(nameFolder,'/weights',int2str(counterWeights),'.dat'));

        %read the weights
        [weights,sizesW] = readWeights(path);

        %calculate plotting dimentions
        [faceZ,faceH,faceW]=setFaces(counterWeights, (totalWeights-1),sizesW);
        
        %transpose the weights since it is decoder
        weights = transpose(weights);
        
        %get the size of the weights (m - number of features)
        [numberOfFeatures,~]=size(weights);

        %set the number of columbs and rows to display in a subplot
        numb_cols_disp = ceil(sqrt(numberOfFeatures-1));
        numb_rows_disp = numb_cols_disp+1;

        %set the features array for a weight in a decoder (needed for correlation)
        featuresDecoder=zeros(numberOfFeatures,faceW*faceH*faceZ);
                
        %go through the features


        %plot slices in weights
        plotLayerOfWeights(pathDecoderWeights,'slice',weights,numberOfFeatures,counterWeights,numb_cols_disp,numb_rows_disp,faceH, faceW, faceZ,'z');
        
        %plot shot x in weights
        plotLayerOfWeights(pathDecoderWeights,'shotx',weights,numberOfFeatures,counterWeights,numb_cols_disp,numb_rows_disp,faceZ, faceW, faceH,'x');
        
        %plot shot y in weights
        plotLayerOfWeights(pathDecoderWeights,'shoty',weights,numberOfFeatures,counterWeights,numb_cols_disp,numb_rows_disp,faceZ, faceH, faceW,'y');
        
        for featureIdx=1:numberOfFeatures          
            %copy weights into the cross correlation array
            for i = 1:faceH
                for j = 1:faceW
                    for f = 1:faceZ
                        idx = (i-1)*faceW*faceZ+(j-1)*faceZ+f;
                        featuresDecoder(featureIdx,idx) = weights(featureIdx,idx);
                    end
                end
             end
            
        end

        %path to save correlation in the features
        pathsaveCor = char(strcat(pathDecoderWeights,'/correlation_',int2str(counterWeights),'.png'));
            
        %plot the correlation in the features
        averageCorrDec = averageCorrDec+plotCorrelation(featuresDecoder,pathsaveCor);
        
        writeVideoWeights(pathDecoderWeights,counterWeights,faceZ,faceH,faceW);

        counterWeights = counterWeights+1;             
    end 
        
	%get the average correlation
	averageCorrDec = averageCorrDec/(counterWeights-totalWeights/2);

end    
