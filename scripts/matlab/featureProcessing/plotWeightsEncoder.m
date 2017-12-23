function [counterWeights,averageCorrEnc,featureStackEncoder] = plotWeightsEncoder(counterWeights,decoderNumberOfWeights,nameFolder)
    
    averageCorrEnc = 0;
	
    %Derictory for Encoder weights
	pathEncoderWeights = char(strcat(nameFolder,'/weights/encoder/'));        
    
    %run through all layers of weights
    while counterWeights < decoderNumberOfWeights
            
                %get the path to weights file
                path = char(strcat(nameFolder,'/weights',int2str(counterWeights),'.dat'));

                %read the weights
                [weights,sizesW] = readWeights(path);

                %calculate plotting dimentions
                [faceZ,faceH,faceW]=setFaces(counterWeights,0,sizesW);

                %get the size of the weights (m - number of features)
                [numberOfFeatures,~]=size(weights);

                %set the number of columbs and rows to display in a subplot
                numb_cols_disp = ceil(sqrt(numberOfFeatures));
                numb_rows_disp = numb_cols_disp;
                
                %set the features array for a weight in a decoder (needed for correlation)
                featureStackEncoder=zeros(numberOfFeatures,faceW*faceH*faceZ);

                %plot slices in weights
                plotLayerOfWeights(pathEncoderWeights,'slice',weights,numberOfFeatures,counterWeights,numb_cols_disp,numb_rows_disp,faceH, faceW, faceZ,'z');
                
                %plot shot x in weights
                plotLayerOfWeights(pathEncoderWeights,'shotx',weights,numberOfFeatures,counterWeights,numb_cols_disp,numb_rows_disp,faceZ, faceW, faceH,'x');
                
                %plot shot y in weights
                plotLayerOfWeights(pathEncoderWeights,'shoty',weights,numberOfFeatures,counterWeights,numb_cols_disp,numb_rows_disp,faceZ, faceH, faceW,'y');
                
                for featureIdx = 1:numberOfFeatures     
                    %copy weights into the cross correlation array
                    for i = 1:faceH
                        for j = 1:faceW
                            for f = 1:faceZ
                                idx = (i-1)*faceW*faceZ+(j-1)*faceZ+f;
                                featureStackEncoder(featureIdx,idx) = weights(featureIdx,idx);
                            end
                        end
                    end                    
                end
                
                %path to save correlation in the features
                pathsaveCor = char(strcat(pathEncoderWeights,'/correlation_',int2str(counterWeights),'.png'));
                
                %plot the correlation in the features
                averageCorrEnc = averageCorrEnc + plotCorrelation(featureStackEncoder,pathsaveCor);
                counterWeights = counterWeights+1;  
    end
    
	%get the average correlation
	averageCorrEnc = averageCorrEnc/counterWeights;

end