function [crossCorr] = plotCrossCorrelation(featureStackEncoder,featureStackDecoder,pathDirWeights,totalWeights)
% A function to plot cross correlation between the encoder and the decoder:
% the output is stored in the head of the weights directory

    % Cross correlation is set to 0
    crossCorr = 0;
    
    [n,~]=size(featureStackDecoder);
    % If the network is single layer
	if totalWeights==2
        
        % Allocate space for the image
        pictureCrossCor = zeros((n-1),(n-1));

        %run through all the fetures in the encoder
        for i = 1:(n-1)
            
            %run through all the features in the decoder
            for j = 1:(n-1)
                
                %get a vector (feature) from the encoder
                v1 = featureStackEncoder(i,:);
                
                %get a vectore (feature) from the decoder
                v2 = featureStackDecoder(j,:);
                
                %calculate the scalling factor for the cross correlation of the features
                den = norm(v1)*norm(v2);
                
                %get the dot product (covariance)
                num = (v1*v2');
                
                %set the picture to the scaled dot product of the features
                pictureCrossCor(i,j) = num/den;
            end
        end
        
        %set cross correlation the average of the trace of the cross
        %correlation matrix (trace is the similarity between corresponding
        %features in the decoder and the encoder)
        crossCorr = trace(abs(pictureCrossCor))/(n-1);
        
        %plot the result
        figure;
        imagesc(pictureCrossCor); 
        title('Features Cross Correlation');  
        set(gcf,'Visible','off');
        set(gca,'xtick',[]);
        set(gca,'ytick',[]);
        colormap(jet);
        daspect([1 1 1]);
        colorbar;
        caxis([-1.0, 1.0]);
        pathsaveCorBoth = char(strcat(pathDirWeights,'/crossCorrelation.png'));
        saveas(gcf,[pathsaveCorBoth]);

    end
end