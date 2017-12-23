function [averageCorr] = plotCorrelation(featureStack,pathSaveCor,m)
    %function to plot the correlation in the encoder and the decoder
    
    [m,~]=size(featureStack);
    %allocate memory for the picture to plot 
    pictureCor = zeros(m,m);
    
    %run through columns and rows
	for i = 1:m
        for j = 1:m
            %get the vectors containing features
        	v1 = featureStack(i,:);
         	v2 = featureStack(j,:);
          	den = norm(v1)*norm(v2);
         	num = (v1*v2');
            %set the picture to the scaled dot product
         	pictureCor(i,j) = num/den;
        end
	end
    
    %average correlation in the features is the sum of the cross
    %correlations devided by the number of permutations
    averageCorr = (sum(sum(abs(pictureCor)))-trace(pictureCor))/(m*(m-1));
    
    %save the image
    figure;
	imagesc(pictureCor); 
	title('Features Correlation');  
	set(gcf,'Visible','off');
	set(gca,'xtick',[]);
	set(gca,'ytick',[]);
	colormap(jet);
	daspect([1 1 1]);
	colorbar;
    caxis([-1.0, 1.0]);
	saveas(gcf,[pathSaveCor]);