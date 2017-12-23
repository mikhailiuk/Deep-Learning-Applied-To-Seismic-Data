function plotOutSubplot(titleSub,image,numberOfPlotsY,numberOfPlotsX,positionInSubplot,spaceBetweenPlots,flagHeatMap)
    %function to plot the individual images in a subplot
    
    %create tights subplot
    subplot_tight(numberOfPlotsY,numberOfPlotsX,positionInSubplot,spaceBetweenPlots);                        
    
    %create an image
    imagesc(image);  
    title(titleSub,'FontSize',9);      
    
    %depnding on whether need a heatMap parameters difer
    if flagHeatMap==1
        colorbar;
        colormap(jet);
    else
        colormap(gray);
    end
    daspect([1 1 1]);
    set(gca,'xtick',[]);
    set(gca,'xtick',[],'ytick',[])