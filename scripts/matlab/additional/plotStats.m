function []=plotStats(nameFolder)

        %path to a file with statistircs
        path = char(strcat(nameFolder,'/stats.dat'));
        
        %make sure that the file does not exist        
        if exist(path,'file')
            
            %read the statistics from a specified file
            [matrix,sizes]=readStats(path);
            
            
            [m,n]= size(sizes);
            
            %run through all weight layers
            for layer = 1:n
                    
                    %allocate the save path
                    pathsave = char(strcat(nameFolder,'/stats_',num2str(layer),'.png'));
                    
                    %find the size of the image to display and allocate the
                    %memory
                    numb_cols_disp = ceil(sqrt(sizes(1,layer)));
                    numb_rows_disp = numb_cols_disp;
                    features=zeros(numb_cols_disp,numb_rows_disp);
                    
                    %set count for the number 
                    count = 1;
                    
                    %run through all columns
                    for i=1:numb_cols_disp
                        
                        %run through all rows
                        for j=1:numb_rows_disp
                            
                            %if did not exceed the number of features in a
                            %layer
                            if count<=sizes(1,layer)
                                 features(i,j)=matrix(layer,count);
                            end
                            count=count+1;
                        end 
                    end

                    %plot stats
                    figure;
                    imagesc(features);
                    title('Features Statistics');  
                    set(gcf,'Visible','off');
                    set(gca,'xtick',[]);
                    set(gca,'ytick',[]);
                    colormap(jet);
                    daspect([1 1 1]);
                    colorbar;
                    caxis([0.0, 1.0])
                    saveas(gcf,[pathsave]);
                    
            end
      
        end
            
end
        