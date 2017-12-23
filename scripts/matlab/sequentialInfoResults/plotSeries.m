function [] = plotSeries(xlabelName, ylabelName, pathFileSave, pathFile)

    
     formatSpec = '%f';
     
     %chech that the file has not yet been created and that the reference
     %file exists
     if ~exist(pathFileSave,'file') && exist(pathFile,'file')
         
        %open file, read it, close it 
        fileId = fopen(pathFile,'r');
        data = fscanf(fileId,formatSpec);
        fclose(fileId);
        
        %find the dimentions of the data
        [m,n]=size(data);
        
        %check that the data is not empty and plot
        if m>0
            figure;
            set(gcf,'Visible','off');
            set(gca,'visible','off');
            set(gca,'xtick',[]);
            plot(data(:,1));
            xlim([0 inf]);
            ylim([0 inf]);
            xlabel(xlabelName,'FontSize',18);
            ylabel(ylabelName,'FontSize',18);
            saveas(gcf,[pathFileSave]);
        end
        

     end

end