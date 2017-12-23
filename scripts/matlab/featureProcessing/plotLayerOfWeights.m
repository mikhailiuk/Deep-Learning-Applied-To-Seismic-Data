function [] = plotLayerOfWeights(pathEncoderWeights,name,weights,numberOfFeatures,counterWeights,numb_cols_disp,numb_rows_disp,size1, size2, size3,flag)
	if size1>1 && size2>1
        for frame  = 1:size3
            figure;
            set(gcf,'Visible','off');
            set(gca,'visible','off');
            set(gca,'xtick',[]);
            feature=zeros(size2,size1);
            for featureIdx = 1:numberOfFeatures  
                for i=1:size2
                    for j=1:size1
                        if flag=='z'
                            feature(i,j)=weights(featureIdx,(frame-1)*size1*size2 + (i-1)*size1 + j);
                        elseif flag =='x'
                            feature(j,i)=weights(featureIdx,(frame-1)*size2 + (j-1)*size2*size3 + i);
                        else
                            feature(j,i)=weights(featureIdx,frame + (j-1)*size2*size3 + (i-1)*size3);
                        end
                    end 
                end
                plotOutSubplot('',feature,numb_rows_disp,numb_cols_disp,featureIdx,[0.01,0.01],0);
            end
            pathSave = char(strcat(pathEncoderWeights,'/weights_',name,'_',int2str(counterWeights),'_',int2str(frame),'.png'));
            saveas(gcf,[pathSave]);
        end
	end
end