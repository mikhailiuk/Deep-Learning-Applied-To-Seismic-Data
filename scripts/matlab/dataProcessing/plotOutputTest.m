%Function to plot images of the test sequence and next to reconstuction
function [] = plotOutputTest (nameFolder)
        pathDir = char(strcat(nameFolder,'/outputtest/'));
        FSimTotShot = 0;
        FSimTotSlice = 0;
        numberOfImagesToPlot = 10;
        
        zeroSlice = 0;
        zeroShot = 0;
        
        pathPredictionFile = char(strcat(nameFolder,'/testoutCube.dat'));
        pathIdealFile = char(strcat(nameFolder,'/testinCubeClean.dat'));
        pathNoizyFile = char(strcat(nameFolder,'/testinCubeNoize.dat'));
        
        whitenedFlag = 0;
        if exist(pathPredictionFile,'file') && exist(pathIdealFile,'file') && exist(pathNoizyFile,'file') && ~exist(pathDir,'file')

            mkdir(pathDir);
            
            %read the cubes of data
            [faceZ,faceH,faceW,outputCube,inputCubeIdeal,inputCubeNoizy] = readDataCubes(pathPredictionFile,pathIdealFile,pathNoizyFile,whitenedFlag);

            if faceZ <numberOfImagesToPlot
              numberOfImagesToPlot = faceZ;
            end
            %allocate memory for the slices and shots
            [predictedSlice,idealSlice,noizySlice,errorPredictedIdealSlice,errorPredictedNoizeSlice,predictedShot,noizyShot,errorPredictedNoizeShot,idealShot,errorPredictedIdealShot] = zeroCubes(faceH,faceW,faceZ);

            %if the cube is not empty
            if faceZ>0
                
                
                for k=1:numberOfImagesToPlot
                        
                        %go through the cube and plot slices
                        for i=1:faceW
                           for j=1:faceH
                               predictedSlice(i,j) = outputCube(k,j,i);
                               idealSlice(i,j) = inputCubeIdeal(k,j,i);
                               noizySlice(i,j) = inputCubeNoizy(k,j,i);
                               errorPredictedIdealSlice(i,j) = abs(outputCube(k,j,i)-inputCubeIdeal(k,j,i));
                               errorPredictedNoizeSlice(i,j) = abs(outputCube(k,j,i)-inputCubeNoizy(k,j,i));
                           end
                        end

                        %go through the cube and plot shots
                        for i=1:faceZ
                           for j=1:faceH
                               predictedShot(i,j)=outputCube(i,j,k);
                               idealShot(i,j) = inputCubeIdeal(i,j,k);
                               errorPredictedIdealShot(i,j) = abs(outputCube(i,j,k)-inputCubeIdeal(i,j,k));
                               noizyShot(i,j) = inputCubeNoizy(i,j,k);
                               errorPredictedNoizeShot(i,j) = abs(outputCube(i,j,k)-inputCubeNoizy(i,j,k));
                           end
                        end

                        %calculate the FSIM for current shot and slice
                        [FSIMShot,FSIMSlice,zeroShot,zeroSlice]= calculateFSIM(idealSlice,predictedSlice,predictedShot,idealShot,zeroShot,zeroSlice);
                        
                        %update the total values
                        FSimTotSlice = FSimTotSlice+FSIMSlice;
                    	FSimTotShot = FSimTotShot+FSIMShot;
                        
                        %generate the output plot
                        generateOutputPlot(pathDir,k,errorPredictedIdealSlice,predictedSlice,idealSlice,FSIMSlice,noizySlice,errorPredictedNoizeSlice,predictedShot,idealShot,FSIMShot,noizyShot,errorPredictedNoizeShot,errorPredictedIdealShot);

                end
            end
            
            %write FSIM results into a file 
            fileIDFsim = fopen(char(strcat(nameFolder,'/fsim.txt')),'w');
            FSimAvSlice = FSimTotSlice/(numberOfImagesToPlot-zeroSlice);
            FSimAvShot = FSimTotShot/(numberOfImagesToPlot-zeroShot);
            fprintf(fileIDFsim,'%5f %5f\n',[FSimAvSlice,FSimAvShot]);
            fclose(fileIDFsim);
        end        
end