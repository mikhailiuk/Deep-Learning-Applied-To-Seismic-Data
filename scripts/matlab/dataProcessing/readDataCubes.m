function [depth,faceH,faceW,outputCube,inputCubeIdeal,inputCubeNoizy] = readDataCubes(pathPredictionFile,pathIdealFile,pathNoizyFile,whitenedFlag)
    %set format and set files to read
    formatSpec = '%f';
	filePredicted = fopen(pathPredictionFile,'r');
	fileIdeal = fopen(pathIdealFile,'r');
	fileNoizy =fopen(pathNoizyFile,'r');

    %read the data into variables
	dataNoizyInput = fscanf(fileNoizy,formatSpec);
	dataPredicted = fscanf(filePredicted,formatSpec);
	dataIdeal = fscanf(fileIdeal,formatSpec);
    
    %read the dimentions of the cube
	depth = dataPredicted(1,1);
	faceH = dataPredicted(2,1);
	faceW = dataPredicted(3,1);
	
    %alocate memory for the cubes of data
    outputCube = zeros(depth,faceH,faceW);
	inputCubeIdeal = zeros(depth,faceH,faceW);
	inputCubeNoizy = zeros(depth,faceH,faceW);
	count = 1;
    
    %iterate throught the data read from the file, writing information into 3D matrices
    for d=1:depth
        for h=1:faceH
            for w=1:faceW
                outputCube(d,h,w) = dataPredicted(count+3,1);
                inputCubeIdeal(d,h,w) = dataIdeal(count+3,1);
                inputCubeNoizy(d,h,w) = dataNoizyInput(count+3,1);  
                count = count+1;
            end
         end
     end
    
    %if the data was wightened before need to do some preprocessing before
    %we can visualise it, otherwise wight noize
	if whitenedFlag==1
        inputCubeNoizy(:,:,:) = patchPreproc(inputCubeNoizy);
        inputCubeIdeal(:,:,:) = patchPreproc(inputCubeIdeal);
        outputCube(:,:,:) = patchPreproc(outputCube);  
	end    
    
end
            