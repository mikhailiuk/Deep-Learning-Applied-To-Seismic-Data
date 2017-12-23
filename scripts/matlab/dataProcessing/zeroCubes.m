function [predictedSlice,idealSlice,noizySlice,errorPredictedIdealSlice,errorPredictedNoizeSlice,predictedShot,noizyShot,errorPredictedNoizeShot,idealShot,errorPredictedIdealShot] = zeroCubes(faceH,faceW,faceZ)
% function to allocate memory and set variables to zero

    predictedSlice=zeros(faceW,faceH);
	idealSlice=zeros(faceW,faceH);
	noizySlice=zeros(faceW,faceH);
	errorPredictedIdealSlice=zeros(faceW,faceH);
	errorPredictedNoizeSlice=zeros(faceW,faceH);
	predictedShot=zeros(faceZ,faceH);
	noizyShot=zeros(faceZ,faceH);
	errorPredictedNoizeShot=zeros(faceZ,faceH);
	idealShot=zeros(faceZ,faceH);
	errorPredictedIdealShot=zeros(faceZ,faceH);
    
end