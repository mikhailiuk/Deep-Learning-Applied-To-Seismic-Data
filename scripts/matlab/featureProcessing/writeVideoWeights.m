function []=writeVideoWeights(pathDecoderWeights,counter,faceZ,faceH, faceW)
%   A function to create and write a wideo with weights
    if faceZ>1 && faceH>1 && faceW>1
        videoWeights(faceZ,pathDecoderWeights,counter,'slice');
        videoWeights(faceH,pathDecoderWeights,counter,'shotx');
        videoWeights(faceW,pathDecoderWeights,counter,'shoty');
    end
         
end