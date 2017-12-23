%function to plot the error from the training session
function [] = plotSequentialData (nameFolder)

        %plot the learning rate
        pathFile = char(strcat(nameFolder,'/learning.dat'));
        pathSaveFile = char(strcat(nameFolder,'/learning.png'));
        plotSeries('Iteration', 'Learning rate', pathSaveFile, pathFile);
        
        %plot the error train
        pathFile = char(strcat(nameFolder,'/errorTrain.dat'));
        pathSaveFile = char(strcat(nameFolder,'/errorTrain.png'));
        plotSeries('Iteration', 'Error Train', pathSaveFile, pathFile);

        %plot the validation error
        pathFile = char(strcat(nameFolder,'/errorValidation.dat'));
        pathSaveFile = char(strcat(nameFolder,'/errorValidation.png'));
        plotSeries('Iteration', 'Error Validation', pathSaveFile, pathFile);

        %plot the ratio
        pathFile = char(strcat(nameFolder,'/ratio.dat'));
        pathSaveFile = char(strcat(nameFolder,'/ratio.png'));
        plotSeries('Iteration', 'Validation ratio', pathSaveFile, pathFile);
        
end