function [] = generateOutputPlot(pathDir,k,errorPredictedIdealSlice,predictedSlice,idealSlice,FSIMSlice,noizySlice,errorPredictedNoizySlice,predictedShot,idealShot,FSIMShot,noizyShot,errorPredictedNoizyShot,errorPredictedIdealShot)

    %set the variables passed to plotOutSubplotFunction (here for clarity)
    flagHearMapTrue=1;
    flagHearMapFalse=0;

    %Create a new figure for output subplots
    figure
    set(gca,'xtick',[]);
    set(gca,'visible','off');
    set(gcf,'Visible','off');
    
    %Creat a subplot containing all the output
    plotOutSubplot('Predicted Slice',predictedSlice,2,3,1,.01,flagHearMapFalse);
    plotOutSubplot('Target Slice',idealSlice,2,3,2,.01,flagHearMapFalse);
    plotOutSubplot('Input Slice',noizySlice,2,3,3,.01,flagHearMapFalse);
    plotOutSubplot('Predicted Shot',predictedShot,2,3,4,.01,flagHearMapFalse);
    plotOutSubplot('Target Shot',idealShot,2,3,5,.01,flagHearMapFalse);
    plotOutSubplot('Input Shot',noizyShot,2,3,6,.01,flagHearMapFalse);
    pathSave = char(strcat(pathDir,int2str(k),'_target_vs_predicted.png'));

    saveas(gcf,[pathSave]);

    %create a new figure for error subplots
    figure
    set(gca,'xtick',[]);
    set(gca,'visible','off');
    set(gcf,'Visible','off');

    %create a subplot containing all the errors
    plotOutSubplot(strcat('FSIM: ',num2str(FSIMSlice)),errorPredictedIdealSlice,2,2,1,.1,flagHearMapTrue);
    plotOutSubplot('Error',errorPredictedNoizySlice,2,2,2,.1,flagHearMapTrue);
    plotOutSubplot(strcat('FSIM: ',num2str(FSIMShot)),errorPredictedIdealShot,2,2,3,.1,flagHearMapTrue);
    plotOutSubplot('Error',errorPredictedNoizyShot,2,2,4,.1,flagHearMapTrue);

    pathSave = char(strcat(pathDir,int2str(k),'_target_vs_predicted_errorMap.png'));
    saveas(gcf,[pathSave]);

end