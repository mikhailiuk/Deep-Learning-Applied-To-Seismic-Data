function [] = plotNoSubDir(nameDir)
    d = dir(nameDir);
    isub = [d(:).isdir];
    nameFolds = {d(isub).name}';

    for ii = 3:size(nameFolds)
        [m,~]=size(nameFolds);
        sprintf('Current directory is: %s. Left %d, out of %d ', char(nameFolds(ii)),m-ii, m-2)
        name = char(strcat(nameDir,nameFolds(ii)));
        plotSequentialData(name);
        plotOutputTest(name);
        plotWeightsOut(name);
        plotStats(name);
    end
end
