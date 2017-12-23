function [minCubeInverted] = cubeWhitening(cubeWhitened)
%function to undo the whitening following the inverted SVD procedure

    %load the precomputed matrices, used to whiten the data
    load('./additional/S.mat');
    load('./additional/meanX.mat');
    load('./additional/U.mat');

    %set variable for the dimentions of the whitened data
    faceH = 128;
    faceW = 128;
    faceZ = 64;
    
    %alocate memory
    X = zeros(64,128);
    start = 1;
    fin = 8;
    range = fin - start +1;
    rangeSqre = range*range;

    for kk = 1:faceZ
       %get one time slice
       Xtot=squeeze(cubeWhitened(kk,:,:));
       
       %split into patches
       res = mat2cell(Xtot,repmat(8,16,1),repmat(8,16,1));
       for ii = 1:16
           for jj = 1:16
               %convert a cell into matrix
               matTmp = cell2mat(res(ii,jj));
               for hh = 1:8
                    for tt=  1:8
                        %indeces
                        idxN = ((kk-1)*256+(ii-1)*16+jj);
                        idxD = (hh-1)*8+tt;
                        
                        %set Y
                        Y(idxD,idxN) = matTmp(hh,tt);
                    end
                end
            end
        end
    end

    %do the transformation following the SVD procedure
    inversionY = inv(U')*inv(inv(S))*Y;
    [D,N] = size(Y);
    
    %shift
    for ii=1:N
      XInvertion (:,ii)=inversionY(:,ii)+meanX';
    end

    %create a transformed cube (real, not whitened data)
    minCubeInverted =zeros(faceZ,faceH,faceW);

    
    for kk = 1:faceZ
       for ii = 1:16
           for jj = 1:16
               for hh = 1:8
                    for tt= 1:8
                        idxN = ((kk-1)*256+(ii-1)*16+jj);
                        idxD = (hh-1)*8+tt;
                        matTmp(hh,tt) = XInvertion(idxD,idxN);
                    end
               end
               minCubeInvertedCell(ii,jj) = {matTmp};
           end
       end
       minCubeInverted(kk,:,:)=cell2mat(minCubeInvertedCell);
    end

