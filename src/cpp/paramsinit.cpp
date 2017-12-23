#include "../headers/ParamsInit.h"

void ParamsInit::updatePatchSize(){
        
        // If augmenting the patch
        if (augment > 0){

                // if patches are greater than 1, then make the size larger, otherwise leave
                patchX = (patchX > 1) ? patchX + 2*augment : patchX; 
                patchY = (patchY > 1) ? patchY + 2*augment : patchY; 
                patchZ = (patchZ > 1) ? patchZ + 2*augment : patchZ;


                // Update the input and output layer of the network according to changes
                layersVec[0] = patchX*patchY*patchZ;
                layersVec[numbLayers-1] = patchX*patchY*patchZ;

                // If patches are greater than 1 set the shift to the initial size of the patch, or leave otherwise
                shiftXTest         = (patchX > 1) ? (patchX - 2*augment) : shiftXTest;        
                shiftYTest         = (patchY > 1) ? (patchY - 2*augment) : shiftYTest;  
                shiftZTest         = (patchZ > 1) ? (patchZ - 2*augment) : shiftZTest;  
                shiftXTrain        = (patchX > 1) ? (patchX - 2*augment) : shiftXTrain;        
                shiftYTrain        = (patchY > 1) ? (patchY - 2*augment) : shiftYTrain;  
                shiftZTrain        = (patchZ > 1) ? (patchZ - 2*augment) : shiftZTrain;
        }


}


template <typename T>
bool ParamsInit::range(T const value, T const lower, T const upper){
        if (value>lower && value<upper){
                return true;
        }
        return false;
}

void ParamsInit::sanityCheck(){

        // Number of validation iterations must be less than the number of test iterations 
        // This is so because the maximum amount of data read is in the test cube
        if (numbItValidation>numbItTest){
                std::cout<<"Number of iterations validations should be less than or equal to number of iterations test\n";
                exit(0);
        }


        // The upper limit must be larger than the lower
        if (learningRateUpper<learningRateLower){
                std::cout<<"'learningRateUpper' must be less than 'learningRateLower'\n";
                exit(0);
        }

        // The size of the mini-batch must be smaller than the number of iterations
        if (miniBatch>numbItTrain){
                std::cout<<"'miniBatch' must be less than or equal to 'numbItTrain'\n";
                exit(0);
        }

        // Number of layers must be uneven if the weight freezing is used 
        if (numbLayers%2==0 && weightFreezeFlag!=0){
                std::cout<<"Cannot freeze weights in a neural network with even number of layers\n";
                exit(0);
        }

        // Learning rate cannot exceed 1
        if (learningRateUpper>=1.0){
                std::cout<<"'learningRate' cannot be more than 0.99999\n";
                exit(0);
        }
/*
        if(freezeFractionEpochs>=1.0 || freezeFractionEpochs<0){
                std::cout<<"'freezeFractionEpochs' ranges between 0 and 0.999\n";
                exit(0);
        }


        if (!range(lambda, -0.0001, 1.0)){
                std::cout<<"'lambda' ranges between 0.0 and 0.999\n";
                exit(0);
        }

        if(!range(learningRate, -0.0001, 1.0)){
                std::cout<<"'learningRate' ranges between 0.0 and 0.999\n";
                exit(0);
        }

        if(!range(learningRateLower, -0.0001, 1.0)){
                std::cout<<"'learningRateLower' ranges between 0.0 and 0.999\n";
                exit(0);
        }

        if(!range(annealing, -0.0001, 1.0)){
                std::cout<<"'annealing' ranges between 0.0 and 0.999\n";
                exit(0);
        }

        if(randomFlag!=1.0){
                if(randomFlag!=0.0){
                        std::cout<<"'randomFlag' is either 0.0 or 1.0\n";
                        exit(0);
                }
        }


        if(curriculum!=1){
                if(curriculum!=0){
                        std::cout<<"'curriculum' is either 0 or 1\n";
                        exit(0);
                }
        }


        if(!range(huberDelta, -0.0001, 1.0)){
                std::cout<<"'huberDelta' ranges between 0 and 0.999\n";
                exit(0);
        }

        if(!range(momentum, -0.0001, 1.0)){
                std::cout<<"'momentum' ranges between 0 and 0.999\n";
                exit(0);
        }


        if(!range(adaGrad, -0.0001, 1.0)){
                std::cout<<"'adaGrad' ranges between 0 and 0.999\n";
                exit(0);
        }

        if(!range(dropOut, -0.0001, 1.0)){
                std::cout<<"'dropOut' ranges between 0 and 0.999\n";
                exit(0);
        }


        if(!range(sparse, -0.0001, 1.0)){
                std::cout<<"'sparse' ranges between 0 and 0.999\n";
                exit(0);
        }


        if(!range(sparsityParameter, -0.0001, 1.0)){
                std::cout<<"'sparsityParameter' ranges between 0 and 0.999\n";
                exit(0);
        }
*/

}


