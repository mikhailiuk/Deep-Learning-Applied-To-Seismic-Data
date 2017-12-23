#ifndef PARAMSINIT_H
#define PARAMSINIT_H

#include <stdlib.h>
#include <string.h> 
#include <vector>
#include <iostream>

using std::vector;
using std::string;

//template<typename T>

/*! \class ParamsInit
    \brief Used to store parameters read in Initialiser
        
*/
class ParamsInit{
private:

public:
        template<typename T>
        bool range(T const value, T const lower, T const upper);

        //! A function to alter the patch size if it is augmented
        void updatePatchSize();

        //! Check the input parameters
        void sanityCheck();

        //! Lambda penalty for the contractive autoencoder
        double lambda;

        //! The learning rate
        double learningRate;

        //! The lower limit of the learning rate
        double learningRateUpper;

        //! The upper limit of the learning rate
        double learningRateLower;

        //! Annealing
        double annealing;

        //! Bias
        double bias;

        //! Randomly set the weight inputs or not
        double randomFlag;

        //! Value of Delta in huberDelta
        double huberDelta;

        //! Momentum - ranges from [0;1] 
        double momentum;

        //! AdaGrad - ranges from [0;1]
        double adaGrad;

        //! The scaling factor for the weights
        double weightMagnitude;

        //! The drop out rate [0;1] 
        double dropOut;

        //! The scaling factor of the input
        double inputScale;

        //! The maximum value of the output neurons expected
        double sparsityParameter;

        //! Magnitude of the penalty
        double sparse;

        //! Mini-batch
        int miniBatch;

        //! Whether to use or not the curriculum learning
        int curriculum;

        //! Shuffle the data or not
        int shuffleFlag;

        //! The number of layers in the neural network
        int numbLayers;

        //! The number of training iterations
        int numbItTrain;

        //! The number of test iterations
        int numbItTest;

        //! The number of epochs
        int numbEpoches;

        //! The scaling factor for the weights
        int weightsInitFlag;

        //! The augmentation size of the patch
        int augment;

        //! Objective function
        int objective;

        //! Size of patch x
        int patchX;

        //! Size of patch y
        int patchY;

        //! Size of patch z
        int patchZ;

        //! Shift in X direction test
        int shiftXTest;

        //! Shift in Y direction test
        int shiftYTest;

        //! Shift in Z direction test
        int shiftZTest;

        //! Shift in X direction train
        int shiftXTrain;

        //! Shift in Y direction train
        int shiftYTrain;

        //! Shift in Z direction train
        int shiftZTrain;

        //! Flag mask validation, if 1 - there is mask
        int maskFlagValidation;

        //! Flag mask test, if 1 - there is mask
        int maskFlagTest;

        //! Stats flag, if 1, gather the statistics
        int statsFlag;

        //! Weight the error on the edges, if 1 then yes
        int weightedErrorFlag;

        //! Number of the validation iterations
        int numbItValidation;

        //! Weight freeze flag 
        //! 0 - no weight freezing   1 - freeze encoder for fractionOfEpochs 2 - freeze
        //! decoder for fractionOfEpochs 3 -freeze decoder for fractionOfEpochs and then freeze 
        //! encoder for fractionOfEpochs, the remaining of time both trained together
        int weightFreezeFlag;

        int polishing;

        //! Freeze the weights for a fraction of epochs
        double freezeFractionEpochs;

        //! Vector holding the sizes of the layers
        vector<int> layersVec;

        //! Vector holding the activation functions
        vector<int> actVec;

        //! Path to the training data
        string nameDataTrainIn;

        //! Path to the test data
        string nameDataTestIn;

        //! Path to the file initialising the weights
        string weightsInit;

        //! Path to the test mask
        string nameMaskTest;

        //! Path to the validation mask
        string nameMaskValidation;

        //! Path to the folder to save the result
        string saveFolder;


};
#endif
