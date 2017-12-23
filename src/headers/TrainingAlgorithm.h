#ifndef TRAININGALGORITHM_H
#define TRAININGALGORITHM_H

#include <string.h>
#include <fstream>   
#include <vector>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

#include <algorithm>  
#include <ctime>   
#include <cstdlib>     
using std::vector;
#include "NeuralNetwork.h"
#include "Activation.h"
#include "DataReader.h"
#include "ParamsInit.h"
#include "Objective.h"
#include "Timer.h"


/*! \class TrainingAlgorithm
    \brief The algorithm feeds the data into the networks and envokes functions of the backpropagation algorithm
    
*/
class TrainingAlgorithm{

private:

        Objective m_objective;
        enum FlagWeightsFreeze {none, encoder, decoder, decoderEncoder};

        FlagWeightsFreeze m_weightFreezeFlag; 

        enum Flag {disabled, enabled};

        Flag m_spraceFlag;

        Flag m_shuffleFlag;

        Flag m_polishing;

        DataReader dataReader;


        //! Learning rate
	double m_learningRate;

        //! Lambda - magnitude of the penalty in the Contractive Autoencoder
	double m_lambda;

        //! Annealin, used for learningRate = learningRate*(1.0/(1.0+annealing*iteration));
	double m_annealing;

        //! The average error of the validation iteration 
        double m_averageTestValid;

        //! Minimum validation error over all iterations
        double m_minValueValidation;

        //! Number of epochs for which the weights will be frozen
        float m_freezeFractionEpochs;

        //! Time in the feedForward
        double m_timeFeedForward;

        //! Time in the deltaCompute
        double m_timeDeltaCompute; 

        //! Time in the backPropagation
        double m_timeBackPropagate;
        
        //! The upper limit of the learning rate when dynamic learning rate depending on validation is used
        double m_learningRateUpper;

        //! The lower limit of the learning rate when dynamic learning rate depending on validation is used
        double m_learningRateLower;

        //! The number of training epoches
	int m_numbEpoches;

        //! Flag eigther 1 or 0. It 1, the number of training iterations is increased by one in every subsequent epoch
        int m_curriculum;

        //! The current number of iterations
        int m_currIterations;

        //! The number of test iterations
	int m_numbItTest;

        //! The number of validation iterations
        int m_numbItValidation;

        //! The number of training iterations
	int m_numbItTrain;

        //! The number of inputs to the neural network
        int m_numbInputs;

        //! The number of outputs of the neural network
        int m_numbOutputs;
        
        //! The image height (patchY)
        int m_imageHeight;

        //! The image height (patchX)
        int m_imageWidth;

        //! The image height (patchZ)
        int m_imageDepth;

        //! Dimentions of the data cube in X
        int m_dataWidth;

        //! Dimentions of the data cube in Y
        int m_dataHeight;

        //! The name of the folder where the results are saved
        std::string m_saveFolder;

        //! The array to hold the order in which the training examples are fed into the network
        vector  < int >         m_orderTraining;

        //! The array to hold values of the learning rate 
	vector  < double >      m_learningRateArr;

        //! The array to hold the output of the testing (prediction values)
	vector  < double >      m_outputTest;

        //! The array to hold training input
	vector  < double >      m_dataTrainIn;

        //! The array to hold test input
        vector  < double >      m_dataTestIn;

        //! The test mask
        vector  < double >      m_maskTest;

        //! The training mask
        vector  < double >      m_maskTrain;

        //! The validation mask
        vector  < double >      m_maskValidation;

        //! The training error
	vector  < double >      m_errorTrain;

        //! The validation error
	vector  < double >      m_errorValidation;

        //! The ratio of the errors 
	vector  < double >      m_ratioOfErrors;

public:
        //! Constructor
	TrainingAlgorithm();

        //! Destructor
	~TrainingAlgorithm();

        //! The function to initialise private fields of the class
        //! \param parameters holds the values read from the config file
        int initialise (ParamsInit parameters);

        //! The function iterating over the training examples and setting the backpropagation in the NN
        //! \param neuralNetwork neural network 
        int trainNeuralNetworkTaskParallel(NeuralNetwork *neuralNetwork);

        //! The function to test the neural network, feeds the data in and gets the prediction for the missing values
        //! \param neuralNetwork neural network
	void testNeuralNetwork(NeuralNetwork *neuralNetwork);


        //! The function to polish the results produced by the neural network, feeds the reconstructed data (instead of 0s) in and gets the prediction for the reconstructed values
        //! \param neuralNetwork neural network
	void polishing(NeuralNetwork *neuralNetwork);

        //! The function for sanity check, compares the number of inputs set in the neural network to the number of inputs specified in the config file
        //! \param neuralNetwork neural network
	int consistencyTest(NeuralNetwork *neuralNetwork);

        //! The function which envoken when the results are written to a result folder
        //! \param path to the output folder
        void writeOutput(std::string path);

        //! The function called in every training epoch to update the parameters 
        //! \param ii current epoch
        //! \param mb mini-batch size
        int updateTrainingParameters(int ii,int mb);

        //! The function for the sliding window on the validation error, triggers leariningRateControl
        //! \param error validation error in the current epoch
        //! \param neuralNetwork neural network
        int updateValidation(double error, NeuralNetwork * neuralNetwork);

        //! The function to update the weight freezing flag
        //! \param ii current epoch
        int weightFreezeUpdate(int ii);

        //! The function to shuffle the data before the next epoch
        //! \param mb mini-batch
        int updateShuffle(int mb);

        //! The function to update the learning rate with annealing
        //! \param ii current epoch
        int updateLearningRate(int ii);

        //! The function to control the learning rate depending on the slope of the validation error
        //! \param change learningRate=learningRate*change
        int leariningRateControl(double change);

        //! The function to set the inputs of the neural network
        //! \param kk - training iteration counter
        //! \param ff - mini-batch counter
        //! \param maskCounter starting point for the next patch in the mask
        //! \param neuralNetwork neural network
        void copyDataToNN(int kk, int ff, int maskCounter,NeuralNetwork *neuralNetwork);

        //! The function to update the counters after every training iteration
        //! \param ff mini-batch counter
        //! \param maskCounter starting point for the next patch in the mask
        int updateCounters(int *ff, int *maskCounter);

        double getOutTst(int ii)               const   {return m_outputTest.at(ii);};

        double getLearningRate()                const   {return m_learningRate;};

        double getLearningRateArr(int ii)      const   {return m_learningRateArr.at(ii);};

        int    getLearningRateArrSize()         const   {return m_learningRateArr.size();};

        double getErrorTrain(int id)            const   {return m_errorTrain[id];};

        int    getErrorTrainSize()              const   {return m_errorTrain.size();};

        double getErrorValidation(int ii)       const   {return m_errorValidation[ii];};

        int    getErrorValidationSize()         const   {return m_errorValidation.size();};

        double getRatio(int ii)                 const   {return m_ratioOfErrors[ii];};

        int    getRatioSize()                   const   {return m_ratioOfErrors.size();};

        double getLambda()                      const   {return m_lambda;};

        double getAnnealing()                   const   {return m_annealing;};

        int    getNumbEpoches()                 const   {return m_numbEpoches;};

        int    getNumbItTrain()                 const   {return m_numbItTrain;};

        int    getNumbItValidation()            const   {return m_numbItValidation;};

        int    getNumbItCurr()                  const   {return m_currIterations*m_curriculum+(1-m_curriculum)*getNumbItTrain();};

        int    getNumbItTest()                  const   {return m_numbItTest;};

        int    getNumbOutputs()                 const   {return m_numbOutputs;};

	std::string getNameSaveFolder()         const   {return m_saveFolder;};

        double getTimeFeedForward()             const   {return m_timeFeedForward;};

        double getTimeBackpropagate()           const   {return m_timeBackPropagate;};

        double getTimeDeltaCompute()            const   {return m_timeDeltaCompute;};

        int    getImageHeight()                 const   {return m_imageHeight;};

        int    getImageWidth()                  const   {return m_imageWidth;};

        int    getImageDepth()                  const   {return m_imageDepth;};


};

#endif
