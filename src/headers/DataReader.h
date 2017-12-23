#ifndef DATAREADER_H
#define DATAREADER_H

#include <string.h>
#include <fstream>   
#include <vector>
#include <iostream>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <omp.h>
#include <cmath>   
#include "ParamsInit.h"


using std::vector;

/*! \class TrainingAlgorithm
    \brief Encapsulates functionality associated with reading and writing the Seismic Data
*/
class DataReader{
private:

        //! A flag (more verbal than 0 and 1)
        enum Flag {disabled, enabled};

        //! Enabled if validation mask is used
        Flag m_flagMaskValidation;    
        
        //! Enabled if test mask is used
        Flag m_flagMaskTest;

        //! Enabled if augmentation is used 
        Flag m_augmentFlag;

        //! A path to the training data
	std::string m_nameDataTrainIn;

        //! A path to the test data
	std::string m_nameDataTestIn;

        //! A path to the test mask
        std::string m_nameMaskTest;

        //! A path to the validation mask
        std::string m_nameMaskValidation;

        //! A cube with coefficients (used to scale overlaps when writing the output cube 
        //! (if the shift in test is used)
        vector< vector < vector < double > > > m_coefCube;

        //! A cube with final (predicted values)
        vector< vector < vector < double > > > m_cube;
        
        //! A cube with clean data to write
        vector< vector < vector < double > > > m_cubeClean;

        //! A matrix to hold validation mask
        vector< vector < double > > m_maskValidation;

        //! A matrix to hold the mask for test and traing (real missing values)
        vector< vector < double > > m_mask;


        //! The scaling factor of the input - e.g. if 0.001 all values in 
        //! the cube are divided by 1000. 
        //! If -1.0, cube/m_maxAbs, if -2.0, (cube - m_min)/(m_max-m_min)
        double m_inputScale;

        //! Maximum value of the seismic data cube
        double m_max;

        //! Minimum value of the seismic data cube
        double m_min;

        //! Maximum absolute value of the seismic data cube
        double m_maxAbs;

        //! Total MSE error of the reconstruction
        double m_MSError;

        //! SNR measure of the reconstructed cube (see Georgious paper)
        double m_qMeasure;
 
        //! Z dimention of the Train Cube
        int m_zWidthTrain;

        //! X dimention of the Train Cube
        int m_xWidthTrain;

        //! Y dimention of the Train Cube
        int m_yWidthTrain;

        //! Z dimention of the Test Cube
        int m_zWidthTest;

        //! X dimention of the Test Cube
        int m_xWidthTest;

        //! Y dimention of the Test Cube
        int m_yWidthTest;
 
        //! Size of the patch in X dimention
        int m_patchX;

        //! Size of the patch in Z dimention
        int m_patchZ;

        //! Size of the patch in Y dimention
        int m_patchY;

        //! Shift in X dimention of the patch while training
        int m_shiftXTrain;

        //! Shift in Y dimention of the patch while training
        int m_shiftYTrain;

        //! Shift in Z dimention of the patch while training
        int m_shiftZTrain;

        //! Shift in X dimention of the patch while testing 
        int m_shiftXTest;

        //! Shift in Y dimention of the patch while testing
        int m_shiftYTest;

        //! Shift in Z dimention of the patch while testing
        int m_shiftZTest; 

        //! Number of training iterations        
        int m_numbItTrain;

        //! Number of test iterations
        int m_numbItTest;

        //! Total patch size (m_patchX*m_patchY*m_patchZ)
        int m_patchSize;

        //! Total number of test examples
        int m_totCubesTest;

        //! Width of the mask in X dimention
        int m_xWidthMask;

        //! Width of the mask in Y direction
        int m_yWidthMask;

        //! Size of the band of additional values in X direction
        int m_augmentX;

        //! Size of the band of additional values in Y direction
        int m_augmentY;

        //! Size of the band of additional values in Z direction
        int m_augmentZ;

        //! Flag, 1 if polishing is enabled;
        int m_polishing;
        

        //! Ratio of average target to predicted values
        double m_ratioTargetPredictedNeg;
        double m_ratioTargetPredictedPos;

public:



        //! A function to read the training data and pass it back to the training algorithm
        //! \param dataTrainIn the vector holding the training data used in the training algorithm
        //! \param numbIn
        //! \param numbItTrain number of training iterations (number of exemplars)
        //! \param maskRearangedTrain the vector holding the mask, so that it corresponds to the order of patches
        //! \param xWidthTrain - x dimention of the training cube
        //! \param yWidthTrain - y dimention of the training cube
        //! \param maskRearangedValidation - validation mask
        int readDataTrain(vector<double> *dataTrainIn,
                          int *numbIn,
                          int *numbItTrain,
                          vector<double> *maskRearangedTrain, 
                          int *xWidthTrain,
                          int *yWidthTrain,
                           vector<double> *maskRearangedValidation);

        //! A function to read the test data and pass it back to the training algorithm
        //! \param dataTestIn the vector holding the test data used in the training algorithm
        //! \param numbItTest number of test iterations (number of exemplars)
        //! \param maskRearanged the vector holding the test mask, so that it corresponds to the order of patches
        int readDataTest(vector<double> *dataTestIn,
                         int *numbItTest,
                         vector<double> *maskRearanged
                        );

        //! A function to convert an array of data into a cube with predicted values
        //! \param data target data 
        //! \param path to the file to save
        //! \param flagMask whether to apply mask or not (1.0 if the input cube is written, 0.0 if target)
        int writeDataCube(vector<double> data,
                          std::string path,
                          double flagMask);

        //! A function used to set all the values of the data cubes to zero
        void zeroCube();


        //! A function to write the data into a file 
        //! \param path to the save file
        //! \param flagMask - whether to apply the mask or not
        void writeCubeIntoFile(std::string path,
                               double flagMask);


        //! A function to convert an array of data into a cube with predicted values
        //! \param dataTarget target data 
        //! \param dataPred predicted data
        //! \param path to the file to save
        int writeDataCubePrediction(vector<double> dataTarget,
                                    vector<double> dataPred,
                                    std::string path);


        //! A function to calculate the values stored in m_coefCube, used to scale the
        //! overlapping patches in the final data cube
        double calculateCoefficientCube();


        //! A sensibility check function for the dimentions of the Patch
        int checkDimensionsPatch();

        //! A sensibility check function for the dimentions of the Mask
        int checkDimensionsMask();

        //! A function used to map the cube coefficients to the coefficients of the patches
        //! \param trainFlag - set to 1 if called from the function calculating bounds for training patches
        //! \param startYWidth - starting index in the Y in the data cube from where the patch is taken
        //! \param startXWidth - starting index in the X in the data cube from where the patch is taken 
        //! \param startZWidth - starting index in the Z in the data cube from where the patch is taken 
        //! \param ii - current iteration of the 
        //! \param numbIterations - total number of patches
        int calculateBounds(int trainFlag,int *startYWidth,int *startXWidth,int *startZWidth,int ii,int *numbIterations);


        //! A function to initialise the DataReader class (setting private variables)
        //! \param parameters hold all values red from the config file
        int initialise(ParamsInit parameters);

        //! A function to read the Training cube from the file
        int readTrainCube();

        //! A function to read the Test cube from the file
        int readTestCube();

        //! A function to read the mask from the file        
        int readMasks();

        //! A function to calculate the SNR 
        //! \param dataClean target data
        //! \param dataPredicted predicted data
        //! \param path - path to the file where to write the SNR
        double calculateSNR(vector<double> dataClean, vector<double> dataPredicted, std::string path);

        //! A function to calculate ratio between average target and predicted values
        double calculateAverageDataPredictedTarget();

};


#endif
