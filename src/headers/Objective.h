// Author: Aliaksei Mikhailiuk, 2017.

#ifndef OBJECTIVE_H
#define OBJECTIVE_H

#include <iostream> 
#include <new> 
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <string.h> 
#include <fstream> 
#include <cstdlib>
#include <ctime>
#include <algorithm>  
#include <stdio.h>    
#include <time.h> 
#include "ParamsInit.h"
using std::vector;

/*! \class Objective
    \brief Encapsulates objective functinos

*/
class Objective
{

private:

        //! The enum to hold objective functions
        enum ObjectiveName {leastSquares, huberLoss, lOneNorm, qMeasure};

        //! The name of the objective function (one of 4 in the enum)
        ObjectiveName m_objective;

        //! The delta in the huber delta objective function
        double m_huberDelta;

        //! Number of non-zero elements in the mask
        double m_sumMask;

        //! Mean of the target values in the patch
        double m_meanTarget ;

        //! Mean of the predicted values in the patch
        double m_meanPredicted ;

        //! Variance target
        double m_varianceTarget;

        //! Predicted variance
        double m_variancePredicted;

        //! Average standard deviation target
        double m_meanDevTarget;

        //! Average standard deviation predicted
        double m_meanDevPredicted;

        //! Covariance in the target and predicted values
        double m_covariance;

        //! The vector to hold distances of the values from the center
        vector  < float > m_distFromCenter;

public:
        //! A function to initialise
        //! \param parameters - stores values from the config file
        void initialise (ParamsInit parameters);

        //! A function to initialise m_distFromCenter
        //! \param parameters - stores values from the config file
        void initialiseWeightsFromCenter(ParamsInit parameters);

        //! A function to get the Squared Error 
        //! \param target - a target value
        //! \param predicted - a predicted value
        double leastSquaresError(double target, double predicted);

        //! A function to get the derivative of the Squared Error
        //! \param target - a target value
        //! \param predicted - a predicted value
        double leastSquaresDer(double target, double predicted);
        
        //! A function to get the error of the L1 norm
        //! \param target - a target value
        //! \param predicted - a predicted value
        double lOneNormError(double target, double predicted);

        //! A function to get the derivative of the L1 norm
        //! \param target - a target value
        //! \param predicted - a predicted value
        double lOneNormDer(double target, double predicted);

        //! A function to get the huber loss error
        //! \param target - a target value
        //! \param predicted - a predicted value
        double huberLossError(double target, double predicted);

        //! A function to get the derivative for huber loss
        //! \param target - a target value
        //! \param predicted - a predicted value
        double huberLossDer(double target, double predicted);

        //! A function to calculate the SSIM error 
        //! \param target - a target value
        //! \param predicted - a predicted value
        double qMeasureError(double target, double predicted);

        //! A function to calculate parameters for the SSIM error
        //! \param target - a target patch 
        //! \param predicted - a predicted patch
        //! \param mask - mask
        double meanVarCovarQmeasure(vector<double> target, vector<double> predicted, vector<double> mask);

        //! A function returning the value of the chosen objective function or its derivative
        //! \param stage - 0 for feedforward, 1 for backpropagate
        //! \param target - a target value
        //! \param predicted - a predicted value
        double objective(int stage, double target, double predicted);

        //! A function returning the value of the chosen objective function or its derivative
        //! \param stage - 0 for feedforward, 1 for backpropagate
        //! \param position in the neural network output layer (used to calculate the distance from the patch)
        //! \param target - a target value
        //! \param predicted - a predicted value
        double objective(int stage, int position, double target, double predicted);
        
        //! A function used to give a scaled value for the Squared Error depending on the distance from the center of the patch
        //! \param position in the neural network output layer (used to calculate the distance from the patch)
        //! \param target - a target value
        //! \param predicted - a predicted value
        double getLSError(double target, double predicted, int position);
};


#endif
