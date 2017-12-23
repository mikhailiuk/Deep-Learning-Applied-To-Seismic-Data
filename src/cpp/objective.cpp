#include "../headers/Objective.h"

void Objective::initialise (ParamsInit parameters){
        // Need to initialise the scaling factor of the error depending on the distance from the patch center
        initialiseWeightsFromCenter(parameters);
        
        // Set objective
        m_objective = static_cast<ObjectiveName>(parameters.objective); 

        // Set the delta for the huber delta objective function
        m_huberDelta=parameters.huberDelta;

};

void Objective::initialiseWeightsFromCenter(ParamsInit parameters){

        double valZ,valY,valX;
        int id;

        // Find the size of the diagonal of the patch
        double distanceDiagonal = sqrt(pow(static_cast<double>(parameters.patchZ)/2.0,2)+
                                       pow(static_cast<double>(parameters.patchY)/2.0,2)+
                                       pow(static_cast<double>(parameters.patchX)/2.0,2));


        // Allocate memory for the array holding the distance from the senter
        m_distFromCenter.resize(parameters.layersVec.at(0)+1,distanceDiagonal);

        // If using weighted error
        if (parameters.weightedErrorFlag==1) {
                // Go through the depth of the patch
                for (int ii = 0; ii < parameters.patchZ; ++ii) {
                        // Go through the wigth in X
                        for (int jj = 0; jj < parameters.patchX; ++jj) {
                                // Go through the width in Y
                                for (int kk = 0; kk < parameters.patchY; ++kk) {
                                        // Compute the index for the patch weight
                                        id = ii*parameters.patchY*parameters.patchX+jj*parameters.patchY+kk;
                                        // Compute the distance in Z direction from the center
                                        valZ = static_cast<double>(static_cast<double>(ii)-static_cast<double>(parameters.patchZ-1)/2.0);
                                        // Compute the distance in Y direction from the center
                                        valY = static_cast<double>(static_cast<double>(kk)-static_cast<double>(parameters.patchY-1)/2.0);
                                        // Compute the distance in X direction from the center
                                        valX = static_cast<double>(static_cast<double>(jj)-static_cast<double>(parameters.patchX-1)/2.0);

                                        // Compute the Eucledian distance from the center to the point
                                        m_distFromCenter[id]=sqrt(pow(valZ,2)+pow(valY,2)+pow(valX,2));    
                                }
                        }
                }
        }
        //Last one is the distance from the center of the cube to the corner (half diagonal)
}

double Objective::objective(int stage, int position, double target, double predicted){

        double error, derivative;

        // By what should be the error scaled  
        double factor = m_distFromCenter[position]/m_distFromCenter[m_distFromCenter.size()-1];

        // Case to decide which objective function to use
        switch(m_objective){

                case leastSquares:{
                        error = static_cast<double>(1-stage)*leastSquaresError(target,predicted);
                        derivative = static_cast<double>(stage)*leastSquaresDer(target,predicted);
                        return (error+derivative)*(1.0+factor)*0.5; // either error or derivative is 0
                }
                case huberLoss:{
                        error = static_cast<double>(1-stage)*huberLossError(target,predicted);
                        derivative = static_cast<double>(stage)*huberLossDer(target,predicted);
                        return (error+derivative)*(1.0+factor)*0.5; // either error or derivative is 0
                }
                case lOneNorm:{
                        error = static_cast<double>(1-stage)*lOneNormError(target,predicted);
                        derivative = static_cast<double>(stage)*lOneNormDer(target,predicted);
                        return (error+derivative)*(1.0+factor)*0.5; // either error or derivative is 0
                }
                case qMeasure:{
                        error = static_cast<double>(1-stage)*qMeasureError(target,predicted);
                        derivative = static_cast<double>(stage)*qMeasureError(target,predicted);
                        return (error+derivative)*(1.0+factor)*0.5; // either error or derivative is 0
                }
                default:{
                        printf("No such objective exists\n");
                        exit(0);
                }

        }
        
        return 0.0;
}

// Stage is either feedforward (0), or backpropagate (1)
double Objective::objective(int stage, double target, double predicted){

        double error, derivative;

        switch(m_objective){

                case leastSquares:{
                        //If stage is backpropagate error is 0
                        error = static_cast<double>(1-stage)*leastSquaresError(target,predicted);

                        //If stage is feedforward derivative is 0
                        derivative = static_cast<double>(stage)*leastSquaresDer(target,predicted);
                        return (error+derivative); // either error or derivative is 0
                }
                case huberLoss:{
                        error = static_cast<double>(1-stage)*huberLossError(target,predicted);
                        derivative = static_cast<double>(stage)*huberLossDer(target,predicted);
                        return (error+derivative); // either error or derivative is 0
                }
                case lOneNorm:{
                        error = static_cast<double>(1-stage)*lOneNormError(target,predicted);
                        derivative = static_cast<double>(stage)*lOneNormDer(target,predicted);
                        return (error+derivative); // either error or derivative is 0
                }
                case qMeasure:{
                        error = static_cast<double>(1-stage)*qMeasureError(target,predicted);
                        derivative = static_cast<double>(stage)*qMeasureError(target,predicted);
                        return (error+derivative); // either error or derivative is 0
                }
                default:{
                        printf("No such objective exists\n");
                        exit(0);
                }

        }

        return 0.0;
}


double Objective::getLSError(double target, double predicted, int position){
        // Calculate the weight of the error
        double factor = m_distFromCenter[position]/m_distFromCenter[m_distFromCenter.size()-1];
        return leastSquaresError(target,predicted)*(1.0+factor)*0.5;
}

double Objective::leastSquaresError(double target, double predicted){
        //0.5*(t-p)^2
        return 0.5*(target-predicted)*(target-predicted);
}

double Objective::leastSquaresDer(double target, double predicted){
        //dJ/dp = d(0.5(t-p)^2)/dp = -(t-p)
        return -(target-predicted);
}

double Objective::lOneNormError(double target, double predicted){
        
        //Absolute value |t-p|
        return fabs(target-predicted);
}

double Objective::lOneNormDer(double target, double predicted){

        //Slope if less than 0 -> negative, else positive
        return ((predicted-target)<0)?(-1.0):(1.0);
}


double Objective::huberLossError(double target, double predicted){
        double diff = target-predicted;

        //0.5x^2 (squared)
        if (m_huberDelta>=fabs(diff)){
                return 0.5*diff*diff;

        //Linear
        }else {
                return m_huberDelta*(fabs(diff)-0.5*m_huberDelta);
        }
                

}

double Objective::huberLossDer(double target, double predicted){
        double diff = target-predicted;
        if (m_huberDelta>=fabs(diff)){
                return -diff;
        }else {
                return (diff>0)?(-m_huberDelta):(m_huberDelta);
                //return (diff>0)?(-1.0):(1.0);
        }
}

double Objective::meanVarCovarQmeasure(vector<double> target, vector<double> predicted, vector<double> mask){
        double sumPredicted = 0.0;
        double sumTarget = 0.0;
        m_sumMask =  0.0;

        // Set the predicted values to 0 if missing
        for (std::size_t ii = 0; ii<predicted.size(); ii++){
                predicted[ii] *= mask[ii];
        }

        // sum predicted and target values and find the # of non-zero values in the mask
        for (std::size_t ii = 0; ii<target.size(); ii++){
                sumPredicted += predicted[ii]*mask[ii];
                m_sumMask += mask[ii];
                sumTarget += target[ii]*mask[ii];
        }
        
        // Find the mean values
        m_meanTarget = sumTarget/m_sumMask;
        m_meanPredicted = sumPredicted/m_sumMask;

        // Set the variances to 0
        m_varianceTarget = 0.0;
        m_variancePredicted = 0.0;
        m_meanDevTarget = 0.0;
        m_meanDevPredicted = 0.0;
        m_covariance = 0.0;

        // Calculate variance and covariances
        for (std::size_t ii = 0; ii<target.size(); ii++){
                m_meanDevTarget += (target[ii]-m_meanTarget)*mask[ii];
                m_meanDevPredicted += (predicted[ii]-m_meanPredicted)*mask[ii];
                m_varianceTarget += (target[ii]-m_meanTarget)*(target[ii]-m_meanTarget)*mask[ii];
                m_variancePredicted += (predicted[ii]-m_meanPredicted)*(predicted[ii]-m_meanPredicted)*mask[ii];
                m_covariance += (target[ii]-m_meanTarget)*(predicted[ii]-m_meanPredicted)*mask[ii];
        }

        // Devide by the number of non-zero coefficients
        m_varianceTarget/=(m_sumMask-1);
        m_variancePredicted/=(m_sumMask-1);
        m_covariance /=(m_sumMask-1);
        m_meanDevTarget /=m_sumMask;
        m_meanDevPredicted /=m_sumMask;
        
        // From the formula, see the project
        double coef1 = 0.01;
        double coef2 = 0.01;

        //Split large equation into terms to simplify
        double lTermDenom = (m_meanTarget*m_meanTarget+m_meanPredicted*m_meanPredicted+coef1);
        double lTermNumer = 2.0*m_meanTarget*m_meanPredicted+coef1;
        double lTerm = lTermNumer/lTermDenom;
        double lTermDer = 2.0*(m_meanTarget*m_meanTarget-m_meanPredicted*lTermNumer)/(m_sumMask*lTermDenom);


        double bTermCs = m_varianceTarget+m_variancePredicted+coef2;
        double dTermCs = 2.0*m_covariance+coef2;
        double csTerm =dTermCs/bTermCs;
        double Qmeasure = csTerm*lTerm;

        return Qmeasure;
                
}

double Objective::qMeasureError(double valTarget, double valPredicted)
{

        double coef1 = 0.02;
        double coef2 = 0.01;

        double lTermDenom = (m_meanTarget*m_meanTarget+m_meanPredicted*m_meanPredicted+coef1);

        double lTermNumer = 2.0*m_meanTarget*m_meanPredicted+coef1;

        double lTerm = lTermNumer/lTermDenom;

        double lTermDer = 2.0*(m_meanTarget*m_meanTarget-m_meanPredicted*lTermNumer)/(m_sumMask*lTermDenom);

        double aTermCs = 2.0/(m_sumMask-1)*((valTarget-m_meanTarget)-m_meanDevTarget);

        double bTermCs = m_varianceTarget+m_variancePredicted+coef2;

        double cTermCs = 2.0/(m_sumMask-1)*((valPredicted - m_meanPredicted) - m_meanDevPredicted);

        double dTermCs = 2.0*m_covariance+coef2;

        double eTermCs = bTermCs*bTermCs;

        double csTerm =dTermCs/bTermCs;

        double csTermDer = (aTermCs*bTermCs-cTermCs*dTermCs)/eTermCs;
        
        double Qmeasure = csTerm*lTermDer;

        double QmeasureDer = lTermDer*csTerm+csTermDer*lTermDer;
    
        return -QmeasureDer;

}


