#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <string.h> 
#include <vector>
#include <fstream> 
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>    // std::random_shuffle
#include <stdio.h>      /* printf, scanf, puts, NULL */
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include "Layer.h"
#include "ParamsInit.h"
#include "Objective.h"
#include <omp.h>

using std::vector;


/*! \class NeuralNetwork
    \brief A base class for Neural Network Algorithms.

    A base class for contractive autoencoder and simple autoencoder.
*/
class NeuralNetwork{

	protected:

                //! Flag to hold activated and deactivated values
                enum Flag {disabled, enabled};
                
                //! Flag used to mark whether fast versions of the functions can be used
                Flag m_flagFast;   

                //! Flag to mark whether the user wants to gather statistics
                Flag m_statsFlag;
                
                //! Flag to mark whether the weights have been swapped (i.e. local minimum reached)
                bool m_flagSwapped;

                //! Number of layers in the neural network
		int m_numberLayers;

                //! Mini-batch size
		int m_miniBatch;

                //! Number of test iterations
                int m_itTest; 

                //! Number of train iterations
                int m_itTrain; 

                //! Time step
                int m_timeStep;
                
                //! Magnitude of the adaptive gradient [0;1)
                float m_adaGrad;

                //! Magnitude of the momentum [0;1)
                float m_momentum;

                //! Fraction of the units set to 0
                float m_dropOut;

                //! Magnitude of the penalty
                double m_sparse; 

                //! The maximum value of the output neurons expected
                double m_sparsityParameter;

                //! Variable to hold objective function
                Objective m_objective;
                
                //! Matrix of layers (1D - minibatch, 2D - layers)
		vector  < vector < Layer *> >           m_layers;

                //! Vector to hold the sizes of layers 
                vector  < int >                         m_layersSizeVec;

                //! Matrix to hold the stats about output of the hidden units
                vector  < vector < float > >            m_statsHiddenUnits;

                //! Vector to hold regularization for the sparse autoencoder
                vector  < double >                      m_avUnits;

                //! Vector used for OpenMP to store calculations produced by different threads 
                vector  < double >                      m_sumDelta;   

                //! Vector with elements 1 and 0 marking the neurons set to 0
                vector  < vector < double > >           m_dropOutVec;

                //! Cube of weights
                vector < vector < vector < double > > > m_weights;

                //! Cube of weights with which local minimum is achieved
                vector < vector < vector < double > > > m_weightsMinCp;

                //! Holding momentum for each weight
		vector < vector < vector < double > > > m_momentumArr;

                //! Holding deltas for the adaGrad
		vector < vector < vector < double > > > m_sumDeltaArr;


	public:

		NeuralNetwork();

                //! The function to get the Squared error between the target and predicted values used in to weight error on the edges of the patches
                //! \param target - target value
                //! \param predicted - predicted value
                //! \param distFromCenter - distance of the point from the center of the patch
                double getLSError(double target, double predicted, int distFromCenter);

                //! A function called when local minimum in the validation error is achieved, to store the combination of weights 
                int makeWeightCpy();
                
                //! A function to compute deltas for a given mini-batch
                //! \param mb mini-batch
                double deltaCompute (int mb);

                //! A parallel function to compute deltas for a given mini-batch
                //! \param mb mini-batch
                double deltaComputeNodeParallel (int mb);

                //! Feedforward
                //! \param mb - mini-batch
                int feedforward(int mb);

                //! Node parallel feedforward
                //! \param mb - mini-batch
                int feedforwardNodeParallel(int mb);

                //! A feed forward function used to compute the penalty for the sparse autoencoder
                //! \param mb mini-batch
                int feedforwardNodeParallelCompSparse(int mb);

                //! A function to apply the derivative of the non-linearity and get the propagated value
                //! \param mb mini-batch
                //! \param la layer
                //! \param ii unit in the layer
        	double backPropagateU(int mb,int la,int ii);
        
                //! A function to initialise the Neural netowrk
                //! \param parameters used to stor contents of the config file
                void   initialise(ParamsInit parameters);

                //! A function to initialise the vectors holding positions of the nodes to be set to 0
                void   initialiseDropOut();

                //! Allocate memory for the array holding the statistics about hidden units
                void   initialiseStatsHiddenUnits();

                //! A function to initialise weights
                //! \param parameters - used to stor the results from the config file
                void   initialiseWeights (ParamsInit parameters);

                //! A function to set bias and delta values 
                //! \param parameters - used to stor the results from the config file
                void   initialiseDeltaBias(ParamsInit parameters);


                //! A function used to initialise weights from the file
                //! \param path - a path to the weights file
		void initWeightsFromFile(std::string path);

                //! A function used to get the gradient in the backporapagation stage
                //! \param kk - layer
                //! \param jj - unit in the kk layer
                //! \param ii - unit in the (kk-1) layer
                double getGrad(int kk, int jj, int ii);

                //! A function
                //! \param mb
                //! \param kk
                //! \param jj
                double dotProduct(int mb, int kk,int jj);
                

                //! A function called from the feedforward to compute the dot product of the neuron outputs with associated weights and pass it through the non-linearity
                //! \param mb
                //! \param kk
                //! \param jj
                void dotProductProp(int mb, int kk,int jj);

                //! A function to set the output of the unit 
                //! \param mb - mini-batch
                //! \param la - layer
                //! \param u - unit
                //! \param val - value
		void setLayerUOutput (int mb, int la, int u, double val);

                //! A function to set the values of the drop out (shuffle the vectors) 
                int setDropOut();

                //! A function to set the delta of the unit 
                //! \param mb - mini-batch
                //! \param la - layer
                //! \param u - unit
                //! \param val - value
		void setLayerUDelta (int mb, int la, int u, double val);

                //! A function to swap the weights (current with the save with which the local minimum in the error is obtained)
                void swapWeights();

                //! The function to set timestep
	        void setTimeStep(int step){m_timeStep=step;};


                //! A virtual function implemented by CAE and AE to w = w-lr*grad with momentum and adaGrad
                inline virtual int updateWeights(int ii,int jj, int kk,double val,double penalty)       = 0;
                
                //! A virtual function implemented by CAE and AE to w = w-lr*grad without momentum and adaGrad
                inline virtual int updateWeightsFast(int ii,int jj, int kk,double val,double penalty)   = 0;

                //! A virtual function implemented by CAE and AE to compute the backpropagation
                virtual int backpropagate(double learningRate, double lambda)                           = 0;

                //! A virtual function implemented by CAE and AE to compute the backpropagation in parallel
                virtual int backpropagateNodeParallel(double learningRate,double lambda,int flagRelease)= 0;

                //! A virtual destructor
                virtual ~NeuralNetwork();

                //! A virtual function giving the type of the encder (string)
                virtual std::string type()                      const        =      0;

		Layer* getLayer(int mb, int id)                 const   {return m_layers[mb][id];};

		int    getNumbLayers()                          const   {return m_numberLayers;};

		int    getMiniBatch()                           const   {return m_miniBatch;};

		double getLayerUDelta(int mb, int la, int u)    const   {return m_layers[mb][la]->getUnitDelta(u);};

		int    getLayerNumbU(int la)                    const   {return m_layersSizeVec[la];};

		double getLayerUOutput(int mb, int la, int u)   const   {return m_layers[mb][la]->getUnitOutput(u);};

                float  getStats(int ii, int jj)                 const   {return m_statsHiddenUnits[ii][jj];};

		int    getLayerAct(int mb, int la)              const   {return m_layers[mb][la]->getActivation();};

		double getWeightO(int ii,int jj,int kk)         const   {return m_weights[ii][jj][kk];};

                double getMomentum()                            const   {return m_momentum;};

                double getDropOut()                             const   {return m_dropOut;}

                double getAdaGrad()                             const   {return m_adaGrad;};

                int    getStatsFlag()                           const   {return m_statsFlag;};

                bool   getSwapped()                             const   {return m_flagSwapped;};

};

#endif
