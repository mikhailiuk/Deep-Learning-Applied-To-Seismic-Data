#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "NeuralNetwork.h"
#include <omp.h>
#include <stdlib.h>

/*! \class Autoencoder
    \brief Class inherits functionality from Neural Network.
        

    Simple autoencoder.
*/
class Autoencoder: public NeuralNetwork{

public:

        ~Autoencoder();
      
        //! A function to update the weights with the gradient
        //! \param ii layer
        //! \param jj node in the current layer
        //! \param kk node in the previous layer
        //! \param learningRate learning rate
        //! \param penalty lambda for the contractive autoencoder
        inline int updateWeights(int ii,int jj, int kk,double learningRate,double penalty);

        //! A function to update the weights with the gradient (fast, no redundunt calculations for momentum and adaGrad if both are not used)
        //! \param ii layer
        //! \param jj node in the current layer
        //! \param kk node in the previous layer
        //! \param learningRate learning rate
        //! \param penalty lambda for the contractive autoencoder
        inline int updateWeightsFast(int ii,int jj, int kk,double learningRate,double penalty);

        //! The backpropagation algorithm
        //! \param learningRate - the learning rate
        //! \param lambda - the contractive autoencoder parameter
        int backpropagate(double learningRate,double lambda);

        //! The backpropagation algorithm with parallel nodes
        //! \param learningRate - the learning rate
        //! \param lambda - the contractive autoencoder parameter
        //! \param flagRelease - 0 if the weights are to be released
        int  backpropagateNodeParallel(double learningRate,double lambda,int flagRelease);

        //! Returns the type (string) of the autoencoder
        std::string type()      const     {return "Autoencoder";};

};

#endif
