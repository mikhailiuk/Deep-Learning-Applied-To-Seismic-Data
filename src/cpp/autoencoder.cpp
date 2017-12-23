// Author: Aliaksei Mikhailiuk, 2017.

#include "../headers/Autoencoder.h"

// Serial backpropagate
int Autoencoder::backpropagate(double learningRate,double lambda){

        int numberWeights = getNumbLayers()-1;
	for (int kk=numberWeights; kk>0; --kk){
	    for (int jj = 0; jj<getLayerNumbU(kk); ++jj){
	        for (int ii=0; ii<getLayerNumbU(kk-1)+1; ++ii){
                        updateWeights(kk-1,jj,ii,learningRate,0.0);
	        }
	    }
	}
	return 0;
}



// Node parallel backpropagate
int Autoencoder::backpropagateNodeParallel(double learningRate,double lambda,int flagWeightsFreeze){

        if (getNumbLayers() == 3 && m_sparse != 0.0){
                #pragma omp for schedule(guided) 
                for (int ii = 0; ii<m_avUnits.size(); ii++){
                        m_avUnits[ii]=0.0;
                }
        }

        // Counters are private thus every thread updates its own local copy and thus explicit synchronization is required
        int kk=0, lim=0;

        // If the flag is set to release the weights 
        if (flagWeightsFreeze==0){

               // Set the last layer to the last layer of the network
               kk  = getNumbLayers()-1;
               // Set the first layer to the first layer of the network 
               lim = 0;

        // If the flag is set to freeze the encoder 
        }else if (flagWeightsFreeze==1){

                // Set the last layer to the last layer of the network
                kk = getNumbLayers()-1;

                // Set the first layer to the middle layer of the network
                lim = (getNumbLayers()-1)/2;

        // If the flag is set to freeze the encoder 
        } else if (flagWeightsFreeze == 2){

                // Set the last layer to the middle layer of the network
                kk =(getNumbLayers()-1)/2;

                // Set the first layer to the first layer of the network 
                lim=0;

        }

        // Iterating backwards from the last layer to the first
	while (kk>lim){
                //Iterate through the current layer
                #pragma omp for schedule(guided)                
                for (int jj = 0; jj<getLayerNumbU(kk); ++jj){
                        // Iterate through the deeper layer 
                        for (int ii=0; ii<getLayerNumbU(kk-1)+1; ++ii){
                                // If there is either momentum or adagrad then go to the function which calculates parameters for them
                                if(m_flagFast == disabled){
                                        updateWeights(kk-1,jj,ii,learningRate,0.0);

                                }else{
                                        updateWeightsFast(kk-1,jj,ii,learningRate,0.0);
                                }
                        }
                }

                kk--;


        }
	return 0;
}


int Autoencoder::updateWeightsFast(int kk,int jj, int ii,double lr,double penalty){

        // Calculate the gradient for the connection between jj and ii 
        double gradient = getGrad(kk+1, jj, ii);

        // Update the weight
        m_weights[kk][jj][ii] = m_weights[kk][jj][ii] - lr*gradient;

        return 0;
}

int Autoencoder::updateWeights(int kk,int jj, int ii,double lr,double penalty){


        // Get the gradient
        double gradient = getGrad(kk+1, jj, ii);

        // sumD = (sumD*timeStep+gradient)/(timeStep+1)      
        m_sumDeltaArr[kk][jj][ii] = m_sumDeltaArr[kk][jj][ii]*((float)m_timeStep/((float)(m_timeStep+1)))+gradient/(float)(m_timeStep+1);


        // mom = momMag*mom+(1-momMag)*grad 
        m_momentumArr[kk][jj][ii] = m_momentum*m_momentumArr[kk][jj][ii] + (1.0-m_momentum)*gradient;

        // Update the weight w = w - lr/sqrt(sumD*sumDmag+1.0)*momentum
        m_weights[kk][jj][ii] = m_weights[kk][jj][ii] - lr/sqrt(abs(m_sumDeltaArr[kk][jj][ii])*m_adaGrad + 1.0)*m_momentumArr[kk][jj][ii]; 

        return 0;
}

Autoencoder::~Autoencoder(){


        // Delete layers as memory for them was allocated dynamically
	for (int ii = 0; ii<m_miniBatch; ++ii){
		for (int jj = 0; jj<m_numberLayers; ++jj){
			delete m_layers[ii][jj];
		}
	}

}
