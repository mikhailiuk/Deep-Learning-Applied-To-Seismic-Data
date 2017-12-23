// Author: Aliaksei Mikhailiuk, 2017.

#include "../headers/ContractiveAutoencoder.h"



int ContractiveAutoencoder::backpropagateNodeParallel(double learningRate, double lambda,int a){

        int numberWeights = getNumbLayers()-1;
        int kk = numberWeights;
	double sum ,avOut,avHOut,avSqTerm, penalty;
	while (kk>0){
        #pragma omp for schedule(guided)
	    for (int jj = 0; jj<getLayerNumbU(kk); ++jj){
	    	sum = 0.0;
	    	avOut= 0.0;
	        avHOut = 0.0;
		avSqTerm = 0.0;
                // if the last layer in contractive autoencoder, compute values required for penalty
                if (kk == 1) {
                	for (int ii=0;ii<getLayerNumbU(kk-1)+1;++ii){
                   		sum+=getWeightO((kk-1),jj,ii)*getWeightO((kk-1),jj,ii);
                	}
                	for (int ff = 0; ff<m_miniBatch;ff++){
	                	avHOut += getLayerUOutput(ff,kk,jj);
	         		avSqTerm +=backPropagateU(ff,kk,jj)*backPropagateU(ff,kk,jj);
	                }
                }

	        for (int ii=0; ii<getLayerNumbU(kk-1)+1; ++ii){

                        // if the last layer is updated, calculate penalty and send it to update weights function
	        	if (kk == 1) {
                                for (int ff = 0; ff<m_miniBatch;ff++){
	        		        avOut += getLayerUOutput(ff,kk-1,ii);
	        	        }
	        	        penalty = 2*avSqTerm*(m_weights[kk-1][jj][ii]+avOut*(1-2*avHOut)*sum);                               
                                updateWeights(kk-1,jj,ii,learningRate,lambda*penalty);
                        
	        	}else{ 
                                updateWeights(kk-1,jj,ii,learningRate,0.0);
                        }
	        }
	    }
            kk--;
	}
	return 0;
}



int ContractiveAutoencoder::backpropagate(double learningRate, double lambda){
	double sum ,avOut,avHOut,avSqTerm, penalty;
	for (int kk=(getNumbLayers()-1); kk>0; kk--){
	    for (int jj = 0; jj<getLayerNumbU(kk); ++jj){
	    	sum = 0.0;
	    	avOut= 0.0;
	        avHOut = 0.0;
		avSqTerm = 0.0;
                // if the last layer in contractive autoencoder, compute values required for penalty
                if (kk == 1) {

                        // Sum of squared weights
                	for (int ii=0;ii<getLayerNumbU(kk-1)+1;++ii){
                   		sum+=getWeightO((kk-1),jj,ii)*getWeightO((kk-1),jj,ii);
                	}

                	for (int ff = 0; ff<m_miniBatch;ff++){
                                      
                        	avHOut += getLayerUOutput(ff,kk,jj);

                                // Squared back propagated value
	         		avSqTerm +=backPropagateU(ff,kk,jj)*backPropagateU(ff,kk,jj);
	                }
                }

	        for (int ii=0; ii<getLayerNumbU(kk-1)+1; ++ii){
                        // if the input is updated, calculate penalty and send it to update weights function
	        	if (kk == 1) {
                                for (int ff = 0; ff<m_miniBatch;ff++){
                                        // Sum the output of all values in all mini-batches
	        		        avOut += getLayerUOutput(ff,kk-1,ii);
	        	        }

                                // Calculate the penalty for a specific weight
	        	        penalty = 2*avSqTerm*(m_weights[kk-1][jj][ii]+avOut*(1-2*avHOut)*sum);

                                // Update the weights                             
                                updateWeights(kk-1,jj,ii,learningRate,lambda*penalty);

                        // If not the input layer is updated                        
	        	}else{ 
                                
                                updateWeights(kk-1,jj,ii,learningRate,0.0);
                        }
	        }
	    }
	}
	return 0;
}

int ContractiveAutoencoder::updateWeightsFast(int kk,int jj, int ii,double lr,double penalty){

        m_weights[kk][jj][ii] = m_weights[kk][jj][ii] - lr*getGrad(kk+1, jj, ii);
        return 0;
}

int ContractiveAutoencoder::updateWeights(int kk,int jj, int ii,double lr,double penalty){

        m_weights[kk][jj][ii] = m_weights[kk][jj][ii] -  lr*getGrad(kk+1, jj, ii)+penalty; 
        return 0;
}

ContractiveAutoencoder::~ContractiveAutoencoder(){

        // Remove allocated for the layers memory
	for (int ii = 0; ii<m_miniBatch; ++ii){
		for (int jj = 0; jj<m_numberLayers; ++jj){
			delete m_layers[ii][jj];
		}
	}

}
