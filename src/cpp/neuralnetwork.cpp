#include "../headers/NeuralNetwork.h"

NeuralNetwork::NeuralNetwork()
{
        m_flagSwapped = false;
}

double NeuralNetwork::getLSError(double targ, double pred, int jj){
        // Calculate the squared error for the target and prediction
        return m_objective.getLSError(targ,  pred,  jj);        

}


void NeuralNetwork::swapWeights(){
        // Swap the weights (current with the saved copy which gives minimal validation error)
        m_weights.swap(m_weightsMinCp);

}

int NeuralNetwork::makeWeightCpy() {  

        // Save the weights which give the minimum validation error 
        for( int ii = 0; ii < getNumbLayers()-1; ++ii) {
	        for (int jj = 0; jj < getLayerNumbU(ii+1); ++jj) {
		        for (int kk= 0; kk<getLayerNumbU(ii)+1; ++kk) {
		       		m_weightsMinCp[ii][jj][kk] = m_weights[ii][jj][kk];
		        }
		}
        }

        // Set the validation flag to true
        m_flagSwapped = true;

	return 0;
}



int NeuralNetwork::setDropOut(){

        // Shuffle the drop out array for every layer (indicates which units will be missing)
        for (int ii = 0; ii < getNumbLayers()-1; ++ii){
                std::random_shuffle(m_dropOutVec[ii].begin(), m_dropOutVec[ii].end());
        }
        return 0;
}

int NeuralNetwork::feedforward(int mb){

        // Go through every layer
	for (int kk = 0; kk < getNumbLayers()-1; ++kk){

                // Go through every unit in current+1 layer
	        for (int jj = 0; jj < getLayerNumbU(kk+1); ++jj){

                        // Get the dot product (going through the current layer)
                        dotProductProp(mb,kk,jj);
        
                        // If stats flag is enabled and we are in the encoder
                        if (m_statsFlag == enabled && kk<getNumbLayers()-2){
                              // Want to see what each hidden unit output is
                              m_statsHiddenUnits[kk][jj] += (getLayerUOutput(mb,kk+1,jj)/(float)m_itTest);
                        }
	        }
	}
	return 0;
}



int NeuralNetwork::feedforwardNodeParallelCompSparse(int mb){
        // Counters are private thus every thread updates its own local copy and thus explicit synchronization is required
        int kk = 0;
        // Go through every layer
	while(kk < getNumbLayers()-2){

                // Go through every unit in current+1 layer
                #pragma omp for schedule(guided)
	        for (int jj = 0; jj < getLayerNumbU(kk+1); ++jj){

                        // Get the dot product (going through the current layer)
                        dotProductProp(mb,kk,jj);

                        // Get the average of all units used in regularisation for a sparse autoencoder
                        m_avUnits[jj] += (getLayerUOutput(mb,kk+1,jj)/(float)m_miniBatch);
	        }
                ++kk;
	}
	return 0;
}

int NeuralNetwork::feedforwardNodeParallel(int mb){
        // Counters are private thus every thread updates its own local copy and thus explicit synchronization is required
        int kk = 0;
        // Go through every layer
	while(kk < getNumbLayers()-1){

                // Go through every unit in current+1 layer
                #pragma omp for schedule(guided)
	        for (int jj = 0; jj < getLayerNumbU(kk+1); ++jj){
                        dotProductProp(mb,kk,jj);
	        }
                ++kk;
	}
	return 0;
}



double NeuralNetwork::dotProduct(int mb, int kk,int jj){
        double tmp = 0.0;

        //Get the dot product of the neurons in the kk layer with associated weights
        for (int ii = 0; ii < getLayerNumbU(kk)+1; ++ii){
        	tmp += m_weights[kk][jj][ii]*m_layers[mb][kk]->getUnitOutput(ii);
        }

        // Return only real values
        return tmp*m_dropOutVec[kk][jj];
}


void NeuralNetwork::dotProductProp(int mb, int kk,int jj){

     //Propagate the dot product throught the non-linearity
     m_layers[mb][kk+1]->setLayerUOutProp(jj,dotProduct(mb,kk,jj));

}



double NeuralNetwork::deltaComputeNodeParallel (int mb){
        int  tid = omp_get_thread_num();
	double errorder, erroro = 0.0;

        // Set the indeces to not to compute them in every iteration
	int outLayer=getNumbLayers()-1;
        int kk = (outLayer-1);


        //Go through the last layer
        #pragma omp for schedule(guided)
	for (int jj = 0; jj < getLayerNumbU(outLayer); ++jj){

                // Get the error der (1 - backprop, jj - unit, delta, output*mask_value)
    	        errorder = m_objective.objective(1,jj,
                                                 getLayerUDelta(mb,outLayer,jj),
                                                 getLayerUOutput(mb,outLayer,jj))*getLayerUDelta(mb,0,jj);

                // Get the error between prediction and target
   		erroro = erroro+ m_objective.objective(0,
                                                       getLayerUDelta(mb,outLayer,jj),
                                                       getLayerUOutput(mb,outLayer,jj))*getLayerUDelta(mb,0,jj);
                

                // Update the delta of the mini-batch mb, in the last layer outLayer, in the unit jj, with val
                setLayerUDelta(mb,
                               outLayer,
                               jj,
                               errorder*backPropagateU(mb,outLayer,jj));
        }

        // Array holds the error in the locations of the thread ids
        m_sumDelta[tid] = erroro;

        // Sum all the errors from the split workload in serial (cannot use reduction since all the variables are private)
        #pragma omp barrier
        if (tid==0){
                for (int gg = 1; gg < omp_get_num_threads(); ++gg){
                          m_sumDelta[0] += m_sumDelta[gg];
                          m_sumDelta[gg] = 0.0;
                }
        }

        #pragma omp barrier
        erroro= m_sumDelta[0];

        double sumDelta;

        // Go through all the layers backwards
	while (kk > 0){

                // Go through the current layer
                #pragma omp for schedule(guided)
   		for (int jj = 0; jj < getLayerNumbU(kk)+1; ++jj){
        	        sumDelta = 0.0;

                        // Go through the layer closer to the output and comput the sum delta for the current unit in the current layer
                        // \sum delta(ii)*weights(ii,jj)
       		        for (int ii = 0; ii < getLayerNumbU(kk+1); ++ii){
            	                sumDelta = sumDelta+(getLayerUDelta(mb,kk+1,ii))*getWeightO(kk,ii,jj);
        	        }

                        // If there are 3 layers and current layer is the middle and we use a sparse autoencoder
                        if (kk == 1 && outLayer == 2 && m_sparse != 0.0){
                                sumDelta += m_sparse*(-m_sparsityParameter/m_avUnits[jj]+(1-m_sparsityParameter)/(1-m_avUnits[jj]));
                                
                                ///m_avUnits[jj]=0.0; YOU CANNOT 0, beacuse the value is used in the next iteration
                        }

                        // Update the delta
        	        setLayerUDelta(mb,kk,jj,sumDelta*backPropagateU(mb,kk,jj));
		}
                --kk;
	}
	return erroro;
}



double NeuralNetwork::deltaCompute(int mb){

        double errorder;
        double erroro = 0.0;
        int outLayer = getNumbLayers()-1;

        //Go through the last layer
        for (int jj = 0; jj < getLayerNumbU(outLayer); ++jj){

                // Get the error der (1 - backprop, jj - unit, delta, output*mask_value)
    	        errorder = m_objective.objective(1,getLayerUDelta(mb,outLayer,jj),getLayerUOutput(mb,outLayer,jj));

                // Get the error between prediction and target
   		erroro = erroro+ m_objective.objective(0,getLayerUDelta(mb,outLayer,jj),getLayerUOutput(mb,outLayer,jj));

                // Update the delta of the mini-batch mb, in the last layer outLayer, in the unit jj, with val
                setLayerUDelta(mb,outLayer,jj,errorder*backPropagateU(mb,outLayer,jj));
        }

        // Go through all the layers backwards
        for (int kk = (outLayer-1); kk > 0; --kk){

                // Go through the current layer
   		for (int jj = 0; jj < getLayerNumbU(kk)+1; ++jj){
        	        double sumdelta = 0.0;

                        // Go through the layer closer to the output and comput the sum delta for the current unit in the current layer
                        // \sum delta(ii)*weights(ii,jj)
       		        for (int ii = 0; ii < getLayerNumbU(kk+1); ++ii){
            	                sumdelta = sumdelta+(getLayerUDelta(mb,kk+1,ii))*getWeightO(kk,ii,jj);
        	        }

                        // Update the delta
        	        setLayerUDelta(mb,kk,jj,sumdelta*backPropagateU(mb,kk,jj));
	        }
        }
	return erroro;
}

double NeuralNetwork::backPropagateU(int mb,int la,int ii){
        //In the mini-batch mb, in the layer la, apply the derivative of the non-linearity to the units output
        return m_layers[mb][la]->unitBackPropagate(ii);
}

double NeuralNetwork::getGrad(int kk, int jj, int ii){

        // Gradient = \sum delta*output for every mini-batch
        double grad = 0.0;
        for (int ff = 0; ff < m_miniBatch; ++ff){
       	        grad += getLayerUDelta(ff,kk,jj)*getLayerUOutput(ff,kk-1,ii);
	}
        return grad/(double)m_miniBatch;
                        
}

void  NeuralNetwork::setLayerUOutput (int mb, int la, int u, double val){
        m_layers[mb][la]->setUnitOutput(u,val);
}

void  NeuralNetwork::setLayerUDelta(int mb, int jj, int ii, double val){
        m_layers[mb][jj]->setUnitDelta(ii, val);
}


void NeuralNetwork::initialise(ParamsInit parameters)
{
        m_sparse            = parameters.sparse;
        m_sparsityParameter = parameters.sparsityParameter;
        m_dropOut           = parameters.dropOut;
        m_statsFlag         = static_cast<Flag>(parameters.statsFlag);
        m_miniBatch         = parameters.miniBatch;
        m_numberLayers      = parameters.numbLayers;
        m_momentum          = parameters.momentum;
        m_adaGrad           = parameters.adaGrad;
        m_itTest            = parameters.numbItTest;
        m_flagFast          = (m_momentum == 0.0 && m_adaGrad == 0.0) ? enabled : disabled;

        m_objective.initialise(parameters);



        //Layers sizes
        for (int ii = 0; ii < m_numberLayers; ++ii){
                m_layersSizeVec.push_back(parameters.layersVec.at(ii));
        }

        //Sparsity set
        if (m_sparse != 0.0 && m_numberLayers == 3){
                m_avUnits.resize(getLayerNumbU(1)+1, 0.0);
        }
       
        //Layers
        for (int ii = 0; ii < m_miniBatch; ++ii){
                m_layers.push_back(vector< Layer *>());
  	        for (int jj = 0; jj < m_numberLayers; ++jj){
                        m_layers[ii].push_back(new Layer((getLayerNumbU(jj)+1),parameters.actVec.at(jj)));
                }
	}

        /// Drop out initialise	
        initialiseDropOut();

        /// Stats hidden units initialise
        initialiseStatsHiddenUnits();

        /// Bias set
        initialiseDeltaBias(parameters);

        /// Function to Initialise Weights
        initialiseWeights(parameters);

        m_sumDelta.resize(16,0.0);

}

void NeuralNetwork::initialiseDeltaBias(ParamsInit parameters){

        // Go through all mini-batches
	for (int jj = 0; jj < m_miniBatch; ++jj){
                // Go through all layers
		for (int ii = 0; ii < m_numberLayers; ++ii){
                        //Set all deltas to 0
			for (int ff = 0; ff<getLayerNumbU(ii); ++ff){
		  		m_layers[jj][ii]->setUnitDelta(ff,0.0);
	 		}

                        //set all biases
			m_layers[jj][ii]->setUnitOutput(getLayerNumbU(ii),parameters.bias);
		}
	}
}

void NeuralNetwork::initialiseDropOut(){

        // Allocate memory for the drop out vectors
        m_dropOutVec.resize(getNumbLayers()-1);
        for (int ii = 0; ii < getNumbLayers()-1; ++ii){
                m_dropOutVec[ii].resize(getLayerNumbU(ii+1),1.0);        
        }

        // If there is a drop out rate
        if (m_dropOut != 0.0){

                // go through the middle layers (not touching output and input)
                for (int ii = 0; ii < getNumbLayers()-2; ++ii){

                        // set #unitsInTheL*dropOutRate to 0
                        for (int jj = 0; jj<(int)(getLayerNumbU(ii+1)*m_dropOut);++jj){
                                m_dropOutVec[ii][jj] = 0.0;
                        }
                }
        }
}

void NeuralNetwork::initialiseStatsHiddenUnits(){

        // If gathering statistics
        if (m_statsFlag == enabled){
                // Allocate memory for the middle layers only
                m_statsHiddenUnits.resize(m_numberLayers-2);
                for (int ii = 0 ; ii < m_numberLayers-2; ++ii){
                       m_statsHiddenUnits[ii].resize(getLayerNumbU(ii+1),0.0);             
                }
        } 


}



void NeuralNetwork::initialiseWeights(ParamsInit parameters){

        // Get the numer of weight layers
        int numberWeights = getNumbLayers() - 1;


        // Allocate memory
        m_weights.resize(numberWeights);
        m_weightsMinCp.resize(numberWeights);
        m_momentumArr.resize(numberWeights);
        m_sumDeltaArr.resize(numberWeights);

	for (int ii = 0; ii < numberWeights; ++ii) {
	  	m_weights[ii].resize(getLayerNumbU(ii+1));
                m_weightsMinCp[ii].resize(getLayerNumbU(ii+1));
	  	m_momentumArr[ii].resize(getLayerNumbU(ii+1));
                m_sumDeltaArr[ii].resize(getLayerNumbU(ii+1));
	  	for (int jj = 0; jj <getLayerNumbU(ii+1); ++jj){
                        m_weights[ii][jj].resize(getLayerNumbU(ii)+1);
                        m_weightsMinCp[ii][jj].resize(getLayerNumbU(ii)+1);
                        m_momentumArr[ii][jj].resize(getLayerNumbU(ii)+1);
                        m_sumDeltaArr[ii][jj].resize(getLayerNumbU(ii)+1);
	  	}
	}

        // Set the randomness
        srand(time(NULL));


        // Go through all weights
	for (int ii = 0;ii < numberWeights; ++ii){
		for (int jj = 0; jj < getLayer(0,ii+1)->getNumbUnits(); ++jj){
			for (int kk= 0; kk<getLayer(0,ii)->getNumbUnits()+1; ++kk){
                                // Random number
	    		        double r = static_cast <double> (rand()) / static_cast <double> (RAND_MAX);

                                // Scale the random number
	    		        (m_weights)[ii][jj][kk] =(-1.0+2.0*r)*parameters.weightMagnitude*parameters.randomFlag+(1.0-parameters.randomFlag)*parameters.weightMagnitude;

                                // Set other arrays to 0
                                (m_momentumArr)[ii][jj][kk]  = 0.0;
                                (m_sumDeltaArr)[ii][jj][kk]  = 0.0;
                                (m_weightsMinCp)[ii][jj][kk] = 0.0;
	    	}
	    }
	}
}


void NeuralNetwork::initWeightsFromFile(std::string folderName){


        // Go through all layers of weights
        for( int ii = 0; ii < getNumbLayers()-1; ++ii){


                // Open the file containing required weights
                std::string s = std::to_string(ii);
                std::string fileName = folderName+"/weights"+s+".dat";
                std::fstream myfilewin(fileName, std::ios_base::in);
                

                // If file is fine
                if (myfilewin.good()){
                        // Read the parameters
                        int sizeii,sizeiip1,tmp;
                        myfilewin>>sizeii;
                        myfilewin>>sizeiip1;
                        myfilewin>>tmp;
                        myfilewin>>tmp;
                        myfilewin>>tmp;

                        // Check whether dimentions in the file are the same as used in the setting 
                        if (sizeiip1 == getLayerNumbU(ii+1) && sizeii == getLayerNumbU(ii)+1){

                                // Read the weights from the file intro the variable
                                for (int jj = 0; jj < getLayerNumbU(ii+1); ++jj){
                                        for (int kk= 0; kk < getLayerNumbU(ii)+1; ++kk){
                                   		myfilewin >> m_weights[ii][jj][kk];
                                         }
                                }
                                myfilewin.close();
                        }else{
                                printf("Input dimentions of the weights do not match\n");
                                exit(0);
                        }
                }else{
                        printf("Skipping weights %d\n", ii);
                }
        }
        std::cout<<"Inititlaised weights from folder::"<<folderName<<"\n";

}

NeuralNetwork::~NeuralNetwork(){


}


