#include "../headers/main.h"

int main(){

	//std::unique_ptr<NeuralNetwork> NN(new NeuralNetwork ());
        //std::unique_ptr<TrainingAlgorithm> Alg(new TrainingAlgorithm());

        // Create an instance of the writer, which will write results of the execution at the end
	OutWriter writer;

        Timer timer;

        // Creat an object which will read .cfg file
        Initialiser initialiser;

        // Pointer to Neural Network, will be initialised by initialiser
	NeuralNetwork *neuralNetwork;
        
        // Pointer to Training Algorithm
	TrainingAlgorithm *algorithm = new TrainingAlgorithm();


        // Read .cfg file and initialise Neural Network and training algrorithm
        initialiser.initialise(neuralNetwork,algorithm);

        // Time the algorithm

        timer.startTiming(); 

        algorithm->trainNeuralNetworkTaskParallel(neuralNetwork);

        // End timing
        std::cout<<"Time total : "<<timer.endTiming()<<std::endl;
 
        // Test trained Neural Network
        algorithm->testNeuralNetwork(neuralNetwork);
    
        // Write results
        writer.write(neuralNetwork,algorithm,timer.getElapsedTime(),"fu");
        
        if (neuralNetwork->getSwapped()){
                neuralNetwork->swapWeights();
                // Test trained Neural Network
                algorithm->testNeuralNetwork(neuralNetwork);
                // Write results
                writer.write(neuralNetwork,algorithm,timer.getElapsedTime(),"es");
        }

        // Clean Up
        delete neuralNetwork;
        delete algorithm;
	return 0;

}

