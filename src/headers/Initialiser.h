// Author: Aliaksei Mikhailiuk, 2017.

#ifndef INITIALISER_H
#define INITIALISER_H

#include <libconfig.h++>
#include <string.h> 
#include <vector>
#include "TrainingAlgorithm.h"
#include "NeuralNetwork.h"
#include "ContractiveAutoencoder.h"
#include "Autoencoder.h"
#include "ParamsInit.h"

using namespace libconfig;
using std::vector;


/*! \class Initialiser
    \brief Class to write from .cfg file

    Reads parameters from the .cfg file and initialises Training algorithm and Neural Network
*/
class Initialiser{

private:
public:
//! Function reading .cfg file.
        int initialise(NeuralNetwork * &neuralNetwork,TrainingAlgorithm *algorithm);
};

#endif
