// Author: Aliaksei Mikhailiuk, 2017.

#ifndef UNIT_H
#define UNIT_H
 
#include "../headers/Activation.h"

/*! \class Unit
    \brief Encapsulates information and functions associated with neurons

    Units are separate entities, have own output and delta values, every neuron
    also has an associated activation function.
*/
class Unit
{

//! Let Layer have access to private fields to remove redundunt function calls.
        friend class Layer;

private:

//! Output valuer of the neuron
    double m_output;

//! Delta is used to store contribution of the unit towards overall error.
    double m_delta;

//! Every unit has a member activation function to use functionality
    Activation m_activation;

//!  Every unit has an activation function associated with it (id from 1 to 7).
    int m_activationId;


public:

//! Constructor default
    Unit();

//! Function returns the value of the output passed through the derivative of the activation function
    double backPropagate();

//! Function aplies activation to the supplied value and sets output of the neuron to this value
    double propagateVal(double val);

//! Function to set output
    void setOutput      (double output)         {m_output = output;};

//! Function to set delta
    void setDelta       (double delta)          {m_delta = delta;};

//! Function to set activation function
    void setAct         (int activation)        {m_activationId = activation;};

//! Function returns the value of the ouput field of the neuron
    double getOutput()  const   {return m_output;};

//! Function returns the value of the delta field of the neuron
    double getDelta()   const   {return m_delta;};

};
 
#endif
