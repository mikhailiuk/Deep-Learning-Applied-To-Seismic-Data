#include "../headers/Unit.h"


Unit::Unit(){
	m_output = 0.0;
	m_delta  = 0.0;
}
 	
double Unit::backPropagate(){

        //The returned is set to a value passed through the activation
        //The second 1 is to identify that the process is backpropagation 
	return m_activation.activate(m_activationId,1,m_output);
}

double Unit::propagateVal(double val){

        //The output is set to a value passed through the activation
        //The second 0 is to identify that the process is feedforward
	this->setOutput(m_activation.activate(m_activationId,0,val));

	return m_output;
}
