// Author: Aliaksei Mikhailiuk, 2017.

#include "../headers/Layer.h"    


Layer::Layer(int numb,int act){
        // Number of units is less by 1 than the number passed as numb is with bias
	m_numberOfUnits = numb-1;

        // Allocate memory
        m_layer.resize(numb);

        // Set activation functions
        for (int ii=0;ii<m_numberOfUnits;++ii){
                m_layer[ii].setAct(act);
        }
        setActivation(act);
}


double Layer::unitBackPropagate(int unitId){

        // Get the derivative of the activation function in the unit
       return m_layer[unitId].backPropagate();
}

void Layer::setLayerUOutProp(int unitId,double value){

        // Feedforward the value set to the input of the unit
        m_layer[unitId].propagateVal(value);       
}


Layer::~Layer(){
}
