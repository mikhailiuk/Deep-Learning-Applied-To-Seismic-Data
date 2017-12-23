// Author: Aliaksei Mikhailiuk, 2017.

#ifndef LAYER_H
#define LAYER_H
 
#include "../headers/Activation.h"
#include "../headers/Unit.h"
#include <iostream> 
#include <new> 
#include <vector>

using std::vector;


/*! \class Layer
    \brief Encapsulates information and functions associated with layers

    Layer is composed of Units (Neurons) has associated activation funciton (if all the neurons
   within this layer share the same). 
*/
class Layer
{
private:

        //! Id of the activation function if the same for all units in the layer
        int m_activation;

        //! Number of units
        int m_numberOfUnits;

        //!
        vector <Unit> m_layer;

public:

        //! Constructor
        //! \param numb - number of units with bias
        //! \param act - activation function 
        Layer(int numb,int activation);

        //! Destructor
        ~Layer();

        //! Function set the value of the unit and feedforward it
        //! \param unitId - id of a unit
        //! \param value - value set as an input to the unit
        void setLayerUOutProp(int unitId,double value);
 
        //! Function to pass the value of the unit in the backpropagation algorith
        //! \param unit id
        double unitBackPropagate(int ii);

        void setUnitOutput(int ii, double value)         {m_layer[ii].setOutput(value);};

        void setUnitDelta(int ii, double value)          {m_layer[ii].setDelta(value);}; 

        void setActivation (int activation)    {m_activation = activation;};

        void setNumbUnits(int numberOfUnits)     {m_numberOfUnits = numberOfUnits;};

        double getUnitOutput(int ii)      const  {return m_layer[ii].m_output;};

        double getUnitDelta(int ii)       const  {return m_layer[ii].m_delta;};

        int getNumbUnits()                const  {return m_numberOfUnits;};

        double getActivation()            const  {return m_activation; };

};
 
#endif
