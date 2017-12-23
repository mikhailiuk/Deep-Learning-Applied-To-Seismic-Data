#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <math.h>  

/*! \class Activation
    \brief Holds activation functions

    Used to compute feedforaward and backpropagation activation of the neurons
*/
class Activation{
        public:

        //! function called by a neuron. Decides which activation function should be used
        //! \param activation id of the neuron activation function
        //! \param stage either feedforward or backpropagate
        //! \x value to be updated
        double activate (int activation,int stage,double x)const;

        //! gausian activation function
        inline double gaus(double x)            const;

        //! derivative of gausian activation function
        inline double gausder(double x)         const;

        //! softplus activation function
        inline double softplus(double x)        const;

        //! derivative of softplus activation function
        inline double softplusder(double x)     const;

        //! sigmoid activation function
        inline double sigmf(double x)           const;

        //! derivative of a sigmoid activation function
        inline double sigmfder(double x)        const;

        //! linear activation function
        inline double linef(double x)           const;

        //! derivative of a linear activation function
        inline double linefder()        const;

        //! rectifier activation function
        inline double rectifier(double x)       const;

        //! derivative of rectifier activation function
        inline double rectifierder(double x)    const;

        //! steprectifier activation function
        inline double steprectifier(double x)   const;

        //! derivative of steprectifier activation function
        inline double steprectifierder(double x)const;

        //! inversetangent activation function
        inline double invtg(double x)           const;

        //! derivative of an inversetangent activation function
        inline double invtgder(double x)        const;

};

#endif
