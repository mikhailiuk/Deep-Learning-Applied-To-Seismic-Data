#include "../headers/Activation.h"


// Function to chose activation of the value x on either backpropagation or feedforward stages
// stage is 0, when feedforward and 1 when backpropagate.
double Activation::activate(int activation,int stage,double x)const{
	 if (activation==1){
	     return (1-stage)*rectifier(x)+stage*rectifierder(x);
	} else if (activation==2){
	     return (1-stage)*sigmf(x)+stage*sigmfder(x);
	} else if (activation==3){
	     return (1-stage)*softplus(x)+stage*softplusder(x);		
	} else if (activation==4){
	     return (1-stage)*invtg(x)+stage*invtgder(x);
	} else if (activation==5){
	     return (1-stage)*linef(x)+stage*linefder();
	} else if (activation==6){
	     return (1-stage)*gaus(x)+stage*gausder(x);
	} else {
	     return (1-stage)*steprectifier(x)+stage*steprectifierder(x);
	} 
	return 0.0;
}

double Activation::softplus(double x)const{
	return log(1+exp(x));
}

double Activation::softplusder(double x)const{
	return 1/(1+exp(-x));
}

double Activation::gaus(double x)const{
	return exp(-x*x);
}

double Activation::gausder(double x)const{
	return -2*x*exp(-x*x);
}

double Activation::rectifier (double x)const{
	if (x<0.0) {
		return 0.0;
	} else {
		return x;
	}
}

double Activation::rectifierder (double x)const{
	if (x<0.0) {
		return 0.0;
	} else {
		return 1.0;
	}
}

double Activation::sigmf(double x)const{
	return 1/(1+exp(-x));
}

double Activation::sigmfder(double x)const{
	return x*(1-x);
}

double Activation::linef(double x)const{
 	return x;
}

double Activation::linefder()const{
	return 1.0;
}

double Activation::steprectifier(double x)const{
	if (x<0.0) {
		return 0.0;
	} else if (x>1) {
		return 1.0;
	} else {
	   return x;	
	}
}

double Activation::steprectifierder(double x)const{
	if (x<0.0 || x>1.0){
		return 0.0;
	} else{
	   return 1.0;
	}
}

double Activation::invtgder(double x)const{
	return 1.0/(x*x+1.0);
}

double Activation::invtg(double x)const{
	return atan(x);
}


