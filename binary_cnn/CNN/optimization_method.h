#pragma once
#include "gradient_t.h"
#include "tensor.hpp"

#define MOMENTUM 0.6
#define WEIGHT_DECAY 0.000

static float update_weight( float &w, gradient_t& grad, float multp, bool clip, float learning_rate)
{

	float m = (grad.grad + grad.oldgrad * MOMENTUM);
	w -= learning_rate  * m * multp +
		 learning_rate * WEIGHT_DECAY * w;
		 
		 if ( w < -1 and clip) 	  w = -1.0;
		 else if (w > 1 and clip) w = 1.0;
	return w;

}

void update_weight( tensor_2d& weights, tensorg_2d weights_grad, float multp, bool clip, float learning_rate ){
	int h = weights.shape()[0];
	int w = weights.shape()[1];

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++){
			gradient_t& grad = weights_grad(i,j);
			float m = grad.grad + grad.oldgrad * MOMENTUM;
			weights(i,j) -= learning_rate * m * multp + learning_rate * WEIGHT_DECAY * w;
		}
}

static void update_gradient( gradient_t& grad )
{
	grad.oldgrad = (grad.grad + grad.oldgrad * MOMENTUM);
}

static void update_gradient( tensorg_2d weights_grad){

	int h = weights_grad.shape()[0];
	int w = weights_grad.shape()[1];

	for (int i=0; i<h; i++)
		for (int j=0; j<w; j++){
			gradient_t &w = weights_grad(i, j);
			w.oldgrad = w.grad + w.oldgrad * MOMENTUM;
		}
}

void clip_gradients(bool chk, float & gradient_value){

	if(chk and gradient_value > 1e2)
		gradient_value = 1e2;

	else if(chk and gradient_value < -1e2)
		gradient_value = -1e2;
}
