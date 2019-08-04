/*
To implement softmax 
*/

#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"
#include "tensor_bin_t.h"

#pragma pack(push, 1)
struct softmax_layer_t
{
	layer_type type = layer_type::softmax;
	tdsize in_size, out_size;
	bool to_normalize;
	tensor_2d out;
	xarray<float> sum_;
	bool debug, clip_gradients_flag;	
	softmax_layer_t( tdsize in_size, bool to_normalize=true,bool clip_gradients_flag = true, bool debug_flag = false)
	{
		this->in_size = in_size;
		this->out_size = in_size;
		this->to_normalize = to_normalize;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
	}
	
	tensor_2d activate( tensor_2d in, bool train = true)
	{	xarray<float> sum_ = xt::sum(xt::exp(in), {-1});
		tensor_2d out = in - xt::view(xt::log(sum_), all(), newaxis() );
		if(train) this->out = out;	
		if (train) {
			out = xt::exp(in) / xt::view(sum_), all(), newaxis()); 
			this->sum_ = sum_;
		}
		
		return out;

	}
	
	
	void fix_weights(float learning_rate)
	{
		
	}
	
	tensor_2d calc_grads( tensor_2d& grad_next_layer )
	{
		
		float m = grad_next_layer.shape()[0];
		tensor_1d target = argmax(grad_next_layer, 1);
		
		tensor_2d grads_in( {m, in_size.x});

		// for(int e = 0; e < m; e++){
		// 	int idx;
		// 	for(int i=0; i<out_size.x; i++){
		// 		if(int(grad_next_layer(e,i,0,0)) == 1){
		// 			idx = i;
		// 			grads_in(e,i,0,0) = -(1-out(e,i,0,0))/m;
		// 		}
		// 	}
		// 	for(int i=0; i<out_size.x; i++){
		//  		if(idx!=i){
		// 			grads_in(e,i,0,0) = ((out(e,i,0,0))/m);
		// 		}
		// 		clip_gradients(clip_gradients_flag, grads_in(e,i,0,0));
				
		// 	}	
		// }
		grads_in = out / m;
		for (int e=0; e<m; e++){
			grads_in(e, target(e)) = (out(e, target(e)) - 1) / m;
		}	
		// if(debug)
		// {
		// 	cout<<"********grads_in for softmax*********\n";
		// 	print_tensor(grads_in);
		// }

		return grads_in;
	}

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "softmax" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "to_normalize", to_normalize },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}	

	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		json j = {{"type", "softmax"}};
		file << j;
		file.close();
	}

	void load_layer_weight(string fileName){

	}
	void print_layer(){
		cout << "\n\n Softmax Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)
