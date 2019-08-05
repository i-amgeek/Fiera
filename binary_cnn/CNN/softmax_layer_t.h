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
	xarray<float> out;
	bool debug, clip_gradients_flag;	
	softmax_layer_t( tdsize in_size, bool to_normalize=true,bool clip_gradients_flag = true, bool debug_flag = false)
	{
		this->in_size = in_size;
		this->out_size = in_size;
		this->to_normalize = to_normalize;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
	}
	
	xarray<float> activate( xarray<float> in, bool train = true)
	{	
		#ifdef measure_time
		auto start = Clock::now();
		#endif

		xarray<float> sum_ = xt::sum(xt::exp(in), {-1});
		if(train) this->out = out;	
		if (train) {
			xarray<float> out = xt::exp(in) / xt::view(sum_, all(), newaxis()); 
			this->out = out;
			out = xt::log(out);
		}
		else
			xarray<float> out = in - xt::view(xt::log(sum_), all(), newaxis() );
		

		#ifdef measure_time
		auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Softmax Forward Elipsed: "<< elipsed.count() << "s\n";
		#endif
		return out;
	}
	
	
	void fix_weights(float learning_rate)
	{
		
	}
	
	xarray<float> calc_grads( xarray<float>& grad_next_layer )
	{
		#ifdef measure_time
		auto start = Clock::now();
		#endif

		float m = grad_next_layer.shape()[0];
		auto target = xt::argmax(grad_next_layer, 1);
		xarray<float> grads_in( {m, in_size.x});

		grads_in = out / m;
		for (int e=0; e<m; e++){
			grads_in(e, target(e)) = (out(e, target(e)) - 1) / m;
		}	

		#ifdef measure_time
		auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Softmax Backward Elipsed: "<< elipsed.count() << "s\n";
		#endif

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
