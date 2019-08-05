/*
To implement softmax 
*/

#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"

#pragma pack(push, 1)
struct flatten_t
{
	layer_type type = layer_type::flatten;
	tdsize in_size, out_size;
    bool debug;
	flatten_t( tdsize in_size, bool train = true, bool debug_flag = false)
	{
		this->in_size = in_size;
		this->out_size = {in_size.m, in_size.x * in_size.y * in_size.z, 1,1};
		this->debug = debug_flag;
	}
	
	xarray<float> activate( xarray<float> in, bool train = true)
	{	

        assert(in.dimension() == 4);
        in.reshape({in.shape()[0], out_size.x});
		return in;

	}
	
	
	void fix_weights(float learning_rate)
	{
		
	}
	
	xarray<float> calc_grads( xarray<float>& grad_next_layer )
	{
        assert(grad_next_layer.dimension() == 2);
        int m = grad_next_layer.shape()[0];
        xarray<float> grads_in = xt::reshape_view(grad_next_layer, {m, in_size.z, in_size.y, in_size.x});
		return grads_in;
	}

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "flatten" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} }
		} );
	}	

	void save_layer_weight( string fileName ){
	}

	void load_layer_weight(string fileName){

	}
	void print_layer(){
		cout << "\n\n Flatten Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)
