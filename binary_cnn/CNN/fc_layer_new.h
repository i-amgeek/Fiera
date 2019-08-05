#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "tensor.hpp"
#include "optimization_method.h"
#include "gradient_t.h"
#include "utils/extras.hpp"

#pragma pack(push, 1)
struct fc_layer_t
{
	layer_type type = layer_type::fc;
    xarray<float> in;
	xarray<float> weights;
	xarray<gradient_t> weights_grad;
	tdsize in_size, out_size;
	bool train = false;
	bool debug, clip_gradients_flag;

	fc_layer_t( tdsize in_size, tdsize out_size,  bool clip_gradients_flag=false, bool debug_flag = false )
	{
		this->weights = eval(xt::random::rand<float>({out_size.x, in_size.x * in_size.y * in_size.z }, -1, 1));
		this->in_size = in_size;
		this->out_size = out_size;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
	}

	xarray<float> activate( xarray<float> in, bool train )
	{
		#ifdef measure_time
		auto start = Clock::now();
		#endif

		if ( train ) this->in = in;  	// Only save `in` while training to save RAM during inference
		xarray<float> out = linalg::dot(in, transpose(this->weights));

		#ifdef measure_time
		auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "FC Forward Elipsed: "<< elipsed.count() << "s\n";
		#endif

		return out;
	}

	void fix_weights(float learning_rate)
	{
		update_weight( weights, weights_grad, 1, false, learning_rate );
		update_gradient( weights_grad );
	}

	xarray<float> calc_grads( xarray<float>& grad_next_layer )
	
	// Calculates backward propogation and saves result in `grads_in`. 
	{
		#ifdef measure_time
		auto start = Clock::now();
		#endif
			
		xarray<float> temp_grads = linalg::dot(transpose(grad_next_layer), in);
		this->weights_grad = convert_2d_float_to_gradient(temp_grads);

		xarray<float> grads_in = linalg::dot( grad_next_layer, weights );

	    grads_in = eval(grads_in);
		
		#ifdef measure_time
		auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "FC Backward Elipsed: "<< elipsed.count() << "s\n";
		#endif
		return grads_in;	

	}
	
	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "fc" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "out_size", {out_size.m, out_size.x, out_size.y, out_size.z} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}

	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		json weight;
		weight["type"] = "fc"; 
		xt::to_json(weight["data"], this->weights);
		file << weight << endl;
		file.close();
	}

	void load_layer_weight(string fileName){
		ifstream file(fileName);
		json weight;
		file >> weight;
		assert(weight["type"] == "fc");
		xt::from_json(weight["data"], this->weights);
		file.close();
	}

	void print_layer(){
		cout << "\n\n FC Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)
