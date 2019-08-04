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
    tensor_2d in;
	tensor_2d weights;
	tensorg_2d weights_grad;
	tdsize in_size, out_size;
	bool train = false;
	bool debug, clip_gradients_flag;

	fc_layer_t( tdsize in_size, tdsize out_size,  bool clip_gradients_flag=false, bool debug_flag = false )
	{
		this->weights = eval(xt::random::rand<float>({out_size.x, in_size.x }, -1, 1));
		this->in_size = in_size;
		this->out_size = out_size;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
	}

	tensor_2d activate( tensor_2d in, bool train )
	{
		if ( train ) this->in = in;  	// Only save `in` while training to save RAM during inference
		auto start = Clock::now();
		tensor_2d out = linalg::dot(in, transpose(this->weights));
		auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Elipsed: "<< elipsed.count() << "s\n";
		return out;
	}

	void fix_weights(float learning_rate)
	{
		update_weight( weights, weights_grad, 1, false, learning_rate );
		update_gradient( weights_grad );
		// if(debug)
		// {
		// 	cout<<"*******new weights for float fc*****\n";
		// 	print_tensor(weights);
		// }
	}

	tensor_2d calc_grads( tensor_2d& grad_next_layer )
	
	// Calculates backward propogation and saves result in `grads_in`. 
	{
		
		tensor_2d temp_grads = linalg::dot(transpose(grad_next_layer), in);
		this->weights_grad = convert_2d_float_to_gradient(temp_grads);

		tensor_2d grads_in = linalg::dot( grad_next_layer, weights );

	    grads_in = eval(grads_in);
		// if(debug)
		// {
		// 	cout<<"**********grads_in for float fc***********\n";
		// 	print_tensor(grads_in);
		// }
		
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
