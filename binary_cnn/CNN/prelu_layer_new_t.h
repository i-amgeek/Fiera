/*Implement parametric ReLU activation function
   It follows:
    f(x) = alpha * x	if x < 0
    f(x) = x			if x >= 0
    where alpha is a learnable parameter

*/
#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct prelu_layer_t
{
	layer_type type = layer_type::prelu;
	xarray<float> in;
	float alpha;
	gradient_t grads_alpha;
	float prelu_zero;		// Differential of PReLU is undefined at 0. 'p_relu_zero' defines value to be used instead.
	bool debug,clip_gradients_flag;
	tdsize in_size;
	tdsize out_size;

	prelu_layer_t( tdsize in_size, bool clip_gradients_flag = false, bool flag_debug = false )
	/**
	* 
	* Parameters
	* ----------
	* in_size : (int m, int x, int y, int z)
	* 		Size of input matrix.
	*
	* clip_gradients_flag : bool
	* 		Whether gradients have to be clipped or not
	* 
	* debug_flag : bool
	* 		Whether to print variables for debugging purpose
	*
	**/
	{
		alpha=0.05;
		prelu_zero = 0.5;
		this->in_size = in_size;
		this->out_size = in_size;
		this->debug=flag_debug;
		this->clip_gradients_flag = clip_gradients_flag;
	}

	tensor_4d activate(tensor_4d& in, bool train)

	//  `activate` FORWARD PROPOGATES AND SAVES THE RESULT IN `out` VARIABLE.
	{
		if (train) this->in = in;
		tensor_4d out = xt::where(in < 0, in * alpha, in);

		return out;
	}

	tensor_2d activate(tensor_2d& in, bool train)

	//  `activate` FORWARD PROPOGATES AND SAVES THE RESULT IN `out` VARIABLE.
	{
		if (train) this->in = in;
		tensor_2d out = xt::where(in < 0, in * alpha, in);

		return out;
	}
	// void fix_weights(float learning_rate)
	// {
	// 	// grads_alpha contains sum of gradients of alphas for all examples. 
	// 	// grads_alpha.grad /= out.size.m;
	// 	alpha = update_weight(alpha,grads_alpha,1,false, learning_rate);
	// 	update_gradient(grads_alpha);
		
	// 	// if(debug)
	// 	// {
	// 	// 	cout<<"*******updated alpha for prelu*****\n";
	// 	// 	cout<<alpha<<endl;
	// 	// }
	// }

	// tensor_2d calc_grads( tensor_2d& grad_next_layer )
	// {
	// 	assert(in.size > 0);

	// 	tensor_2d grads_in = xt::where(in > 0.0, grad_next_layer, grad_next_layer * prelu_zero);

	// 	auto mask = in < 0.0;
	// 	grads_in(mask) = grad_next_layer(mask) * alpha;
	// 	grads_alpha.grad = sum(grad_next_layer(mask) * in(mask));
		
	// 	// if(debug)
	// 	// {
	// 	// 	cout<<"***********grads_in for prelu********\n";
  	//     // 	print_tensor(grads_in);
	// 	// 	cout<<"*********grad alpha***********\n";
	// 	// 	cout<<grads_alpha.grad<<endl;
	// 	// }
	// 	return grads_in;
	// }

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "prelu" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "prelu_zero", prelu_zero},
			{ "clip_gradients", clip_gradients_flag}
		} );
	}

	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		json j = {{"type", "prelu"},
			  	  {"alpha", alpha}};
		file << j;
		file.close();
	}

	void load_layer_weight(string fileName){
		ifstream file(fileName);
		json weight;
		file >> weight;
		assert(weight(type) == "prelu");
		alpha = weight["alpha"];
	}
	void print_layer(){
		cout << "\n\n PReLU Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
		cout << "\n\t alpha:\t\t" << alpha;
	}
};
#pragma pack(pop)
