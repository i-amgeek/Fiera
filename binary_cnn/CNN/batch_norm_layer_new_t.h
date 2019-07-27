#pragma once
#include<math.h>
#include "layer_t.h"

#pragma pack(push, 1)
struct batch_norm_layer_t
{
	layer_type type = layer_type::batch_norm;
	tensor_4d temp_diff;
	tensor_4d in_hat;
	tdsize in_size, out_size;
	float epsilon, momentum;
	tensor_1d gamma, beta, sqrt_sigma;
	tensorg_1d grads_beta, grads_gamma;
    bool adjust_variance;
	bool debug, clip_gradients_flag;
	tensor_1d running_mean, running_var;

	batch_norm_layer_t(tdsize in_size,bool clip_gradients_flag = true, bool debug_flag = false)
	{
		this->in_size = in_size;
		this->out_size = in_size;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
		this->epsilon = 1e-5;
		this->momentum = 0.9;
        this->adjust_variance = false;
		this->running_mean = tensor_1d::from_shape({(uint)in_size.z});
		this->running_var = tensor_1d::from_shape({(uint)in_size.z});
	}

	auto make1Dto4D(tensor_1d in){
		return xt::view(in, newaxis(), all(), newaxis(), newaxis());
	}

	tensor_4d activate(tensor_4d& in, bool train = true){

		if(train) {

			tensor_1d u_mean = xt::mean(in, {0,2,3});
			temp_diff = in - make1Dto4D(u_mean);
			tensor_1d sigma = xt::mean(temp_diff * temp_diff, {0, 2, 3});

			if(in_size.m > 1 and adjust_variance)
				sigma *= in_size.m / (in_size.m - 1);

			sqrt_sigma = xt::sqrt(sigma + epsilon);
			in_hat = temp_diff / make1Dto4D(sqrt_sigma);
			tensor_4d out = make1Dto4D(gamma) * in_hat + beta;

			running_mean = momentum * running_mean + (1.0 - momentum) * u_mean;
			running_var = momentum * running_var + (1.0 - momentum) * sigma;
			return out;
		}
		else{
			tensor_4d in_hat = (in - make1Dto4D(running_mean)) / make1Dto4D(xt::sqrt(running_var + epsilon));
			tensor_4d out = gamma * in_hat + beta;
			return out;
		}
	}
	
	
	// void fix_weights(float learning_rate){
		
	// 	beta = update_weight(beta, grads_beta, 1, false, learning_rate);
	// 	update_gradient(grads_beta);
	// 	gamma = update_weight(gamma,grads_gamma,1,false, learning_rate);
	// 	update_gradient(grads_gamma);
	// }

	tensor_4d calc_grads( tensor_4d& grad_next_layer)
	{
		assert(in_hat.size() > 0); 
		tensorg_1d grads_beta = xt::sum(grad_next_layer, {0,2,3});
		tensorg_1d grads_gamma = xt::sum(in_hat * grad_next_layer, {0,2,3});
		tensor_4d grads_in_hat = grad_next_layer * make1Dto4D(gamma);

		tensor_1d grads_sqrtvar = -1.0 * xt::sum(grads_in_hat * temp_diff, {0,2,3}) / (sqrt_sigma * sqrt_sigma);

		tensor_1d grads_var = 0.5 * grads_sqrtvar / sqrt_sigma;

		tensor_4d grads_xmul1 = grad_next_layer * make1Dto4D(gamma) / make1Dto4D(sqrt_sigma);
		tensor_4d grads_xmul2 = make1Dto4D(grads_var) * 2 * temp_diff / (out_size.m * out_size.x * out_size.y);
		tensor_4d grads_x1 = grads_xmul1 + grads_xmul2;
		tensor_1d grads_u_mean = xt::sum(-grads_x1, {0,2,3});
		tensor_4d grads_in = grads_x1 + make1Dto4D(grads_u_mean) / (out_size.m * out_size.x * out_size.y);


		// if(debug)
		// {
		// 	cout<<"\n*********grads_in for batch_norm************\n";
		// 	print_tensor(grads_in);
		// }

		return grads_in;	
	}

	// void save_layer( json& model ){
	// 	model["layers"].push_back( {
	// 		{ "layer_type", "batch_norm2D" },
	// 		{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
	// 		{ "clip_gradients", clip_gradients_flag}
	// 	} );
	// }
	
	// void save_layer_weight( string fileName ){
	// 	ofstream file(fileName);
	// 	json weights = {
	// 		{ "epsilon", epsilon },
	// 		{ "beta", beta },
	// 		{ "gamma", gamma }
	// 	};
	// 	file << weights;
	// }

	// void load_layer_weight(string fileName){
	// 	ifstream file(fileName);
	// 	json weights;
	// 	file >> weights;
	// 	this->epsilon = weights["epsilon"];
	// 	vector<float> beta = weights["beta"];
	// 	vector<float> gamma = weights["gamma"];
	// 	this->beta = beta;
	// 	this->gamma = gamma;
	// 	file.close();
	// }

	// void print_layer(){
	// 	cout << "\n\n Batch Normalization Layer : \t";
	// 	cout << "\n\t in_size:\t";
	// 	print_tensor_size(in_size);
	// 	cout << "\n\t out_size:\t";
	// 	print_tensor_size(out_size);
	// }
};
#pragma pack(pop)
