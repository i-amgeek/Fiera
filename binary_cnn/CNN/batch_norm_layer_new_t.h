#pragma once
#include<math.h>
#include "layer_t.h"

#pragma pack(push, 1)
struct batch_norm_layer_t
{
	layer_type type = layer_type::batch_norm;
	tensor_4d in;
	tensor_4d in_hat,out;
	tdsize in_size, out_size;
	float epsilon;
	tensor_1d gamma, beta, u_mean, sigma;
	tensorg_1d grads_beta, grads_gamma;
    bool adjust_variance;
	bool debug, clip_gradients_flag;

	batch_norm_layer_t(tdsize in_size,bool clip_gradients_flag = true, bool debug_flag = false)
	{
		this->in_size = in_size;
		this->out_size = in_size;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
		epsilon = 1e-5;
        adjust_variance = false;
	}

	tensor_4d activate(tensor_4d& in, bool train = true){

		if(train) {

			this->in = in;
		}
			
			u_mean = xt::mean(in, {0,2,3});

			tensor_4d temp_diff = in - xt::reshape_view(u_mean, {1, in_size.z, 1, 1});
			sigma = xt::mean(temp_diff * temp_diff, {0, 2, 3});
			
			if(in_size.m > 1 and adjust_variance)
				sigma *= in_size.m / (in_size.m - 1);

			in_hat = (in - xt::view(u_mean, newaxis(), all(), newaxis(), newaxis() )) / xt::view(xt::sqrt(sigma + epsilon), newaxis(), all(), newaxis(), newaxis() );


			out = xt::view(gamma, newaxis(), all(), newaxis(), newaxis()) * in_hat + beta;
			if(train) {
				this->out = out;
				this->in_hat = in_hat;
			}
		return out;
	}
	
	
	// void fix_weights(float learning_rate){
		
	// 	beta = update_weight(beta, grads_beta, 1, false, learning_rate);
	// 	update_gradient(grads_beta);
	// 	gamma = update_weight(gamma,grads_gamma,1,false, learning_rate);
	// 	update_gradient(grads_gamma);
	// }

	// tensor_2d calc_grads( tensor_4d& grad_next_layer)
	// {
	// 	assert(in.size > 0); 

	// 	tensorg_4d grads_gamma = xt::sum(in_hat * grad_next_layer, {0,2,3});
	// 	tensor_4d grads_in_hat = grad_next_layer * gamma;
	// 	tensor_4d grads_sqrtvar = -1.0 * xt::sum(grads_in_hat * (in - u_mean));
	// 	tensor_4d grads_var = 0.5 * grads_sqrtvar / xt::sqrt( epsilon + sigma );

	// 	tensor_4d grads_xmul1 = grad_next_layer * gamma / xt::sqrt(epsilon + sigma);
	// 	tensor_4d grads_xmul2 = - grads_var * 2 * (in - u_mean) / (out_size.m * out_size.x * out_size.y);
	// 	tensor_4d grads_x1 = grads_xmul1 + grads_xmul2;
	// 	tensor_4d grads_in = grads_x1 + grads_u_mean / (out_size.m * out_size.x * out_size.y);

	// 	// if(debug)
	// 	// {
	// 	// 	cout<<"\n*********grads_in for batch_norm************\n";
	// 	// 	print_tensor(grads_in);
	// 	// }

	// 	return grads_in;	
	// }

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "batch_norm2D" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}
	
	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		json weights = {
			{ "epsilon", epsilon },
			{ "beta", beta },
			{ "gamma", gamma }
		};
		file << weights;
	}

	void load_layer_weight(string fileName){
		ifstream file(fileName);
		json weights;
		file >> weights;
		this->epsilon = weights["epsilon"];
		vector<float> beta = weights["beta"];
		vector<float> gamma = weights["gamma"];
		this->beta = beta;
		this->gamma = gamma;
		file.close();
	}

	void print_layer(){
		cout << "\n\n Batch Normalization Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)
