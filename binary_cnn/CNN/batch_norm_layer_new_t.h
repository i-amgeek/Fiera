#pragma once
#include<math.h>
#include"optimization_method.h"
#include "layer_t.h"

#pragma pack(push, 1)
struct batch_norm_layer_t
{
	layer_type type = layer_type::batch_norm;
	xarray<float> mean_diff;
	xarray<float> in_hat;
	tdsize in_size, out_size;
	float epsilon, momentum;
	xarray<float> gamma, beta, sqrt_sigma;
	xarray<gradient_t> grads_beta, grads_gamma;
    bool adjust_variance;
	bool debug, clip_gradients_flag;
	xarray<float> running_mean, running_var;

	batch_norm_layer_t(tdsize in_size,bool clip_gradients_flag = true, bool debug_flag = false)
	{
		this->in_size = in_size;
		this->out_size = in_size;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;
		this->epsilon = 1e-5;
		this->momentum = 0.9;
        this->adjust_variance = false;
		this->gamma = xt::ones<float>({(uint)in_size.z});
		this->beta = xt::zeros<float>({(uint)in_size.z});
		this->running_mean = xarray<float>::from_shape({(uint)in_size.z});
		this->running_var = xarray<float>::from_shape({(uint)in_size.z});
	}

	auto make1Dto4D(xarray<float> in){
		return xt::view(in, newaxis(), all(), newaxis(), newaxis());
	}

	xarray<float> activate(xarray<float>& in, bool train = true){
		
		#ifdef measure_time
        auto start = Clock::now();
        #endif
		
		xarray<float> out;
		
		if(train) {

			xarray<float> u_mean = xt::mean(in, {0,2,3});
			mean_diff = in - make1Dto4D(u_mean);
			xarray<float> sigma = xt::mean(mean_diff * mean_diff, {0, 2, 3});

			if(in_size.m > 1 and adjust_variance)
				sigma *= in_size.m / (in_size.m - 1);

			sqrt_sigma = xt::sqrt(sigma + epsilon);
			in_hat = mean_diff / make1Dto4D(sqrt_sigma);
			out = make1Dto4D(gamma) * in_hat + make1Dto4D(beta);

			running_mean = momentum * running_mean + (1.0 - momentum) * u_mean;
			running_var = momentum * running_var + (1.0 - momentum) * sigma;
		}
		else{
			xarray<float> in_hat = (in - make1Dto4D(running_mean)) / make1Dto4D(xt::sqrt(running_var + epsilon));
			out = gamma * in_hat + beta;
		}
		
		 #ifdef measure_time
        auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Batch_Norm Forward Elipsed: "<< elipsed.count() << "s\n";
        #endif

		return out;
	
	}
	
	
	void fix_weights(float learning_rate){
		
		update_weight(beta, grads_beta, 1, false, learning_rate);
		update_gradient(grads_beta);
		update_weight(gamma,grads_gamma,1,false, learning_rate);
		update_gradient(grads_gamma);
	}

	xarray<float> calc_grads( xarray<float>& grad_next_layer)
	{
		#ifdef measure_time
        auto start = Clock::now();
        #endif

		assert(in_hat.size() > 0); 
		this->grads_beta = xt::sum(grad_next_layer, {0,2,3});
		this->grads_gamma = xt::sum(in_hat * grad_next_layer, {0,2,3});
		xarray<float> grads_in_hat = grad_next_layer * make1Dto4D(gamma);

		xarray<float> grads_sqrtvar = -1.0 * xt::sum(grads_in_hat * mean_diff, {0,2,3}) / (sqrt_sigma * sqrt_sigma);

		xarray<float> grads_var = 0.5 * grads_sqrtvar / sqrt_sigma;

		xarray<float> grads_xmul1 = grad_next_layer * make1Dto4D(gamma) / make1Dto4D(sqrt_sigma);
		xarray<float> grads_xmul2 = make1Dto4D(grads_var) * 2 * mean_diff / (out_size.m * out_size.x * out_size.y);
		xarray<float> grads_x1 = grads_xmul1 + grads_xmul2;
		xarray<float> grads_u_mean = xt::sum(-grads_x1, {0,2,3});
		xarray<float> grads_in = grads_x1 + make1Dto4D(grads_u_mean) / (out_size.m * out_size.x * out_size.y);

		 #ifdef measure_time
        auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Batch_Norm Backward Elipsed: "<< elipsed.count() << "s\n";
        #endif

		return grads_in;	
	}

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
		};
		xt::to_json(weights["beta"], this->beta);
		xt::to_json(weights["gamma"], this->gamma);
		file << weights;
	}

	void load_layer_weight(string fileName){
		ifstream file(fileName);
		json weights;
		file >> weights;
		this->epsilon = weights["epsilon"];
		xt::from_json(weights["beta"], this->beta);
		xt::from_json(weights["gamma"], this->gamma);
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
