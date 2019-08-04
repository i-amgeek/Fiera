#pragma once
#include "layer_t.h"

#pragma pack(push, 1)
struct scale_layer_t
{
    layer_type type = layer_type::scale;
    tensor_2d in;
    gradient_t grads_scale;
    tdsize in_size;
    tdsize out_size;
    bool debug,clip_gradients_flag;
    float s_param;     // scaler learnable parameter

    scale_layer_t(tdsize in_size,bool clip_gradients_flag = true, bool debug_flag = false)        // EXPECTS 1D INPUT
    {
        s_param = 0.001; 
        this->in_size = in_size;
        this->out_size = in_size;
        this->debug = debug_flag;    
        this->clip_gradients_flag = clip_gradients_flag;
    }

    tensor_2d activate(tensor_2d in, bool train)
    {
        if (train) this->in = in;

        tensor_2d out = s_param * in;

        return out;
    }

    void fix_weights(float learning_rate)
    {
        // grads_scale contains sum of gradients of s_param for all examples. 
		s_param = update_weight(s_param,grads_scale,1,false, learning_rate);
		update_gradient(grads_scale);
       
        
    }

    tensor_2d calc_grads(tensor_2d& grad_next_layer)
    {
        assert(in.shape()[0] > 0);
        grads_scale.grad = 0;

        int m = grad_next_layer.shape()[0];

        tensor_2d grads_in({m, out_size.x});

        tensor_2d temp = grad_next_layer * in;
        xarray<float> tt = temp;

        grads_scale.grad = (float)sum(temp)();
        grads_in = grad_next_layer * s_param;
       
        return grads_in;

    }

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "scale" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}

    void save_layer_weight( string fileName ){
        ofstream file(fileName);
        json weights;
        weights["scale_param"] = s_param;
        file << weights;
        file.close();
    }

	void load_layer_weight(string fileName){
        ifstream file(fileName);
        json weights;
        file >> weights;
        this->s_param = weights["scale_param"];
        file.close();
	}

	void print_layer(){
		cout << "\n\n Scale Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
		cout << "\n\t scale parameter:\t" << s_param;
	}


};
#pragma pack(pop)