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

        out = s_param * in;

        // if(debug)
        // {
        //     cout<<"*****output for scale***********\n";
        //     print_tensor(out);
        // }

        return out;
    }

    void fix_weights(float learning_rate)
    {
        // grads_scale contains sum of gradients of s_param for all examples. 
		s_param = update_weight(s_param,grads_scale,1,false, learning_rate);
		update_gradient(grads_scale);
       
        // if(debug)
        // {
        //     cout<<"*******updated s_param*****\n";
		//     cout<<s_param<<endl;
        // }
    }

    tensor_t<float> calc_grads(tensor_t<float>& grad_next_layer)
    {
        assert(in.size > 0);
        grads_scale.grad = 0;

        int m = grad_next_layer.size.m;

        tensor_t<float> grads_in(m, out_size.x, 1, 1);

        grads_scale.grad = sum( grad_next_layer * in );
        grads_in = grad_next_layer * s_param;
        // if(debug)
        // {
        //     cout<<"***********grads_in for scale********\n";
        //     print_tensor(grads_in);
        //     cout<<"***********gradient for s_param*******\n";
        //     cout<<grads_scale.grad<<endl;
        // }

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