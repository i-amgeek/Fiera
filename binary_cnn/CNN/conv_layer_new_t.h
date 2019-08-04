#pragma once
#include "layer_t.h"
#include "optimization_method.h"
#include "utils/im2col.hpp"
#include "utils/extras.hpp"

#pragma pack(push, 1)
struct conv_layer_t
{
    layer_type type = layer_type::conv;
    tensor_4d in;
    tensor_4d filters; 
    tdsize in_size, out_size;
    tensor_3d in_col;
    xarray<float> filter_col;
    tensorg_4d filter_grads;
    uint16_t stride;
    uint16_t extend_filter, number_filters;
    bool debug, clip_gradients_flag;

    conv_layer_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size,bool clip_gradients_flag = true, bool debug_flag=false)

    {   
        this->filters = eval(xt::random::rand<float>({(int)number_filters, (int)extend_filter, (int)extend_filter, in_size.z}, -1, 1));
        this->number_filters = number_filters;
        this->debug=debug_flag;
        this->stride = stride;
        this->in_size = in_size;
        this->out_size =  {in_size.m, (in_size.x - extend_filter) / stride + 1, (in_size.y - extend_filter) / stride + 1, number_filters};      
        this->extend_filter = extend_filter;
        this->clip_gradients_flag = clip_gradients_flag;
        assert( (float( in_size.x - extend_filter ) / stride + 1)
                ==
                ((in_size.x - extend_filter) / stride + 1) );

        assert( (float( in_size.y - extend_filter ) / stride + 1)
                ==
                ((in_size.y - extend_filter) / stride + 1) );

    }

    tensor_4d activate(tensor_4d& in, bool train)
    {
        #ifdef measure_time
        auto start = Clock::now();
        #endif

        if (train) this->in = in;
        int N = in.shape()[0];
        int C = in.shape()[1];
        int H = in.shape()[2];
        int W = in.shape()[3];
        int F = filters.shape()[0];
        int HH = filters.shape()[2];
        int WW = filters.shape()[3];
        int H_prime = (H-HH) / stride + 1;  // Height of `in` after im2col
        int W_prime = (W-WW) / stride + 1;  // Width of `in` after im2col

        in_col = im2col(in, HH, WW, stride);
        filter_col = filters;
        filter_col.reshape({F, -1});
        filter_col = transpose(filter_col);

        tensor_3d mul = linalg::dot(in_col, filter_col);
        tensor_4d out = col2im(mul, H_prime, W_prime);  

        #ifdef measure_time
        auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Elipsed: "<< elipsed.count() << "s\n";
        #endif

        return out;
    }

    void fix_weights(float learning_rate)
    {
        update_weight(filters, filter_grads,1, false, learning_rate );
        update_gradient( filter_grads );
    }

    tensor_4d calc_grads( tensor_4d& grad_next_layer )
    {
        int m = grad_next_layer.shape()[0];
        int f = grad_next_layer.shape()[1];
        
        xarray<float> temp = grad_next_layer;
        
        temp.reshape({m,f,-1});

        tensor_3d dmul = transpose(temp, {0,2,1});
        
        tensor_3d dfilter_col({m, in_col.shape()[2], dmul.shape()[2] }),
                    din_col({m, in_col.shape()[1], in_col.shape()[2]});

        tensor_4d tfilter_grads(filters.shape());

        for(int i=0; i<m ;i++){
            tensor_2d tarray1 =  xt::view(transpose(in_col, {0,2,1}), i, all(), all()),
                         tarray2 = xt::view(dmul, i, all(), all());
            
            tensor_2d dfilter_col = linalg::dot(tarray1,tarray2);
            
            dfilter_col = transpose(dfilter_col);

            tfilter_grads += xt::reshape_view(dfilter_col, filters.shape());
            tarray1 = xt::view(dmul, i, all(), all());
            tarray2 = transpose(filter_col);

            xt::view(din_col, i, all(), all()) = linalg::dot(tarray1, tarray2);
        }

        tensor_4d grads_in = col2im_back(din_col, out_size.x, out_size.y, stride
                                        , filters.shape()[2], filters.shape()[3], filters.shape()[1]);

        filter_grads = convert_4d_float_to_gradient(tfilter_grads);
        return grads_in;
    }   

    void save_layer( json& model ){
        model["layers"].push_back( {
            { "layer_type", "conv" },
            { "stride", stride },
            { "extend_filter", extend_filter },
            { "number_filters", filters.shape()[0] },
            { "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
            { "clip_gradients", clip_gradients_flag}
        } );
    }

    void save_layer_weight( string fileName ){
        ofstream file(fileName);

        json weights;
        weights["type"] = "conv"; 
		xt::to_json(weights["data"], this->filters);
        file << weights << endl;
        file.close();
    }

    void load_layer_weight(string fileName){
        ifstream file(fileName);
        json weights;
        file >> weights;
        assert(weights["type"] == "conv");
		xt::from_json(weights["data"], this->filters);
        file.close();
    }

    void print_layer(){
        cout << "\n\n Conv Layer : \t";
        cout << "\n\t in_size:\t";
        print_tensor_size(in_size);
        cout << "\n\t Filter Size:\t";
        cout << xt::adapt(filters.shape());
        cout << "\n\t out_size:\t";
        print_tensor_size(out_size);
    }
};
#pragma pack(pop)
        
