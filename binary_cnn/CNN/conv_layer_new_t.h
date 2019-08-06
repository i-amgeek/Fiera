#pragma once
#include "layer_t.h"
#include "optimization_method.h"
#include "utils/im2col.hpp"
#include "utils/extras.hpp"

#pragma pack(push, 1)
struct conv_layer_t
{
    layer_type type = layer_type::conv;
    xarray<float> in;
    xarray<float> filters; 
    tdsize in_size, out_size;
    xarray<float> in_col;
    xarray<float> filter_col;
    xarray<gradient_t> filter_grads;
    uint16_t stride;
    uint16_t extend_filter, number_filters;
    bool debug, clip_gradients_flag;

    conv_layer_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size,bool clip_gradients_flag = true, bool debug_flag=false)

    {   
        this->filters = eval(xt::random::rand<float>({(int)number_filters,in_size.z, (int)extend_filter, (int)extend_filter}, -1, 1));
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

    xarray<float> activate(const xarray<float>& in, bool train)
    {
        #ifdef measure_time
        auto start = Clock::now();
        #endif
    
        if (train) this->in = in;
        uint N = in.shape()[0];
        uint C = in.shape()[1];
        uint H = in.shape()[2];
        uint W = in.shape()[3];
        int F = filters.shape()[0];
        uint HH = filters.shape()[2];
        uint WW = filters.shape()[3];
        uint H_prime = (H-HH) / stride + 1;  // Height of `in` after im2col
        uint W_prime = (W-WW) / stride + 1;  // Width of `in` after im2col


        in_col = im2col(in, HH, WW, stride);
        filter_col = filters;
        filter_col.reshape({F, -1});

        xarray<float> mul = linalg::dot(in_col, transpose(filter_col));
        xarray<float> out = col2im(mul, H_prime, W_prime);  

        #ifdef measure_time
        auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Conv_Float Forward Elipsed: "<< elipsed.count() << "s\n";
        #endif

        return out;
    }

    void fix_weights(float learning_rate)
    {
        update_weight(filters, filter_grads,1, false, learning_rate );
        update_gradient( filter_grads );
    }

    xarray<float> calc_grads( xarray<float>& grad_next_layer )
    {
         #ifdef measure_time
        auto start = Clock::now();
        #endif

        int m = grad_next_layer.shape()[0];
        int f = grad_next_layer.shape()[1];
        int H_prime = in_col.shape()[1];
        int W_prime = in_col.shape()[2];
        int C = filters.shape()[1];
        int HH = filters.shape()[2];
        int WW = filters.shape()[3];

        grad_next_layer.reshape({m,f,-1});
        xarray<float> mul_grad = transpose(grad_next_layer, {0,2,1}),
                      in_col_grad =  xt::xarray<float>::from_shape({(uint)m, (uint)H_prime, (uint)W_prime}),
                      tfilter_grads(filters.shape());

        for(int i=0; i<m ;i++){
            xarray<float> in_col_temp =  xt::view(transpose(in_col, {0,2,1}), i, all(), all()),
                          mul_grad_temp = xt::view( mul_grad, i, all(), all());
            xarray<float> dfilter_col = linalg::dot(  in_col_temp,  mul_grad_temp);
            tfilter_grads += xt::reshape_view(transpose(dfilter_col), filters.shape());
            xt::view( in_col_grad, i, all(), all()) = linalg::dot(  mul_grad_temp, filter_col);
        }

        xarray<float> grads_in = col2im_back(in_col_grad, out_size.x, out_size.y, stride, HH, WW, C);
        filter_grads = convert_4d_float_to_gradient(tfilter_grads);
        
        #ifdef measure_time
        auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Conv_Float Backward Elipsed: "<< elipsed.count() << "s\n";
        #endif
        
        return grads_in;
    }   

    inline auto access(xarray<float> in, int i){
        return xt::view(in, i, all(), all());
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
        
