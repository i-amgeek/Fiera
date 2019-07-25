#pragma once
#include "layer_t.h"
#include "utils/im2col.hpp"

#pragma pack(push, 1)
struct conv_layer_t
{
    layer_type type = layer_type::conv;
    tensor_4d in;
    tensor_4d filters; 
    tdsize in_size, out_size;
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


        // if(debug)
        // {
        //  cout<<"**************weights for convolution*******\n";
        //  print_tensor(filters);
        // }
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
        int H_prime = (H-HH) / stride + 1;
        int W_prime = (W-WW) / stride + 1;

        tensor_3d in_col = im2col(in, HH, WW, stride);
        xarray<float> filters_col = filters;
        filters_col.reshape({F, -1});

        tensor_3d mul = linalg::dot(in_col, transpose(filters_col));
        tensor_4d out = col2im(mul, H_prime, W_prime);

        #ifdef measure_time
        auto finish = Clock::now();
		std::chrono::duration<double> elipsed = finish - start;
		cout << "Elipsed: "<< elipsed.count() << "s\n";
        #endif

        return out;
    }

    // void fix_weights(float learning_rate)
    // {
    //     update_weight(filters, filter_grads );
    //     update_gradients( filter_grads );
    // }

    // tensor_4d calc_grads( tensor_4d& grad_next_layer )
    // {
    //     int m = grads_next_layer.shape()[0];
    //     int f = grads_next_layer.shape()[1];

    //     tensor_3d grad_mul = transpose(grad_next_layer.reshape({m,f,-1}));

    //     tensor_2d grad_filters_col = linalg::dot(transpose(in_col), grad_mul);

    //     tensor_3d grad_in_col = linalg::dot(grad_mul, filter_col);

    //     tensor_4d grads_in = col2im_back(grad_in_col, grad_next_layer.shape(), in.shape());

    //     tensor_4d filter_grads = grad_filters_col.reshape(filters.shape());
        
    //     return grads_in;
    // }

    // void save_layer( json& model ){
    //     model["layers"].push_back( {
    //         { "layer_type", "conv" },
    //         { "stride", stride },
    //         { "extend_filter", extend_filter },
    //         { "number_filters", filters.size.m },
    //         { "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
    //         { "clip_gradients", clip_gradients_flag}
    //     } );
    // }

    // void save_layer_weight( string fileName ){
    //     ofstream file(fileName);
    //     int m = filters.size.m;
    //     int x = filters.size.x;
    //     int y = filters.size.y;
    //     int z = filters.size.z;
    //     int array_size = m*x*y*z;
        
    //     vector<float> data;
    //     for ( int i = 0; i < array_size; i++ )
    //         data.push_back(filters.data[i]);    
    //     json weights = { 
    //         { "type", "conv" },
    //         { "size", array_size},
    //         { "data", data}
    //     };
    //     file << weights << endl;
    //     file.close();
    // }

    // void load_layer_weight(string fileName){
    //     ifstream file(fileName);
    //     json weights;
    //     file >> weights;
    //     assert(weights["type"] == "conv");
    //     vector<float> data = weights["data"];
    //     int size  = weights["size"];
    //     for (int i = 0; i < size; i++)
    //         this->filters.data[i] = data[i];
    //     file.close();
    // }

    // void print_layer(){
    //     cout << "\n\n Conv Layer : \t";
    //     cout << "\n\t in_size:\t";
    //     print_tensor_size(in_size);
    //     cout << "\n\t Filter Size:\t";
    //     print_tensor_size(filters.size);
    //     cout << "\n\t out_size:\t";
    //     print_tensor_size(out_size);
    // }
};
#pragma pack(pop)
        
