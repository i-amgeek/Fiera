#pragma once
#include "layer_t.h"
#include "utils/im2col.hpp"
#include "utils/extras.hpp"
#include <climits>

// typedef unsigned int uint128_t __attribute__((mode(TI)));

#pragma pack(push, 1)
struct conv_layer_bin_t
{
	layer_type type = layer_type::conv_bin;
	xarray<float> in;
	xarray<float> filters, al_b; 
	tensor_bin_t filters_bin; 
	xarray<gradient_t> filter_grads;
	tensor_uint64_4d packed_input, packed_weight;
	uint16_t stride;
	uint16_t extend_filter, number_filters;
	tdsize in_size, out_size;
	bool debug,clip_gradients_flag; 	

	conv_layer_bin_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size, bool clip_gradients_flag = true, bool debug_flag = false)
		:
		// filters(number_filters, extend_filter, extend_filter, in_size.z),
		// filter_grads(number_filters, extend_filter, extend_filter, in_size.z),
		filters_bin(number_filters, extend_filter, extend_filter, in_size.z)

	{
        this->filters = eval(xt::random::rand<float>({(int)number_filters, in_size.z, (int)extend_filter, (int)extend_filter}, -1, 1));
		this->number_filters = number_filters;
		this->out_size =  {in_size.m, (in_size.x - extend_filter) / stride + 1, (in_size.y - extend_filter) / stride + 1, number_filters};
		this->in_size = in_size;
		this->debug = debug_flag;
		this->stride = stride;
		this->extend_filter = extend_filter;
		this->clip_gradients_flag = clip_gradients_flag;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

	}

	point_t map_to_input( point_t out, int z )
	{
		out.x *= stride;
		out.y *= stride;
		out.z = z;
		return out;
	}

	struct range_t
	{
		int min_x, min_y, min_z;
		int max_x, max_y, max_z;
	};

	int normalize_range( float f, int max, bool lim_min )
	{
		if ( f <= 0 )
			return 0;
		max -= 1;
		if ( f >= max )
			return max;

		if ( lim_min ) // left side of inequality
			return ceil( f );
		else
			return floor( f );
	}

	void bitpack_64(xarray<float> in){

			packed_input.resize({in.shape()[0], in.shape()[1]/64, in.shape()[2], in.shape()[3] });
			packed_weight.resize({filters.shape()[0],  filters.shape()[1]/64, filters.shape()[2], filters.shape()[3]});

			for(int i=0; i<in.shape()[0]; i++){
				for(int j=0; j<in.shape()[2]; j++){
					for(int k=0; k<in.shape()[3]; k++){
						for(int z=0; z<in.shape()[1]; z+=64){
							
							const size_t UNIT_LEN = 64;
							std::bitset<UNIT_LEN> bits;

							for(int zz = z; zz<z+64; zz++)
								bits[zz-z] = in(i,zz,j,k) >= 0;
							
								// cout<<endl;
							static_assert(sizeof(decltype(bits.to_ullong())) * CHAR_BIT == 64
								,"bits.to_ullong() must return a 64-bit element");
							packed_input(i,z/64,j,k) = bits.to_ullong();
						}
					}
				}
			}
			
			for(int i=0; i<filters.shape()[0]; i++){
				for(int j=0; j<filters.shape()[2]; j++){
					for(int k=0; k<filters.shape()[3]; k++){
						for(int z=0; z<filters.shape()[1]; z+=64){
							
							const size_t UNIT_LEN = 64;
							std::bitset<UNIT_LEN> bits;

							for(int zz = z; zz<z+64; zz++)
								bits[zz-z] = filters(i,zz,j,k) >= 0;
								// cout<<bits[zz-z]<<' '/

								static_assert(sizeof(decltype(bits.to_ullong())) * CHAR_BIT == 64
										,"bits.to_ullong() must return a 64-bit element");
								packed_weight(i,z/64,j,k) = bits.to_ullong();
						}
					}
				}
			}

	}


	xarray<float> activate(xarray<float>& in, bool train){
		
		#ifdef measure_time
		auto start = std::chrono::high_resolution_clock::now();
		#endif

		if (train) this->in = in;

		auto out =  xt::xarray<float>::from_shape({in.shape()[0], number_filters,  (in_size.y - extend_filter) / stride + 1, (in_size.x - extend_filter) / stride + 1 });
		
		assert(in.shape()[1] % 64 == 0);

		bitpack_64(in);

		for(int example = 0; example<packed_input.shape()[0]; example++){
			for ( int filter = 0; filter < packed_weight.shape()[0]; filter++ )
				for ( int x = 0; x < out.shape()[2]; x++ )
					for ( int y = 0; y < out.shape()[3]; y++ ){

						point_t mapped = map_to_input( { 0, (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float sum = 0, sum2 = 0;
						for ( int z = 0; z < packed_input.shape()[1]; z++ )
							for ( int i = 0; i < extend_filter; i++ )
								for ( int j = 0; j < extend_filter; j++ )
								{
									uint64_t xnor = ~(packed_input(example,z, mapped.x + i, mapped.y + j)
														^ packed_weight(filter, z, i, j));
									sum += __builtin_popcount(xnor);
								}
						out(example, filter, x, y ) = (2*sum - extend_filter*extend_filter*in.shape()[1]);
						
					}
		}
		
        #ifdef measure_time
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Conv_Binary Forward Elipsed: " << elapsed.count() << " s\n";
		#endif

		return out;

	}
	
	void fix_weights(float learning_rate)
    {
        update_weight(filters, filter_grads,1, true, learning_rate );
        update_gradient( filter_grads );
    }

	xarray<float> calc_grads( xarray<float> grad_next_layer)
	{
		#ifdef measure_time
		auto start = std::chrono::high_resolution_clock::now();
		#endif
		
		// from conv_layer_new forward
		int N = al_b.shape()[0];
        int C = al_b.shape()[1];
        int H = al_b.shape()[2];
        int W = al_b.shape()[3];
        int F = filters.shape()[0];
        int HH = filters.shape()[2];
        int WW = filters.shape()[3];
        int H_prime = (H-HH) / stride + 1;  // Height of `in` after im2col
        int W_prime = (W-WW) / stride + 1;  // Width of `in` after im2col

		this->al_b = xt::sign(in);
		
		xarray<float> in_col = im2col(al_b, HH, WW, stride);

		xarray<float> sign_filters = xt::sign(filters);
        
		xarray<float> sign_filters_col = sign_filters;
        sign_filters_col.reshape({F, -1});
        sign_filters_col = transpose(sign_filters_col);

		//from con_layer_new backward
		int m = grad_next_layer.shape()[0];
        int f = grad_next_layer.shape()[1];
        
        xarray<float> temp = grad_next_layer;
        
        temp.reshape({m,f,-1});

        xarray<float> dmul = transpose(temp, {0,2,1});
		
		auto dfilter_col = xt::xarray<float>::from_shape({m, in_col.shape()[2], dmul.shape()[2] }),
                    din_col = xt::xarray<float>::from_shape({m, in_col.shape()[1], in_col.shape()[2]});

        xarray<float> tfilter_grads(filters.shape());


        for(int i=0; i<m ;i++){
            xarray<float> tarray1 =  xt::view(transpose(in_col, {0,2,1}), i, all(), all()),
                         tarray2 = xt::view(dmul, i, all(), all());

            xarray<float> dfilter_col = linalg::dot(tarray1,tarray2);
            
            dfilter_col = transpose(dfilter_col);

            tfilter_grads += xt::reshape_view(dfilter_col, filters.shape());
            tarray1 = xt::view(dmul, i, all(), all());
            tarray2 = transpose(sign_filters_col);

            xt::view(din_col, i, all(), all()) = linalg::dot(tarray1, tarray2);
        }

		xarray<float> grads_in = col2im_back(din_col, out_size.x, out_size.y, stride, filters.shape()[2], filters.shape()[3], filters.shape()[1]);


		grads_in = xt::where(in>-1 && in<1, grads_in, 0);
        filter_grads = convert_4d_float_to_gradient(tfilter_grads);

		
		
		#ifdef measure_time		
		auto finish = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = finish - start;
		std::cout << "Conv_Binary Backward Elipsed: " << elapsed.count() << " s\n";
		#endif

		return grads_in;
	}

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "conv_bin" },
			{ "stride", stride },
			{ "extend_filter", extend_filter },
			{ "number_filters", filters.shape()[0] },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}

	void save_layer_weight( string fileName ){
		ofstream file(fileName);
		json weight;
		weight["type"] = "conv_bin"; 
		xt::to_json(weight["data"], this->filters);
		file << weight << endl;
		file.close();
	}

	void load_layer_weight( string fileName ){
		ifstream file(fileName);
		json weight;
		file >> weight;
		assert(weight["type"] == "conv_bin");
		xt::from_json(weight["data"], this->filters);
		file.close();
	}


	void print_layer(){
		cout << "\n\n Conv Binary Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t Filter Size:\t";
		cout << xt::adapt(filters.shape());
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)
