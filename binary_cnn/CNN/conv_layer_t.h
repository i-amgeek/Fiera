#pragma once
#include "layer_t.h"

// #pragma pack(push, 1)
struct conv_layer_t
{
	layer_type type = layer_type::conv;
	tensor_t<float> grads_in;
	tensor_t<float> in;
	tensor_t<float> out;
	tensor_t<float> filters; 
	tensor_t<gradient_t> filter_grads;
	uint16_t stride;
	uint16_t extend_filter;

	conv_layer_t( uint16_t stride, uint16_t extend_filter, uint16_t number_filters, tdsize in_size)
		:
		grads_in( in_size.m, in_size.x, in_size.y, in_size.z ),
		in(in_size.m, in_size.x, in_size.y, in_size.z ),
		out(in_size.m,
		(in_size.x - extend_filter) / stride + 1,
			(in_size.y - extend_filter) / stride + 1,
			number_filters
		),
		filters(number_filters, extend_filter, extend_filter, in_size.z),
		filter_grads(number_filters, extend_filter, extend_filter, in_size.z)

	{
		this->stride = stride;
		this->extend_filter = extend_filter;
		assert( (float( in_size.x - extend_filter ) / stride + 1)
				==
				((in_size.x - extend_filter) / stride + 1) );

		assert( (float( in_size.y - extend_filter ) / stride + 1)
				==
				((in_size.y - extend_filter) / stride + 1) );

		for ( int a = 0; a < number_filters; a++ )
		{

			int maxval = extend_filter * extend_filter * in_size.z;
			
			
			for ( int i = 0; i < extend_filter; i++){
				for ( int j = 0; j < extend_filter; j++){
					for ( int z = 0; z < in_size.z; z++ ){
						/**************temporary*************/
						//  filters(a,i,j,z) = pow(-1,i^j)*2+i+j-3;
						filters(a,i, j, z ) =  (1.0f * (rand()-rand())) / float( RAND_MAX );
					}
				}
			}
		}

		cout<<"**************weights for convolution*******\n";
		print_tensor(filters);
		
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

	range_t map_to_output( int x, int y )
	{
		float a = x;
		float b = y;
		return
		{
			normalize_range( (a - extend_filter + 1) / stride, out.size.x, true ),
			normalize_range( (b - extend_filter + 1) / stride, out.size.y, true ),
			0,
			normalize_range( a / stride, out.size.x, false ),
			normalize_range( b / stride, out.size.y, false ),
			(int)filters.size.m - 1,
		};
	}

	void activate( tensor_t<float>& in )
	{
		this->in = in;
		activate();
	}

	void activate()
	{
		for(int example = 0; example<in.size.m; example++)

			for ( int filter = 0; filter < filters.size.m; filter++ )
			{
				for ( int x = 0; x < out.size.x; x++ )
				{
					for ( int y = 0; y < out.size.y; y++ )
					{
						point_t mapped = map_to_input( { (uint16_t)x, (uint16_t)y, 0 }, 0 );
						float sum = 0;
						for ( int i = 0; i < extend_filter; i++ )
							for ( int j = 0; j < extend_filter; j++ )
								for ( int z = 0; z < in.size.z; z++ )
								{
									float f = filters( filter, i, j, z );
									float v = in(example, mapped.x + i, mapped.y + j, z );
									sum += f*v;
								}
						out(example, x, y, filter ) = sum;
					}
				}
			}
		cout<<"*********out for convolution*************\n";
		print_tensor(out);
	}

	void fix_weights(float learning_rate)
	{
		for ( int a = 0; a < filters.size.m; a++ )
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
					{
						float& w = filters(a, i, j, z );
						gradient_t& grad = filter_grads(a, i, j, z );
						grad.grad /= in.size.m;
						w = update_weight( w, grad,1,false,learning_rate);
						update_gradient( grad );
					}
		cout<<"*******new weights for float conv*****\n";
		print_tensor(filters);
	}

	void calc_grads( tensor_t<float>& grad_next_layer )
	{
		cout<<"*********grads_in for float conv********\n";
		print_tensor(grads_in);

		for ( int k = 0; k < filter_grads.size.m; k++ )
		{
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads(k, i, j, z ).grad = 0;
		}

		for ( int k = 0; k < filter_grads.size.m; k++ )
		{
			for ( int i = 0; i < extend_filter; i++ )
				for ( int j = 0; j < extend_filter; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						filter_grads(k, i, j, z ).grad = 0;
		}

		for(int e=0; e<in.size.m; e++){
			for ( int x = 0; x < in.size.x; x++ )
			{
				for ( int y = 0; y < in.size.y; y++ )
				{
					range_t rn = map_to_output( x, y );
					for ( int z = 0; z < in.size.z; z++ )
					{
						float sum_error = 0;
						for ( int i = rn.min_x; i <= rn.max_x; i++ )
						{
							int minx = i * stride;
							for ( int j = rn.min_y; j <= rn.max_y; j++ )
							{
								int miny = j * stride;
								for ( int k = rn.min_z; k <= rn.max_z; k++ )
								{
									float w_applied = filters(k, x - minx, y - miny, z );
									sum_error += w_applied * grad_next_layer(e, i, j, k );
									//  cout<<w_applied<<' '<<grad_next_layer(e,i,j,k)<<' '<<sum_error<<endl;
									filter_grads(k, x - minx, y - miny, z ).grad += in(e, x, y, z ) * grad_next_layer(e, i, j, k );
								}
							}
						}
						grads_in(e, x, y, z ) = sum_error;
					}
				}
			}
		}
		
		cout<<"*********grads_in for float conv********\n";
		print_tensor(grads_in);
	}
};
// #pragma pack(pop)
