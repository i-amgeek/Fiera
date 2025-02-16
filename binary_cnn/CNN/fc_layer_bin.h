/*! Fully Connected Binary layer
    It follows: 
      y = sum(~x^y)
 */

//TODO: Adding debug flags to ifdef
//    : bitpacking
//    : Adding Flatten layer


// REMEMBER: alpha and alpha2 is different for different image
//         : inbin and inbin2 are referred as 1 and -1 for explanation purpose

#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"
#include "tensor_bin_t.h"


using namespace std;

#pragma pack(push, 1)
struct fc_layer_bin_t
{
	layer_type type = layer_type::fc_bin;
	tensor_t<float> in;
    vector<float> alpha;	// alpha is calculated for each image. 
	vector<float> alpha2;
	bool debug,clip_gradients_flag;
	tensor_t<float> weights;
	tensor_t<gradient_t> weights_grad;
	tensor_t<float> al_b;
    tensor_bin_t weights_bin;
	tdsize in_size,out_size;
	tensor_t<gradient_t> gradients;

	fc_layer_bin_t( tdsize in_size, tdsize out_size, bool clip_gradients_flag = true, bool debug_flag = false )
	/**
	 * Parameters
	 * ----------
	 * in_size : (int m, int x, int y, int z)
	 * 		Size of input matrix.
	 * 
	 * out_size : int
	 * 		No of fully connected nodes in output.
	 * 
	 * clip_gradients_flag : bool
	 * 		Whether gradients have to be clipped or not
	 * 
	 * debug_flag : bool
	 * 		Whether to print variables for debugging purpose
	 * 
	 **/
		:
        weights_bin(in_size.x * in_size.y * in_size.z, out_size.x, 1, 1 ),
		weights(in_size.x * in_size.y * in_size.z, out_size.x, 1, 1 ),
		// to be checked later :(
		gradients(in_size.x * in_size.y * in_size.z, out_size.x, 1, 1)
	{
		this->in_size = in_size;
		this->out_size = out_size;
		this->debug = debug_flag;
		this->clip_gradients_flag = clip_gradients_flag;

		// WEIGHT INITIALIZATION
		for ( int i = 0; i < out_size.x; i++ )
			for ( int h = 0; h < in_size.x*in_size.y*in_size.z; h++ )
			{
				weights(h,i, 0, 0 ) =  (1.0f * (rand()-rand())) / float( RAND_MAX );  // Generates a random number between -1 and 1 
				weights_bin.data[weights_bin(h, i, 0, 0)] = 0;
			}

	}

    tensor_bin_t binarize(tensor_t<float> in)
	//returns first binarization of input
	{
		tensor_bin_t in_bin(in.size.m, in_size.x, in_size.y, in_size.z );
        // BINARIZING `weights`
        for (int i = 0; i < weights.size.m; ++i)
            for (int j = 0; j < weights.size.x; j++)
                weights_bin.data[weights_bin(i, j, 0, 0)] = weights(i, j, 0, 0) >= 0 ? 1 : 0;
        
		// BINARIZING `in`
		for ( int m = 0; m < in.size.m; m++ )
			for ( int i = 0; i < in.size.x; i++ )
				for ( int j = 0; j < in.size.y; j++ )
					for ( int z = 0; z < in.size.z; z++ )
						in_bin.data[in_bin(m, i, j, z)] = in(m, i, j, z) >= 0 ? 1 : 0;

		return in_bin;
					
    }

	tensor_bin_t calculate_alpha( tensor_t<float> in, tensor_bin_t in_bin){
		// calculates alpha, alpha2, in_bin2

		// REMEMBER: alpha and alpha2 is different for different image
		//         : inbin and inbin2 are referred as 1 and -1 for explanation purpose

		alpha.resize(in.size.m);
		alpha2.resize(in.size.m);

		// CALCULATE alpha1
		// alpha1 = sum(abs(in))/size

		for(int e = 0; e < in.size.m; e++)
		{
			float sum = 0;
			for(int x = 0; x < in.size.x; x++)
				for(int y = 0; y < in.size.y; y++)
					for(int z = 0; z < in.size.z; z++)
						sum += abs(in(e,x,y,z));
			
			alpha[e] = sum/(in.size.x*in.size.y*in.size.z);

		}

		// CALCULATE alpha2
		// inbin2 = (in - alpha*in_bin)
		// alpha2 = sum(abs(in - alpha*in_bin))/size
		
		tensor_bin_t in_bin2(in.size.m, in_size.x, in_size.y, in_size.z);
		tensor_t<float> temp(in.size.m, in.size.x, in.size.y, in.size.z);

		for(int e = 0; e<in.size.m; e++)
		{
			float sum = 0;
			for(int x=0; x<in.size.x; x++)
				for(int y=0; y<in.size.y; y++)
					for(int z=0; z<in.size.z; z++){

						temp(e,x,y,z) = in(e,x,y,z) - alpha[e]*(in_bin(e,x,y,z)==1 ? float(1) : float(-1) );
						in_bin2.data[in_bin2(e,x,y,z)] = temp(e,x,y,z)>=0? 1 : 0;
						sum += abs(temp(e,x,y,z));
				}

			alpha2[e] = sum/(in.size.x*in.size.y*in.size.z);

		}

		return in_bin2;
	}
	tensor_t<float> calculate_al_b(tensor_bin_t in_bin, tensor_bin_t in_bin2){
		// CALCULATE al_b 
		// al_b = alpha * in_bin + alpha2 * in_bin2

		tensor_t<float> al_b(in_bin.size.m, in_size.x, in_size.y, in_size.z);
		for (int e = 0; e < in_bin.size.m; e++)
			for(int x=0; x<in_size.x; x++)
				for(int y=0; y<in_size.y; y++)
					for(int z=0; z<in_size.z; z++){
						al_b(e, x, y, z) = alpha[e] * (in_bin.data[in_bin(e, x, y, z)] == 1 ? float(1) : float(-1) ) +
								alpha2[e] * (in_bin2.data[in_bin2(e, x, y, z)] == 1 ? float(1) : float(-1) );
					}
	
		return al_b;
	
	}

	int map( point_t d )
	// Maps weight unit to corresponding input unit.
	{
		return d.m * (in_size.x * in_size.y * in_size.z) +
			d.z * (in_size.x * in_size.y) +
			d.y * (in_size.x) +
			d.x;
	}

	 tensor_t<float> activate( tensor_t<float> in, bool train )
 
	 //  `activate` FORWARD PROPOGATES AND SAVES THE RESULT IN `out` VARIABLE.

	{

		tensor_t<float> out( in.size.m, weights.size.x, 1, 1 );
        tensor_bin_t in_bin = binarize(in); // first binarization 
        tensor_bin_t in_bin2 = calculate_alpha(in, in_bin); // second binarization
		
		// al_b = alpha * in_bin + alpha2 * in_bin2
		tensor_t<float> al_b = 	calculate_al_b(in_bin, in_bin2); 

		if (train) this->al_b = al_b;
		if (train) this->in = in;

		for( int e = 0; e < in.size.m; e++)
			for(int n = 0; n < weights.size.x; n++ ){
				float sum=0, sum2=0;
				for ( int i = 0; i < in.size.x; i++ )
					for ( int j = 0; j < in.size.y; j++ )
						for ( int z = 0; z < in.size.z; z++ )
						{
							int m = map( { 0 , i, j, z } );
							bool f = weights_bin.data[weights_bin( m, n, 0, 0 )];
							bool v = in_bin.data[in_bin(e, i, j, z)];
							bool v2 = in_bin2.data[in_bin2(e, i, j, z)];
							sum += !(f ^ v);
							sum2 += !(f ^ v2);
						}

				// weights.size.m is equals to total size of input. i.e. in.size.x * in.size.y * in.size.z
				
				// alpha * ( 2P - N )
				out(e, n, 0, 0 ) = alpha[e] * ( 2 * sum - weights.size.m );	
				// alpha2 * ( 2P - N )		
				out(e, n, 0, 0 ) += alpha2[e] * (2 * sum2 - weights.size.m ); 
			}

		return out;
	}

	void fix_weights(float learning_rate)
	{
		for ( int n = 0; n < weights.size.x; n++ )
			for ( int i = 0; i < in_size.x; i++ )
				for ( int j = 0; j < in_size.y; j++ )
					for ( int z = 0; z < in_size.z; z++ )
					{
						int m = map( { 0, i, j, z } );
						float& w = weights( m, n, 0, 0 );

						gradient_t& grad = weights_grad( m, n, 0, 0 );
						w = update_weight( w, grad, 1, true, learning_rate); 
						update_gradient( grad );
					}

	}

	tensor_t<float> calc_grads( tensor_t<float>& grad_next_layer )
	
	// CALCULATES BACKWARD PROPOGATION AND SAVES RESULT IN `grads_in`. 
	
	{
		assert (in.size > 0);
		tensor_t<float> grads_in( grad_next_layer.size.m, in_size.x, in_size.y, in_size.z );
		weights_grad.resize(weights.size);


		for(int e=0; e<in.size.m; e++)
			for ( int n = 0; n < weights.size.x; n++ )
			{

				for ( int i = 0; i < in_size.x; i++ )
					for ( int j = 0; j < in_size.y; j++ )
						for ( int z = 0; z < in_size.z; z++ )
						{
							int m = map( {0, i, j, z } );
							if(fabs(in(e,i,j,z)) <= 1)
								grads_in(e, i, j, z ) += grad_next_layer(e, n, 0, 0) * (weights_bin.data[weights_bin( m, n,0, 0 )] == 1? float(1) : float(-1));
							else
								grads_in(e,i,j,z) += 0;
							
							weights_grad(m, n, 0, 0).grad += grad_next_layer(e, n, 0, 0) * al_b(e, i, j, z);	// d W = d A(l+1) * A(l)(binary) 

							clip_gradients(clip_gradients_flag, grads_in(e,i,j,z));
						}
			}

		return grads_in;
	}

	void save_layer( json& model ){
		model["layers"].push_back( {
			{ "layer_type", "fc_bin" },
			{ "in_size", {in_size.m, in_size.x, in_size.y, in_size.z} },
			{ "out_size", {out_size.m, out_size.x, out_size.y, out_size.z} },
			{ "clip_gradients", clip_gradients_flag}
		} );
	}

	void save_layer_weight( string fileName ){
		vector<float> data;
		int m = weights.size.m;
		int x = weights.size.x;
		int y = weights.size.y;
		int z = weights.size.z;
		int array_size = m * x * y * z;
		for ( int i = 0; i < array_size; i++ )
			data.push_back(weights.data[i]);

		ofstream file(fileName);
		json weight = { 
			{ "type", "fc_bin" },
			{ "size", array_size },
			{ "data", data}
		};
		file << weight << endl;
		file.close();
	}

	void load_layer_weight(string fileName){
		ifstream file(fileName);
		json weight;
		file >> weight;
		assert(weight["type"] == "fc_bin");
		vector<float> data = weight["data"];
		int size  = weight["size"];
		for (int i = 0; i < size; i++)
			this->weights.data[i] = data[i];
		file.close();
	}

	void print_layer(){
		cout << "\n\n FC Binary Layer : \t";
		cout << "\n\t in_size:\t";
		print_tensor_size(in_size);
		cout << "\n\t out_size:\t";
		print_tensor_size(out_size);
	}
};
#pragma pack(pop)