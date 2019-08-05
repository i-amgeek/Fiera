#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/batch_norm_layer_new_t.h"
#include "tensor.hpp"

using namespace std;

int main()
{

    xarray<float> gamma {0.7204, 0.0731}, beta {0,0};
    float epsilon;
    
    xarray<float> temp_in
        {{{{ 0.4083,  0.4021},
          {-0.0062, -0.0193}},

         {{-0.0176, -0.0468},
          {-0.0080, -0.0182}}},


        {{{-0.0064, -0.0038},
          { 0.0441, -0.0048}},

         {{-0.0107,  0.5100},
          { 0.0531,  0.8615}}}};


    xarray<float> grad_next_layer
        {{{{-0.0521,  0.0311},
          {-0.0311,  0.0595}},

         {{-0.0969, -0.0763},
          { 0.0155, -0.0732}}},


        {{{-0.0876,  0.1876},
          {-0.0468, -0.0322}},

         {{-0.0515, -0.0657},
          { 0.0460, -0.0768}}}};
      
    xarray<float> expected_output
        {{{{-0.1536,  0.1855},
          {-0.1679,  0.1996}},

         {{-0.0130, -0.0085},
          { 0.0132, -0.0075}}},


        {{{-0.3991,  0.7274},
          {-0.2199, -0.1720}},

         {{-0.0024, -0.0015},
          { 0.0208, -0.0012}}}};
    
    epsilon = 1e-5;

    batch_norm_layer_t * layer = new batch_norm_layer_t({2, 2, 2, 2});
    layer->gamma = gamma;
    layer->beta = beta;
    layer->epsilon = epsilon;
    layer->adjust_variance = false;
    layer->activate(temp_in, true); 
              
    xarray<float> grads_in = layer->calc_grads(grad_next_layer);

    if (xt::allclose(grads_in, expected_output, 1e-2*5))
      cout << "Batch Norm Backward working correctly\n";
    else{
      cout << xt::isclose(grads_in, expected_output, 1e-2*5);
      cout << "Expected output is\n";
      cout << expected_output;
      cout << "\nActual output is\n";
      cout << grads_in;
    }
}
