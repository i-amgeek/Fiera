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
    tensor_1d gamma {0.7204, 0.0731}, beta {0,0};
    float epsilon;

    tensor_4d temp_in
        {{{{ 0.4083,  0.4021},
          {-0.0062, -0.0193}},

         {{-0.0176, -0.0468},
          {-0.0080, -0.0182}}},


        {{{-0.0064, -0.0038},
          { 0.0441, -0.0048}},

         {{-0.0107,  0.5100},
          { 0.0531,  0.8615}}}};

    tensor_4d expected_output
        {{{{ 1.2540,  1.2287},
          {-0.4414, -0.4951}},

         {{-0.0426, -0.0493},
          {-0.0403, -0.0427}}},


        {{{-0.4426, -0.4318},
          {-0.2358, -0.4360}},

         {{-0.0410,  0.0801},
          {-0.0261,  0.1619}}}};

    epsilon = 1e-5;

    batch_norm_layer_t * layer = new batch_norm_layer_t({2, 2, 2, 2}, true);

    layer->gamma = gamma;
    layer->beta = beta;
    layer->epsilon = epsilon;
    layer->adjust_variance = false;

    tensor_4d out layer->activate(temp_in, true); 
    

    // // if (out == expected_output) 
    // //     cout << "Batch Norm working correctly";

    // cout << "\n\n Expected output";
    // cout << expected_output;
    // cout << "\n Actual output";
    // cout << out;

}
