#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/fc_layer_new.h"
#include "tensor.hpp"

using namespace std;

int main()
{

    tensor_2d temp_in
        {{1.2540, 1.2287, -0.4414, -0.4951, -0.0426, -0.0493, -0.0403, -0.0427},
         {-0.4426, -0.4318, -0.2358, -0.4360, -0.0410,  0.0801, -0.0261,  0.1619}};

    temp_in.reshape({2,-1});

    // cout<<temp_in;

    // Can be reshaped like this
    // xtensor<float, 2> b = xt::reshape_view(a, {20,2});

// std::copy(temp_in.shape().cbegin(), temp_in.shape().cend(), std::ostream_iterator<size_t>(std::cout, " ")); // To print shape of xtensor
    tensor_2d weights 
       {{0.3323,  0.2685,  0.1723, -0.1887},
        {-0.2773,  0.0909,  0.3247,  0.3051},
        { 0.2707,  0.1876, -0.0787,  0.3235},
        {-0.0614, -0.2735, -0.1970,  0.0407},
        { 0.1819,  0.2517, -0.0890, -0.0612},
        { 0.1378,  0.1217, -0.2155, -0.0456},
        { 0.0148,  0.0896,  0.1701,  0.1675},
        {0.0659,  0.0488, -0.1747, -0.3301}};
    
    weights = eval(transpose(weights));

    tensor_2d grads_next_layer = {
        { 0.0848,  0.1412,  0.1876, -0.4136},
        {0.1340, -0.3696,  0.1203,  0.1153}};

    tensor_2d expected_grads_in = {{ 0.176461, -0.075956, -0.099119, -0.097616,  0.059581,  0.007302,
        -0.023461,  0.116235},
        {-0.055739,  0.003485, -0.005231,  0.073852, -0.086417, -0.057697,
        0.008643, -0.068283}};

    fc_layer_t * layer = new fc_layer_t( {2, 8}, {4, 1, 1, 1}, false);
    layer->in = temp_in;
    layer->weights = weights;
    // tensor_2d out = layer->activate(temp_in, 1);

    tensor_2d grads_in = layer->calc_grads(grads_next_layer);
    // TODO: For this to work, override == operator of xtensor
    // if (out == expected_output)
    //     cout << "Fc Forward working correctly";

    cout << "\nExpected grads_in is\n";
    cout << expected_grads_in << endl;
    cout << "\nActual grads_in is\n";
    cout << grads_in;

    return 0;
    }