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
        {0.0848, -0.3588,  0.1876,  0.0864},
        {-0.3660,  0.1304,  0.1203,  0.1153}};

    tensor_2d expected_grads_in = {{-0.052139,  0.031144, -0.031169,  0.059484, -0.096869, -0.076348,
   0.015489, -0.073215},
 {-0.087639,  0.187585, -0.046781, -0.032198, -0.051517, -0.065747,
   0.046043, -0.076833}};

    fc_layer_t * layer = new fc_layer_t( {2, 8}, {4, 1, 1, 1}, false);
    
    layer->in = temp_in;
    layer->weights = weights;

    tensor_2d grads_in = layer->calc_grads(grads_next_layer);

    layer->fix_weights(0.01);

    tensor_2d expected_updated_weights{{ 0.329617, -0.279922,  0.270211, -0.062576,  0.181786,  0.138135,
   0.014739,  0.066529},
 { 0.273576,  0.095872,  0.186324, -0.274708,  0.251601,  0.121419,
   0.089489,  0.048436},
 { 0.17048 ,  0.322914, -0.077588, -0.195547, -0.088871, -0.215504,
   0.170207, -0.174815},
 {-0.189273,  0.304536,  0.324153,  0.04163 , -0.061116, -0.04565 ,
   0.167565, -0.33025 }};

    // cout<<"updated weights: \n"<<layer->weights;

    if (xt::allclose(grads_in, expected_grads_in, 1e-2) and xt::allclose(layer->weights, expected_updated_weights, 1e-2))
        cout << "Fc backward working correctly";
    else{
        // cout << xt::isclose(grads_in, expected_grads_in, 1e-2);
        cout << "\nExpected grads_in is\n";
        cout << expected_grads_in << endl;
        cout << "\nActual grads_in is\n";
        cout << grads_in<<endl;

        cout << "\nExpected updated_weights is\n";
        cout << expected_updated_weights << endl;
        cout << "\nActual updated_weights is\n";
        cout << layer->weights<<endl;


    }

    return 0;
    }