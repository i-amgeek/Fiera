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
        {{0.0000, 0.0000, 0.3510, 0.5182, 0.0000, 0.0000, 0.0000, 0.4201},
         {0.0000, 0.3114, 0.0000, 0.0000, 0.0045, 0.2879, 0.4376, 0.1286}};

    tensor_2d weights 
       {{ 0.1558,  0.0148,  0.0896,  0.170},
        {-0.3019,  0.0659,  0.0488, -0.1747},
        { 0.3323,  0.2685,  0.1723, -0.1887},
        {-0.2773,  0.0909,  0.3247,  0.3051},
        { 0.2707,  0.1876, -0.0787,  0.3235},
        {-0.0614, -0.2735, -0.1970,  0.0407},
        { 0.1819,  0.2517, -0.0890, -0.0612},
        { 0.1378,  0.1217, -0.2155, -0.0456}};
    
    weights = eval(transpose(weights));
    tensor_2d expected_output =
       {{ 0.0308,  0.1925,  0.1382,  0.0727},
        {-0.0132,  0.0684, -0.1085, -0.0739}};

    fc_layer_t * layer = new fc_layer_t( {2, 8}, {4, 1, 1, 1}, false);
    layer->in = temp_in;
    layer->weights = weights;
    tensor_2d out = layer->activate(temp_in, 1);

    if (xt::allclose(out, expected_output, 1e-2))
        cout << "Fc Forward working correctly";
    else{
        cout << "\nExpected output is\n";
        cout << expected_output << endl;
        cout << "\nActual output is\n";
        cout << out;
    }
    return 0;
    }