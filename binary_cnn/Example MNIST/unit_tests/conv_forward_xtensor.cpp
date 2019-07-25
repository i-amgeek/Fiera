#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/conv_layer_new.h"
#include "tensor.hpp"

using namespace std;

int main()
{
    tensor_4d temp_in
        {{{{-2.9662, -1.0606, -0.3090},
          { 0.9343, -0.3821, -1.1669},
          { 0.3636, -0.3156,  1.1450}},

         {{-0.3822, -0.3553,  0.7542},
          { 0.6901, -0.1443,  1.6120},
          { 1.5671, -1.2432, -1.7178}}},


        {{{-0.5824, -0.6153,  0.4105},
          { 1.7675, -0.0832,  0.5087},
          { 1.1178,  1.1286,  0.1416}},

         {{-0.5458,  1.1542, -1.5366},
          {-0.5577, -0.4383,  1.1572},
          { 0.0889,  0.2659, -0.1907}}}};

    tensor_4d filters
           {{{{ 0.0247, -0.2130},
              { 0.1126,  0.1109}},

             {{-0.1890, -0.0530},
              {-0.2071,  0.0917}}},

            {{{-0.0952,  0.2484},
              { 0.2510,  0.0360}},

             {{-0.1507, -0.2077},
              {-0.0388, -0.0995}}}};
              
    tensor_4d expected_output
       {{{{ 0.1503,  0.0721},
          {-0.4510,  0.3723}},

         {{ 0.3587, -0.3715},
          {-0.1152, -0.3853}}},


        {{{ 0.4237,  0.0047},
          { 0.4470, -0.0187}},

         {{ 0.2511,  0.2050},
          { 0.2774,  0.2570}}}};
    

    conv_layer_t * layer = new conv_layer_t( 1, 2, 2, {2, 2, 3, 3}, false);
    layer->filters = filters;
    tensor_4d out = layer->activate(temp_in, true);
    // if (out == expected_output) cout << "Convlayer forward working correctly";

    cout << "Expected output is\n";
    cout << expected_output;
    cout << "\nActual output is\n";
    cout << out ;

    return 0;
}
