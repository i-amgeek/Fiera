#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "tensor.hpp"

using namespace std;

int main()
{
    tensor_4d data
       {{{{ 0.1503,  0.0721},
          {-0.4510,  0.3723}},

         {{ 0.3587, -0.3715},
          {-0.1152, -0.3853}}},


        {{{ 0.4237,  0.0047},
          { 0.4470, -0.0187}},

         {{ 0.2511,  0.2050},
          { 0.2774,  0.2570}}}};
    
    layer_type type = layer_type::conv;

    json weights = { 
        { "type", "conv" },
        { "size", array_size},
    };

    xt::to_json(weights["data"], data);
    ofstream file("test.weights");
    file << weights << endl;
    file.close();
}