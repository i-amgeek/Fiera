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
    xarray<float> out_data
       {{{{ 0.1503,  0.0721},
          {-0.4510,  0.3723}},

         {{ 0.3587, -0.3715},
          {-0.1152, -0.3853}}},


        {{{ 0.4237,  0.0047},
          { 0.4470, -0.0187}},

         {{ 0.2511,  0.2050},
          { 0.2774,  0.2570}}}};

    json weights;
    xarray<float> in_data;

    xt::to_json(weights["data"], out_data);
    ofstream fileO("test.weights");
    fileO << weights << endl;
    fileO.close();

    ifstream fileI("test.weights");
    fileI >> weights; 
    xt::from_json(weights["data"], in_data);
    fileI.close();
    
    if (xt::allclose(out_data, in_data, 1e-2))
        cout << "Weights save load working correctly";
    else
        cout << "Weights save load not working correctly";

    return 0;
}