#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
/* #include "../../CNN/prelu_layer_new_t.h" */
#include "tensor.hpp"

using namespace std;

int main()
{
    xarray<float> temp_in
       {{{{ 0.1503,  0.0721},
          {-0.4510,  0.3723}},

         {{ 0.3587, -0.3715},
          {-0.1152, -0.3853}}},


        {{{ 0.4237,  0.0047},
          { 0.4470, -0.0187}},

         {{ 0.2511,  0.2050},
          { 0.2774,  0.2570}}}};


    xarray<float> a = xt::argmax(temp_in, 1);
    cout << xt::adapt(a.shape());
    cout << a;
    return 0;
}
