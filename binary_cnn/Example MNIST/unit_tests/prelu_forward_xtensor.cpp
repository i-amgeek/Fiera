#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/prelu_layer_new_t.h"
#include "tensor.hpp"

using namespace std;

int main()
{
    tensor_4d temp_in
       {{{{ 0.1503,  0.0721},
          {-0.4510,  0.3723}},

         {{ 0.3587, -0.3715},
          {-0.1152, -0.3853}}},


        {{{ 0.4237,  0.0047},
          { 0.4470, -0.0187}},

         {{ 0.2511,  0.2050},
          { 0.2774,  0.2570}}}};

    tensor_4d expected_output
    {{{{ 0.1503  ,  0.0721  },
       {-0.02255 ,  0.3723  }},

      {{ 0.3587  , -0.018575},
       {-0.00576 , -0.019265}}},


     {{{ 0.4237  ,  0.0047  },
       { 0.447   , -0.000935}},

      {{ 0.2511  ,  0.205   },
       { 0.2774  ,  0.257   }}}};

    prelu_layer_t * layer = new prelu_layer_t( {2,2,2,2}, false);
    tensor_4d out = layer->activate(temp_in, true);

    // if (out == expected_output) cout << "Prelu forward working correctly";

    cout << "\nExpected output is\n\n";
    cout << expected_output;
    cout << "\nActual output is\n";
    cout << out ;

    return 0;
}