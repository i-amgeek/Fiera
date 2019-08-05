#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <algorithm>
#include "byteswap.h"
#include "../../CNN/conv_layer_new_t.h"

#include "tensor.hpp"
#include "xtensor/xmasked_view.hpp"


using namespace std;

int main()
{
    // tensor_t<float> temp_in(2, 3, 3, 2), filters(2, 2, 2, 2), expected_output(2, 2, 2, 2), grad_next_layer(2, 2, 2, 2);
    // std::vector<std::vector<std::vector<std::vector<float> > > > vect=

    xarray<float> temp_in{{{{-0.0145, -0.3839, -2.9662},
          {-1.0606, -0.3090,  0.9343},
          {-0.3821, -1.1669, -0.4375}},

         {{-2.1085,  1.1450, -0.3822},
          {-0.3553,  0.7542,  0.6901},
          {-0.1443, -0.5146,  0.8005}}},


        {{{-1.2432, -1.7178,  1.7518},
          { 0.9796,  0.4105,  1.7675},
          {-0.0832,  0.5087,  1.1178}},

         {{ 1.1286,  0.5013,  1.4206},
          { 1.1542, -1.5366, -0.5577},
          {-0.4383,  1.1572,  0.0889}}}};

    // cout<<"*********image*****\n\n";
    // print_tensor(temp_in);

    xarray<float> grad_next_layer{{{{-1.5361e-01,  1.8545e-01},
          {-8.3970e-03,  9.9813e-03}},

         {{-6.4969e-04, -4.2289e-04},
          { 6.6066e-04, -3.7529e-04}}},


        {{{-1.9954e-02,  3.6370e-02},
          {-2.1987e-01, -8.5993e-03}},

         {{-1.1953e-04, -1.4750e-03},
          { 2.0811e-02, -1.2010e-03}}}};
      
      // from_vector(vect);
      // cout<<"**********grad_next_layer**********\n";
      // print_tensor(grad_next_layer);

    xarray<float> filters{{{{ 0.0247, -0.2130},
              { 0.1126,  0.1109}},

             {{-0.1890, -0.0530},
              {-0.2071,  0.0917}}},

            {{{-0.0952,  0.2484},
              { 0.2510,  0.0360}},

             {{-0.1507, -0.2077},
              {-0.0388, -0.0995}}}};
              
    // cout<<xt::sign(filters);
    // .from_vector(vect);

  //  xarray<float> expected_output{{{{ 0.1503,  0.0721},
  //         {-0.4510,  0.3723}},

  //        {{ 0.3587, -0.3715},
  //         {-0.1152, -0.3853}}},


  //       {{{ 0.4237,  0.0047},
  //         { 0.4470, -0.0187}},

  //        {{ 0.2511,  0.2050},
  //         { 0.2774,  0.2570}}}};
    
    // .from_vector(vect);


    conv_layer_t * layer = new conv_layer_t( 1, 2, 2, {2,3,3,2}, false);
    layer->in = temp_in;
    layer->filters = filters;

    xarray<float> out = layer->activate(temp_in, true);

   
    
    xarray<float> grads_in = layer->calc_grads(grad_next_layer);

    // cout<<grads_in;
}
