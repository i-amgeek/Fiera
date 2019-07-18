#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xio.hpp"

int main(int argc, char* argv[])
{
    xt::xarray<double> arr1
      {1.0, 2.0, 3.0};

    xt::xarray<unsigned int> arr2
      {4, 5, 6, 7};

     xt::xarray weights = 
        {{{{ 0.1558,  0.0148,  0.0896,  0.170}}},
        {{{-0.3019,  0.0659,  0.0488, -0.1747}}},
        {{{ 0.3323,  0.2685,  0.1723, -0.1887}}},
        {{{-0.2773,  0.0909,  0.3247,  0.3051}}},
        {{{ 0.2707,  0.1876, -0.0787,  0.3235}}},
        {{{-0.0614, -0.2735, -0.1970,  0.0407}}},
        {{{ 0.1819,  0.2517, -0.0890, -0.0612}}},
        {{{ 0.1378,  0.1217, -0.2155, -0.0456}}}};

    // arr2.reshape({4, 1});

   auto shape = weights.shape();

   for (auto& el : shape) {std::cout << el << ", "; }

    // xt::xarray<double> res = xt::pow(arr1, arr2);

    // std::cout << res;
    return 0;
}