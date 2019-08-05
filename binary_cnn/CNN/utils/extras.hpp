#ifndef INC_EXTRAS
#define INC_EXTRAS


#include "../tensor.hpp"

xarray<gradient_t> convert_2d_float_to_gradient(xarray<float> input){


    int nx = input.shape()[0];
    int ny = input.shape()[1];
    auto temp = xt::xarray<gradient_t>::from_shape({nx,ny});

    for(int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
            temp(i, j).grad = input(i, j);

    return temp;

}

xarray<gradient_t> convert_4d_float_to_gradient(xarray<float> input){


    int nx = input.shape()[0];
    int ny = input.shape()[1];
    int h = input.shape()[2];
    int w = input.shape()[3];

    auto temp = xt::xarray<gradient_t>::from_shape({nx,ny,h,w});

    for(int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
          for(int k=0; k<h; k++)
            for(int l=0; l<w; l++)
            temp( i, j, k, l).grad = input(i, j, k, l);

    return temp;

}

#endif //INC_EXTRAS