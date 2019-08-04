#ifndef INC_EXTRAS
#define INC_EXTRAS


#include "../tensor.hpp"

tensorg_2d convert_2d_float_to_gradient(tensor_2d input){


    int nx = input.shape()[0];
    int ny = input.shape()[1];
    tensorg_2d temp({nx,ny});

    for(int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
            temp(i, j).grad = xt::view(input, i, j);

    return temp;

}

tensorg_4d convert_4d_float_to_gradient(tensor_4d input){


    int nx = input.shape()[0];
    int ny = input.shape()[1];
    int h = input.shape()[2];
    int w = input.shape()[3];

    tensorg_4d temp({nx,ny,h,w});

    for(int i=0; i<nx; i++)
        for(int j=0; j<ny; j++)
          for(int k=0; k<h; k++)
            for(int l=0; l<w; l++)
            temp( i, j, k, l).grad = xt::view(input, i, j, k, l);

    return temp;

}

#endif //INC_EXTRAS