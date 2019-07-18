#include "../tensor.hpp"

tensor_3d im2col(tensor_4d in, xshape w, int stride=0){
    int w_h = w[2];
    int w_w = w[3];
    int m = in.shape()[0];
    int c = in.shape()[1];
    int h = in.shape()[2];
    int w = in.shape()[3];    
    int new_h = (h-hh) / stride + 1;
    int new_w = (w-ww) / stride + 1;
    tensor_3d col = zeros<double>({m, new_h*new_w, c*hh*ww});
    for (int e=0; e<m; e++){
        for (int i=0; i<h; i++){
            for (int j=0; j<w; j++){
                auto patch = strided_view(in, {m, all(),i*stride:i*stride+hh,j*stride:j*stride+ww});
                col(e, i*new_w+j, all()) = patch.reshape({-1});
            }
        }
    }
    return col;
}


tensor_4d col2im(tensor_3d mul, int h_prime, int w_prime ){
    int m = mul.shape()[0];
    int F = mul.shape()[2];
    tensor_4d out = zeros({m, F, h_prime,w_prime});
    for (int e=0; e<m; e++){
        for (int i=0; i<F; i++){
            col = mul({e, all(), i});
            out({e, i, all(), all()}) = col.reshape({h_prime,w_prime});
        }
    }

    return out;
    
}


tensor_3d col2im_back(dim_col,h_prime,w_prime,stride,hh,ww,c){
    int H = (h_prime - 1) * stride + hh;
    int W = (w_prime - 1) * stride + ww;
    int m = dim_col.shape()[0];
    tensor_4d grad_in = xt::zeros({m, c, H, W});
    for (int e=0; e<m; e++){
        for (int i=0; i< h_prime*w_prime; i++){
            tensor_3d row = dim_col({e, i, all()});
            int h_start = (i / w_prime) * stride;
            int w_start = (i % w_prime) * stride;
            xt::view(grad_in, e, all(), range(h_start,h_start+hh), range(w_start, w_start+ww)) = row.reshape({c,hh,ww});
    }
    }
    return grad_in;
}
