#ifndef INC_IM2COL
#define INC_IM2COL

tensor_3d im2col(tensor_4d in, int hh, int ww, int stride){

    auto start = Clock::now();
    auto start_1 = Clock::now();
    auto finish = Clock::now();
    std::chrono::duration<double> elipsed, elipsed_1;

    int m = in.shape()[0];
    int c = in.shape()[1];
    int h = in.shape()[2];
    int w = in.shape()[3];    
    int new_h = (h-hh) / stride + 1;
    int new_w = (w-ww) / stride + 1;
    tensor_3d col ({m, new_h*new_w, c*hh*ww});
    for (int e=0; e<m; e++){
        for (int i=0; i<new_h; i++){
            for (int j=0; j<new_w; j++){

                auto patch = xt::view(in, e, all(),xt::range(i*stride,i*stride+hh),xt::range(j*stride,j*stride+ww));
                xt::view(col, e, i*new_w+j, all()) = xt::reshape_view(patch, {c*hh*ww});

            }
        }
    }

    return col;
}


tensor_4d col2im(tensor_3d mul, int h_prime, int w_prime ){
    int m = mul.shape()[0];
    int F = mul.shape()[2];
    tensor_4d out = zeros<float>({m, F, h_prime,w_prime});
    for (int e=0; e<m; e++){
        for (int i=0; i<F; i++){
            auto col = xt::view(mul, e, all(), i);
            xt::view(out, e, i, all(), all()) = xt::reshape_view(col, {h_prime,w_prime});
        }
    }

    return out;
    
}


tensor_4d col2im_back(tensor_3d dim_col,int h_prime, int w_prime, int stride, int hh,int ww, int c){
    int H = (h_prime - 1) * stride + hh;
    int W = (w_prime - 1) * stride + ww;
    int m = dim_col.shape()[0];
    tensor_4d grad_in = xt::zeros<float>({m, c, H, W});
    xarray<float> dim_col_arr = dim_col;
    for (int e=0; e<m; e++){
        for (int i=0; i< h_prime*w_prime; i++){
            auto row = xt::view(dim_col_arr, e, i, all());
            int h_start = int(i / w_prime) * stride;
            int w_start = (i % w_prime) * stride;
            xt::view(grad_in, e, all(), range(h_start,h_start+hh), range(w_start, w_start+ww)) += xt::reshape_view(row, {c,hh,ww});
        }
    }
    return grad_in;
}

#endif //INC_IM2COL