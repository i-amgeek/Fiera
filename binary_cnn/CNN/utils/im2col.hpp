#ifndef INC_IM2COL
#define INC_IM2COL

xarray<float> im2col(xarray<float> in, int hh, int ww, int stride){

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
    auto col = xt::xarray<float>::from_shape({(uint)m, (uint)new_h*new_w, (uint)c*hh*ww});
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


xarray<float> col2im(xarray<float> mul, int h_prime, int w_prime ){
    int m = mul.shape()[0];
    int F = mul.shape()[2];
    auto out = xt::xarray<float>::from_shape({(uint)m, (uint)F, (uint)h_prime, (uint)w_prime});
    for (int e=0; e<m; e++){
        for (int i=0; i<F; i++){
            auto col = xt::view(mul, e, all(), i);
            xt::view(out, e, i, all(), all()) = xt::reshape_view(col, {h_prime,w_prime});
        }
    }

    return out;
    
}


xarray<float> col2im_back(xarray<float> dim_col,int h_prime, int w_prime, int stride, int hh,int ww, int c){
    int H = (h_prime - 1) * stride + hh;
    int W = (w_prime - 1) * stride + ww;
    int m = dim_col.shape()[0];
    auto grad_in = xt::xarray<float>::from_shape({(uint)m, (uint)c,(uint) H, (uint)W});
    for (int e=0; e<m; e++){
        for (int i=0; i< h_prime*w_prime; i++){
            auto row = xt::view(dim_col, e, i, all());
            int h_start = int(i / w_prime) * stride;
            int w_start = (i % w_prime) * stride;
            xt::view(grad_in, e, all(), range(h_start,h_start+hh), range(w_start, w_start+ww)) += xt::reshape_view(row, {c,hh,ww});
        }
    }
    return grad_in;
}

#endif //INC_IM2COL