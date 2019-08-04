
#pragma once
#include <math.h>
#include <float.h>
#include <string.h>
#include "layer_t.h"
#include "optimization_method.h"
#include "gradient_t.h"
#include "tensor_bin_t.h"
using namespace std;
float cross_entropy(tensor_2d& predicted ,tensor_2d& actual, bool debug=false){
        float cost = 0.0;
        int index;
        int m = predicted.shape()[0]; 
        tensor_1d target = xt::argmax(actual, 1);

        for(int e = 0; e < m; e++){
            cost -= predicted(e, target(e));
        return cost;
    }
