#pragma once
#include "tensor_t.h"

#pragma pack(push, 1)

struct Data{
    xarray<float> images;
    xarray<float> labels;

    void operator = (Data data){
		images = data.images;
		labels = data.labels;
	}
};

#pragma pack(pop)