#pragma once
#include "tensor_t.h"

#pragma pack(push, 1)

struct Data{
    tensor_4d images;
    tensor_4d labels;

    void operator = (Data data){
		images = data.images;
		labels = data.labels;
	}
};

#pragma pack(pop)