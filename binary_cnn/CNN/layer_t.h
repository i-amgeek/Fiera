#pragma once
#include "types.h"
#include "tensor_t.h"
#include "tensor.hpp"

#pragma pack(push, 1)
struct layer_t
{
	layer_type type;
	xarray<float> in;
};
#pragma pack(pop)