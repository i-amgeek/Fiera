#pragma once
#include "tensor.hpp"
#include "data.h"

#pragma pack(push, 1)

struct Dataset
{
	Data train;
	Data test;
	Data validation;

	Dataset(uint ntrain, uint ntest, uint nval, uint nout, uint img_w, uint img_h, uint img_c){
		train.images = xt::xarray<float>::from_shape({ntrain, img_c, img_w, img_h});
		train.labels = xt::xarray<float>::from_shape({ntrain, nout, 1, 1});
		test.images = xt::xarray<float>::from_shape({ntest, img_c, img_w, img_h});
		test.labels = xt::xarray<float>::from_shape({ntest, nout, 1, 1});
		validation.images = xt::xarray<float>::from_shape({nval, img_c, img_w, img_h});
		validation.labels = xt::xarray<float>::from_shape({nval, nout, 1, 1});
	}
	void operator = (Dataset data){
		train = data.train;
		test = data.test;
		validation = data.validation;
	}
};

#pragma pack(pop)