/*! Custom object for holding size in m, x, y, z*/

#pragma once
#include "../Libraries/json.hpp"
using json = nlohmann::json;
struct point_t
{
	int m, x, y, z;
	void from_json( json j){
		// TODO: 'assert 'j' is of form [int, int, int, int]
		m = j[0];
		x = j[1];
		y = j[2];
		z = j[3];
	}

	bool operator >(int i){
		if (m > i && x > i && y > i && z > i)
			return true;
		else
			return false;
	}

	bool operator <(int i){
		if (m < i && x < i && y < i && z < i)
			return true;
		else
			return false;
	}

	void operator = (point_t size){
		m = size.m;
		x = size.x;
		y = size.y;
		z = size.z;
	}
};
typedef point_t tdsize;