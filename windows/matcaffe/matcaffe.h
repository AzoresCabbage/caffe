#pragma once
#include <memory>
#include "mex.h"
#include "caffe/caffe.hpp"


class MultiGPUSolver
{
public:
	MultiGPUSolver();
	~MultiGPUSolver();
private:
	std::shared_ptr<caffe::P2PSync<float>> sync_solvers_;
	std::shared_ptr<caffe::Solver<float>>  root_solvers_;
};