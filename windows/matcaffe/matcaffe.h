#pragma once
#include <memory>
#include <thread>
#include <vector>
#include "mex.h"
#include "caffe/caffe.hpp"


class MultiGPUSolver
{
public:
	explicit MultiGPUSolver(std::vector <int> gpus);
	~MultiGPUSolver();

	int  GetMaxIter();
	int  GetIter();
	void Restore();
	void Solve();
	void Step();
	void Snapshot();
	void Reset();
	caffe::Net<float> GetNet();
	caffe::Net<float> GetTestNet();
private:
	// task

private:
	// variable
	std::shared_ptr<caffe::P2PSync<float>> sync_solvers_;
	std::shared_ptr<caffe::Solver<float>>  root_solvers_;
	std::vector <int> gpus_;
	std::vector <std::thread> gpu_controller_thread_;
	void** gpu_task_ptr_;
};
