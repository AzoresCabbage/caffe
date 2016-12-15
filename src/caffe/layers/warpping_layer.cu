#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/warping_layer.hpp"

namespace caffe {

	template <typename Dtype>
	inline __device__ Dtype caffe_gpu_atomic_add(const Dtype val, Dtype* address);

	template <>
	inline __device__
		float caffe_gpu_atomic_add(const float val, float* address) {
			return atomicAdd(address, val);
		}

	// double atomicAdd implementation taken from:
	// http://docs.nvidia.com/cuda/cuda-c-programming-guide/#axzz3PVCpVsEG
	template <>
	inline __device__
		double caffe_gpu_atomic_add(const double val, double* address) {
			unsigned long long int* address_as_ull =  // NOLINT(runtime/int)
				// NOLINT_NEXT_LINE(runtime/int)
				reinterpret_cast<unsigned long long int*>(address);
			unsigned long long int old = *address_as_ull;  // NOLINT(runtime/int)
			unsigned long long int assumed;  // NOLINT(runtime/int)
			do {
				assumed = old;
				old = atomicCAS(address_as_ull, assumed,
					__double_as_longlong(val + __longlong_as_double(assumed)));
			} while (assumed != old);
			return __longlong_as_double(old);
		}

	template <typename Dtype>
	__global__ void WarpingForward(const int nthreads, const int channels,
		const int spatial_dim, const int height, const int width,
		const Dtype* data, const Dtype* Displacement_data, Dtype* transformed_data,
		const bool pad_zero) {
		CUDA_KERNEL_LOOP(index, nthreads) {
			int cn = index / spatial_dim;
			int n = cn / channels; //num

			int s = index % spatial_dim;
			int h = s / width;
			int w = s % width;


			Dtype botx = w + Displacement_data[n * 2 * spatial_dim + h * width + w];
			Dtype boty = h + Displacement_data[n * 2 * spatial_dim + spatial_dim + h * width + w];

			if (!pad_zero){
				botx = max(min(botx, Dtype(width - 1)), Dtype(0));
				boty = max(min(boty, Dtype(height - 1)), Dtype(0));
			}

			int botidxoffcn = cn * spatial_dim;

			float accum_value = 0;

			int floor_botx = floor(botx);
			int floor_boty = floor(boty);
			for (int by = floor_boty; by <= floor_boty + 1; by++)  {
				for (int bx = floor_botx; bx <= floor_botx + 1; bx++)  {
					if (bx < 0 || bx >(width - 1) || by < 0 || by >(height - 1))
						continue;
					float weight = (1.0f - abs((float)bx - botx)) * (1.0f - abs((float)by - boty));
					float sample = data[botidxoffcn + by * width + bx];
					accum_value += sample * weight;
				}
			}
			transformed_data[index] = accum_value;
		}
	}

	template <typename Dtype>
	__global__ void WarpingBackward(const int nthreads, const int channels,
		const int spatial_dim, const int height, const int width,
		const Dtype* data, const Dtype* Displacement_data, const Dtype* top_diff,
		Dtype* data_diff, Dtype* Displacement_diff, const bool prop_data, const bool prop_dis,
		const bool pad_zero) {
		CUDA_KERNEL_LOOP(index, nthreads) {

			int n = index / spatial_dim;
			int s = index % spatial_dim;
			int h = s / width;
			int w = s % width;

			Dtype botx = w + Displacement_data[n * 2 * spatial_dim + h * width + w];
			Dtype boty = h + Displacement_data[n * 2 * spatial_dim + spatial_dim + h * width + w];

			if (!pad_zero){
				botx = max(min(botx, Dtype(width - 1)), Dtype(0));
				boty = max(min(boty, Dtype(height - 1)), Dtype(0));
			}

			int floor_botx = floor(botx);
			int floor_boty = floor(boty);
			for (int by = floor_boty; by <= floor_boty + 1; by++)  {
				for (int bx = floor_botx; bx <= floor_botx + 1; bx++)  {
					if (bx < 0 || bx >(width - 1) || by < 0 || by >(height - 1))
						continue;
					float weightx = 1.0f - abs((float)bx - botx);
					float weighty = 1.0f - abs((float)by - boty);

					for (int c = 0; c < channels; c++) {
						int botidxoffcn = (n * channels + c) * spatial_dim;

						if (prop_data)
							caffe_gpu_atomic_add(top_diff[botidxoffcn + h * width + w] * weightx * weighty, data_diff + botidxoffcn + by * width + bx);

						if (!prop_dis)
							continue;

						Dtype diff_sample = top_diff[botidxoffcn + h * width + w] * data[botidxoffcn + by * width + bx];
						if (pad_zero || (botx > Dtype(0) && botx < Dtype(width - 1))){
							if (bx > botx)
								Displacement_diff[n * 2 * spatial_dim + h * width + w] += diff_sample * weighty;
							else
								Displacement_diff[n * 2 * spatial_dim + h * width + w] -= diff_sample * weighty;
						}
						if (pad_zero || (boty > Dtype(0) && boty < Dtype(height - 1))){
							if (by > boty)
								Displacement_diff[n * 2 * spatial_dim + spatial_dim + h * width + w] += diff_sample * weightx;
							else
								Displacement_diff[n * 2 * spatial_dim + spatial_dim + h * width + w] -= diff_sample * weightx;
						}
					}
				}
			}
		}
	}

	template <typename Dtype>
	void WarpingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* top_data = top[0]->mutable_gpu_data();
		const Dtype* Displacement_data = bottom[1]->gpu_data();

		int count = bottom[0]->count();
		int num = bottom[0]->shape(0);
		int channels = bottom[0]->shape(1);
		int height = bottom[0]->shape(2);
		int width = bottom[0]->shape(3);

		bool pad_zero = this->layer_param_.warping_param().pad_method() == WarpingParameter_PadMethod_ZERO;

		WarpingForward<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >
			(count, channels, height * width, height, width, bottom_data, Displacement_data, top_data, pad_zero);
	}

	template <typename Dtype>
	void WarpingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* bottom_data = bottom[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		Dtype* data_diff = bottom[0]->mutable_gpu_diff();

		int count = bottom[0]->count();
		int num = bottom[0]->shape(0);
		int channels = bottom[0]->shape(1);
		int height = bottom[0]->shape(2);
		int width = bottom[0]->shape(3);

		const Dtype*  Displacement_data = bottom[1]->gpu_data();
		Dtype* Displacement_diff = bottom[1]->mutable_gpu_diff();

		caffe_gpu_set<Dtype>(bottom[0]->count(), 0, data_diff);
		caffe_gpu_set<Dtype>(bottom[1]->count(), 0, Displacement_diff);

		if (!propagate_down[0] && !propagate_down[1])
			return;

		bool pad_zero = this->layer_param_.warping_param().pad_method() == WarpingParameter_PadMethod_ZERO;

		WarpingBackward<Dtype> << <CAFFE_GET_BLOCKS(num * height * width), CAFFE_CUDA_NUM_THREADS >> >
			(num * height * width, channels, height * width, height, width, bottom_data, Displacement_data,
			top_diff, data_diff, Displacement_diff, propagate_down[0], propagate_down[1], pad_zero);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(WarpingLayer);
}