#include <vector>
#include <algorithm>
#include <cmath>

#include "caffe/layers/conv_gru_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__device__ Dtype sigmoid(const Dtype x) {
	return Dtype(1) / (Dtype(1) + exp(-x));
}

template <typename Dtype>
__device__ Dtype d_sigmoid(const Dtype x) {
	return x * (1 - x);
}

template <typename Dtype>
__device__ Dtype relu(const Dtype x) {
	return max(x, Dtype(0));
}

template <typename Dtype>
__device__ Dtype d_relu(const Dtype x) {
	return x > 0 ? 1 : 0;
}

template <typename Dtype>
__device__ Dtype tanh(const Dtype x) {
	return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
}

template <typename Dtype>
__device__ Dtype d_tanh(const Dtype x) {
	return 1 - x * x;
}

template <typename Dtype>
__device__ Dtype hard_sigmoid(const Dtype x) {
	return max(min(0.2 * x + 0.5, Dtype(1)), Dtype(0));
}

template <typename Dtype>
__device__ Dtype d_hard_sigmoid(const Dtype x) {
	if (x >= 1 || x <= 0)
		return 0;
	else
		return 0.2;
}

template <typename Dtype>
__global__ void ActivationForward(const int feature_dims, Dtype* conv_x_top_t_data, Dtype* conv_h_top_data,
	Dtype* h_t_1_data, Dtype* h_t_data) {
	Dtype* Wr_x_t_data = conv_x_top_t_data + 0 * feature_dims;
	Dtype* Wz_x_t_data = conv_x_top_t_data + 1 * feature_dims;
	Dtype* Wh_x_t_data = conv_x_top_t_data + 2 * feature_dims;

	Dtype* Ur_h_t_1_data = conv_h_top_data + 0 * feature_dims;
	Dtype* Uz_h_t_1_data = conv_h_top_data + 1 * feature_dims;
	Dtype* Uh_h_t_1_data = conv_h_top_data + 2 * feature_dims;
	CUDA_KERNEL_LOOP(d, feature_dims) {
		// reset_gate or Rt = sigmoid(Wr*Xr + Ur*H[t-1])
		// and
		// update_gate or Zt = sigmoid(Wz*Xt + Uz*H[t-1])
		Wr_x_t_data[d] = sigmoid(Wr_x_t_data[d] + Ur_h_t_1_data[d]);
		Wz_x_t_data[d] = sigmoid(Wz_x_t_data[d] + Uz_h_t_1_data[d]);

		// Wh_data = tanh(W*X + U*(Rt .* H[t-1]))
		// Apply nonlinearity
		Wh_x_t_data[d] = tanh(Wh_x_t_data[d] + Wr_x_t_data[d] * Uh_h_t_1_data[d]);

		h_t_data[d] = (1 - Wz_x_t_data[d]) * h_t_1_data[d] + Wz_x_t_data[d] * Wh_x_t_data[d];
	}
}

template <typename Dtype>
__global__ void ActivationBackward(const int feature_dims, Dtype* conv_x_top_t_data, Dtype* conv_x_top_t_diff, 
	Dtype* conv_h_top_data, Dtype* conv_h_top_diff, Dtype* Uh_h_data, Dtype* h_t_1_data, Dtype* h_t_1_diff, Dtype* h_t_diff) {

	Dtype* Wr_x_t_data = conv_x_top_t_data + 0 * feature_dims;
	Dtype* Wz_x_t_data = conv_x_top_t_data + 1 * feature_dims;
	Dtype* Wh_x_t_data = conv_x_top_t_data + 2 * feature_dims;

	Dtype* Wr_x_t_diff = conv_x_top_t_diff + 0 * feature_dims;
	Dtype* Wz_x_t_diff = conv_x_top_t_diff + 1 * feature_dims;
	Dtype* Wh_x_t_diff = conv_x_top_t_diff + 2 * feature_dims;

	Dtype* Ur_h_t_1_data = conv_h_top_data + 0 * feature_dims;
	Dtype* Uz_h_t_1_data = conv_h_top_data + 1 * feature_dims;
	Dtype* Uh_h_t_1_data = conv_h_top_data + 2 * feature_dims;

	Dtype* Ur_h_t_1_diff = conv_h_top_diff + 0 * feature_dims;
	Dtype* Uz_h_t_1_diff = conv_h_top_diff + 1 * feature_dims;
	Dtype* Uh_h_t_1_diff = conv_h_top_diff + 2 * feature_dims;

	CUDA_KERNEL_LOOP(d, feature_dims) {
		// top_diff -> H[t-1]
		h_t_1_diff[d] += h_t_diff[d] * (1 - Wz_x_t_data[d]);

		// top_diff -> Ht_candidate
		Dtype h_candidate_diff = h_t_diff[d] * Wz_x_t_data[d];

		// top_diff -> gate_z
		Dtype gate_z_t_diff = h_t_diff[d] * (Wh_x_t_data[d] - h_t_1_data[d]);

		// gate z -> Wz*X and Uz*H[t-1]
		Wz_x_t_diff[d] = Uz_h_t_1_diff[d] = gate_z_t_diff * d_sigmoid(Wz_x_t_data[d]);

		// h candidate -> Wh*X[t] and R[t] and Uh*H[t-1]
		Wh_x_t_diff[d] = h_candidate_diff * d_tanh(Wh_x_t_data[d]);

		Dtype gate_r_t_diff = Wh_x_t_diff[d] * Uh_h_data[d];
		Uh_h_t_1_diff[d] = Wh_x_t_diff[d] * Wr_x_t_data[d];

		// gate r -> Wr*X and Ur*H[t-1]
		Wr_x_t_diff[d] = Ur_h_t_1_diff[d] = gate_r_t_diff * d_sigmoid(Wr_x_t_data[d]);
    }
}


template <typename Dtype>
void ConvGRULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	int feature_dims = num_output_ * spatial_dims_; // one input's size
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* conv_x_top_data = conv_x_top_blob_.mutable_gpu_data();
	Dtype* conv_h_top_data = conv_h_top_blob_.mutable_gpu_data();

	// Compute input to gate forward propagation
	// W*Xt, Wz*Xt, Wr*Xt
	conv_x_layer_->Forward(conv_x_bottom_vec_, conv_x_top_vec_);

	// Initialize previous state
	if (bottom.size() == 2)
	{
		h_0_.ShareData(*(bottom[1]));
		h_0_.ShareDiff(*(bottom[1]));
	}
	else
	{
		caffe_gpu_set(h_0_.count(0), Dtype(0.), h_0_.mutable_gpu_data());
	}

	// Compute recurrent forward propagation
	for (int tt = 0; tt < seq_len_; ++tt) {
		int t = tt;

		Dtype* conv_x_top_t_data = conv_x_top_data + conv_x_top_blob_.offset(t);

		Dtype* h_t_data = top_data + top[0]->offset(t);
		Dtype* h_t_1_data = t > 0 ? (top_data + top[0]->offset(t - 1)) : h_0_.mutable_gpu_data();

		conv_h_btm_blob_.data()->set_gpu_data(h_t_1_data);
		// Ur*H[t-1], Uz*H[t-1], Uh*H[t-1]
		conv_h_layer_->Forward(conv_h_bottom_vec_, conv_h_top_vec_);

		Dtype* Uh_h_t_1_data = conv_h_top_data + 2 * feature_dims;
		caffe_copy(feature_dims, Uh_h_t_1_data, Uh_h_.mutable_gpu_data() + Uh_h_.offset(t));

		ActivationForward<Dtype> << <CAFFE_GET_BLOCKS(feature_dims), CAFFE_CUDA_NUM_THREADS >> >(
			feature_dims, conv_x_top_t_data, conv_h_top_data, h_t_1_data, h_t_data);
		CUDA_POST_KERNEL_CHECK;
	}
	if (top.size() > 1)
	{
		caffe_copy(conv_x_top_blob_.count(0), conv_x_top_blob_.gpu_data(), top[1]->mutable_gpu_data());
	}
}

template <typename Dtype>
void ConvGRULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>*>& bottom) {

	int feature_dims = num_output_ * spatial_dims_;
	Dtype* top_data = top[0]->mutable_gpu_data();
	Dtype* top_diff = top[0]->mutable_gpu_diff();

	Dtype* conv_x_top_data = conv_x_top_blob_.mutable_gpu_data();
	Dtype* conv_x_top_diff = conv_x_top_blob_.mutable_gpu_diff();

	Dtype* conv_h_top_data = conv_h_top_blob_.mutable_gpu_data();
	Dtype* conv_h_top_diff = conv_h_top_blob_.mutable_gpu_diff();

	caffe_gpu_set(h_0_.count(0), Dtype(0.), h_0_.mutable_gpu_diff());

	for (int tt = seq_len_ - 1; tt >= 0; --tt) {
		int t = tt;

		Dtype* conv_x_top_t_data = conv_x_top_data + conv_x_top_blob_.offset(t);
		Dtype* conv_x_top_t_diff = conv_x_top_diff + conv_x_top_blob_.offset(t);

		Dtype* h_t_diff = top_diff + top[0]->offset(t);

		Dtype* h_t_1_data = t > 0 ? top_data + top[0]->offset(t - 1) : h_0_.mutable_gpu_data();
		Dtype* h_t_1_diff = t > 0 ? top_diff + top[0]->offset(t - 1) : h_0_.mutable_gpu_diff();
		Dtype* Uh_h_data = Uh_h_.mutable_gpu_data() + Uh_h_.offset(t);

		ActivationBackward<Dtype> << <CAFFE_GET_BLOCKS(feature_dims), CAFFE_CUDA_NUM_THREADS >> >(
			feature_dims, conv_x_top_t_data, conv_x_top_t_diff, conv_h_top_data, conv_h_top_diff, Uh_h_data,
			h_t_1_data, h_t_1_diff, h_t_diff);
		CUDA_POST_KERNEL_CHECK;

		conv_h_btm_blob_.data()->set_gpu_data(h_t_1_data);
		conv_h_layer_->Backward(conv_h_top_vec_, vector<bool>{true}, conv_h_bottom_vec_);
		const Dtype* hidden_diff_ = conv_h_btm_blob_.gpu_diff();
		caffe_gpu_add<Dtype>(feature_dims, h_t_1_diff, hidden_diff_, h_t_1_diff);
	}
	// Gradient w.r.t. bottom data 
	// accumulated all diff from input_pre_gate(conv) -> btm data
	// At the same time, calc all gradient for Wr, Wz, W
	conv_x_layer_->Backward(conv_x_top_vec_, vector<bool>{propagate_down[0]}, conv_x_bottom_vec_);
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvGRULayer);

}  // namespace caffe