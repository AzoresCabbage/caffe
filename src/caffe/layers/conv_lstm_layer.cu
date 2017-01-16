#include <vector>

#include "caffe/layers/conv_lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
	template <typename Dtype>
	__device__ Dtype Sigmoid_gpu(const Dtype x) {
		return Dtype(1) / (Dtype(1) + exp(-x));
	}

	template <typename Dtype>
	__device__ Dtype d_Sigmoid_gpu(const Dtype x) {
		return x * (1 - x);
	}

	template <typename Dtype>
	__device__ Dtype Relu_gpu(const Dtype x) {
		return max(x, Dtype(0));
	}

	template <typename Dtype>
	__device__ Dtype d_Relu_gpu(const Dtype x) {
		return x > 0 ? 1 : 0;
	}

	template <typename Dtype>
	__device__ Dtype Tanh_gpu(const Dtype x) {
		return tanh(x);
	}

	template <typename Dtype>
	__device__ Dtype d_Tanh_gpu(const Dtype x) {
		return 1 - x * x;
	}

	template <typename Dtype>
	__global__ void ActivationForward(const int featmap_dim, Dtype* gate_i_t_data, Dtype* gate_f_t_data,
		Dtype* gate_c_t_data, Dtype* gate_o_t_data, Dtype* tanh_data, const Dtype* X_w_t, Dtype* H_w_t_1, Dtype* C_t_1,
		Dtype* H_t) {

		CUDA_KERNEL_LOOP(i, featmap_dim) {

			const Dtype* x_wi_t = X_w_t;
			const Dtype* x_wf_t = X_w_t + featmap_dim * 1;
			const Dtype* x_wc_t = X_w_t + featmap_dim * 2;
			const Dtype* x_wo_t = X_w_t + featmap_dim * 3;

			Dtype* h_wi_t_1 = H_w_t_1;
			Dtype* h_wf_t_1 = H_w_t_1 + featmap_dim * 1;
			Dtype* h_wc_t_1 = H_w_t_1 + featmap_dim * 2;
			Dtype* h_wo_t_1 = H_w_t_1 + featmap_dim * 3;

			gate_i_t_data[i] = Sigmoid_gpu(x_wi_t[i] + h_wi_t_1[i]);
			gate_f_t_data[i] = Sigmoid_gpu(x_wf_t[i] + h_wf_t_1[i]);
			gate_o_t_data[i] = Sigmoid_gpu(x_wo_t[i] + h_wo_t_1[i]);
			tanh_data[i] = Tanh_gpu(x_wc_t[i] + h_wc_t_1[i]);
			gate_c_t_data[i] = gate_f_t_data[i] * C_t_1[i]
				+ gate_i_t_data[i] * tanh_data[i];
			H_t[i] = gate_o_t_data[i] * Tanh_gpu(gate_c_t_data[i]);
		}
	}

	template <typename Dtype>
	__global__ void ActivationBackward(const int featmap_dim, Dtype* gate_i_t_data, Dtype* gate_f_t_data,
		Dtype* gate_c_t_data, Dtype* gate_o_t_data, Dtype* gate_i_t_diff, Dtype* gate_f_t_diff, Dtype* gate_c_t_diff,
		Dtype* gate_o_t_diff, Dtype* H_t_diff, Dtype* H_w_t_1_diff, Dtype* X_w_t_diff, Dtype* tanh_data, Dtype* C_t_1_diff, Dtype* C_t_1_data) {

		Dtype* x_wi_t_diff = X_w_t_diff;
		Dtype* x_wf_t_diff = X_w_t_diff + featmap_dim * 1;
		Dtype* x_wc_t_diff = X_w_t_diff + featmap_dim * 2;
		Dtype* x_wo_t_diff = X_w_t_diff + featmap_dim * 3;

		Dtype* h_wi_t_1_diff = H_w_t_1_diff;
		Dtype* h_wf_t_1_diff = H_w_t_1_diff + featmap_dim * 1;
		Dtype* h_wc_t_1_diff = H_w_t_1_diff + featmap_dim * 2;
		Dtype* h_wo_t_1_diff = H_w_t_1_diff + featmap_dim * 3;

		CUDA_KERNEL_LOOP(i, featmap_dim) {
			gate_o_t_diff[i] = H_t_diff[i] * Tanh_gpu(gate_c_t_data[i]);
			gate_c_t_diff[i] += H_t_diff[i] * gate_o_t_data[i] * d_Tanh_gpu(gate_c_t_data[i]);

			x_wo_t_diff[i] = gate_o_t_diff[i] * d_Sigmoid_gpu(gate_o_t_data[i]);
			h_wo_t_1_diff[i] = x_wo_t_diff[i];


			gate_f_t_diff[i] = gate_c_t_diff[i] * C_t_1_data[i];
			C_t_1_diff[i] = gate_c_t_diff[i] * gate_f_t_data[i];
			gate_i_t_diff[i] = gate_c_t_diff[i] * tanh_data[i];
			x_wc_t_diff[i] = gate_c_t_diff[i] * gate_i_t_data[i] * d_Tanh_gpu(tanh_data[i]);
			h_wc_t_1_diff[i] = x_wc_t_diff[i];

			x_wi_t_diff[i] = gate_i_t_diff[i] * d_Sigmoid_gpu(gate_i_t_data[i]);;
			h_wi_t_1_diff[i] = x_wi_t_diff[i];

			x_wf_t_diff[i] = gate_f_t_diff[i] * d_Sigmoid_gpu(gate_f_t_data[i]);
			h_wf_t_1_diff[i] = x_wf_t_diff[i];
		}
	}


	template <typename Dtype>
	void ConvLSTMLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_gpu_data();
		int featmap_dim = spatial_dims_ * num_output_;

		caffe_gpu_set<Dtype>(H_0_.count(0), Dtype(0.), H_0_.mutable_gpu_data());
		caffe_gpu_set<Dtype>(C_0_.count(0), Dtype(0.), C_0_.mutable_gpu_data());

		// For all input X: X[t] -> Wxi*X[t], Wxf*X[t], Wxc*X[t], Wxo*X[t] in conv_x_top_blob_
		conv_x_layer_->Forward(conv_x_btm_vec_, conv_x_top_vec_);

		for (int t = 0; t < seq_len_; ++t)
		{
			Dtype* gate_i_t_data = gate_i_.mutable_gpu_data() + gate_i_.offset(t);
			Dtype* gate_f_t_data = gate_f_.mutable_gpu_data() + gate_f_.offset(t);
			Dtype* gate_c_t_data = gate_c_.mutable_gpu_data() + gate_c_.offset(t);
			Dtype* gate_o_t_data = gate_o_.mutable_gpu_data() + gate_o_.offset(t);

			const Dtype* X_w_t = conv_x_top_blob_.gpu_data() + conv_x_top_blob_.offset(t);

			Dtype* H_t = top_data + top[0]->offset(t);
			Dtype* H_t_1 = t == 0 ? H_0_.mutable_gpu_data() : top_data + top[0]->offset(t - 1);
			Dtype* H_w_t_1 = conv_h_top_blob_.mutable_gpu_data();

			// H[t-1] -> Whi*H[t-1], Whf*H[t-1], Whc*H[t-1], Who*H[t-1] in conv_h_top_blob_
			conv_h_btm_blob_.data()->set_gpu_data(H_t_1);
			conv_h_layer_->Forward(conv_h_btm_vec_, conv_h_top_vec_);

			Dtype* C_t_1 = t == 0 ? C_0_.mutable_gpu_data() : gate_c_t_data - gate_c_.count(1);
			Dtype* tanh_data = gate_c_tanh_.mutable_cpu_data() + gate_c_tanh_.offset(t);

			ActivationForward<Dtype> << <CAFFE_GET_BLOCKS(featmap_dim), CAFFE_CUDA_NUM_THREADS >> >
				(featmap_dim, gate_i_t_data, gate_f_t_data, gate_c_t_data, gate_o_t_data, tanh_data,
				X_w_t, H_w_t_1, C_t_1, H_t);
			CUDA_POST_KERNEL_CHECK;

		}
	}

	template <typename Dtype>
	void ConvLSTMLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype* top_diff = top[0]->mutable_gpu_diff();
		Dtype* top_data = top[0]->mutable_gpu_data();
		int featmap_dim = spatial_dims_ * num_output_;

		caffe_gpu_set<Dtype>(H_0_.count(0), Dtype(0.), H_0_.mutable_gpu_diff());
		caffe_gpu_set<Dtype>(C_0_.count(0), Dtype(0.), C_0_.mutable_gpu_diff());
		//caffe_gpu_set<Dtype>(gate_i_.count(0), Dtype(0.), gate_i_.mutable_gpu_diff());
		//caffe_gpu_set<Dtype>(gate_f_.count(0), Dtype(0.), gate_f_.mutable_gpu_diff());
		caffe_gpu_set<Dtype>(gate_c_.count(0), Dtype(0.), gate_c_.mutable_gpu_diff());
		//caffe_gpu_set<Dtype>(gate_o_.count(0), Dtype(0.), gate_o_.mutable_gpu_diff());

		for (int t = seq_len_ - 1; t >= 0; --t)
		{
			Dtype* gate_i_t_data = gate_i_.mutable_gpu_data() + gate_i_.offset(t);
			Dtype* gate_f_t_data = gate_f_.mutable_gpu_data() + gate_f_.offset(t);
			Dtype* gate_c_t_data = gate_c_.mutable_gpu_data() + gate_c_.offset(t);
			Dtype* gate_o_t_data = gate_o_.mutable_gpu_data() + gate_o_.offset(t);

			Dtype* gate_i_t_diff = gate_i_.mutable_gpu_diff() + gate_i_.offset(t);
			Dtype* gate_f_t_diff = gate_f_.mutable_gpu_diff() + gate_f_.offset(t);
			Dtype* gate_c_t_diff = gate_c_.mutable_gpu_diff() + gate_c_.offset(t);
			Dtype* gate_o_t_diff = gate_o_.mutable_gpu_diff() + gate_o_.offset(t);

			Dtype* H_t_diff = top_diff + top[0]->offset(t);
			Dtype* H_w_t_1_diff = conv_h_top_blob_.mutable_gpu_diff();
			Dtype* X_w_t_diff = conv_x_top_blob_.mutable_gpu_data() + conv_x_top_blob_.offset(t);

			Dtype* C_t_1_diff = t == 0 ? C_0_.mutable_gpu_diff() : gate_c_t_diff - gate_c_.count(1);
			Dtype* C_t_1_data = t == 0 ? C_0_.mutable_gpu_data() : gate_c_t_data - gate_c_.count(1);
			Dtype* tanh_data = gate_c_tanh_.mutable_cpu_data() + gate_c_tanh_.offset(t);

			ActivationBackward<Dtype> << <CAFFE_GET_BLOCKS(featmap_dim), CAFFE_CUDA_NUM_THREADS >> >
				(featmap_dim, gate_i_t_data, gate_f_t_data, gate_c_t_data, gate_o_t_data,
				gate_i_t_diff, gate_f_t_diff, gate_c_t_diff, gate_o_t_diff,
				H_t_diff, H_w_t_1_diff, X_w_t_diff, tanh_data, C_t_1_diff, C_t_1_data);
			CUDA_POST_KERNEL_CHECK;

			Dtype* H_t_1_diff = t == 0 ? H_0_.mutable_gpu_diff() : top_diff + top[0]->offset(t - 1);
			Dtype* H_t_1_data = t == 0 ? H_0_.mutable_gpu_data() : top_data + top[0]->offset(t - 1);

			conv_h_btm_blob_.data()->set_gpu_data(H_t_1_data);
			conv_h_layer_->Backward(conv_h_top_vec_, vector<bool>{true}, conv_h_btm_vec_);

			// add conv_btm_blob to H[t-1] diff
			const Dtype* conv_h_t_1_diff = conv_h_btm_blob_.gpu_diff();
			caffe_gpu_add<Dtype>(conv_h_btm_blob_.count(0), H_t_1_diff, conv_h_t_1_diff, H_t_1_diff);
		}
		conv_x_layer_->Backward(conv_x_top_vec_, vector<bool>{propagate_down[0]}, conv_x_btm_vec_);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConvLSTMLayer);

}  // namespace caffe