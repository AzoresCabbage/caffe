#include <vector>

#include "caffe/layers/conv_rnn_layer.hpp"
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
	__global__ void ActivationForward(const int nthreads, const int act_type, const Dtype* src1, Dtype* dst) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			if (act_type == 1)
				dst[i] = Tanh_gpu(src1[i]);
			else if (act_type == 2)
				dst[i] = Relu_gpu(src1[i]);
			else
				dst[i] = Sigmoid_gpu(src1[i]);
		}
	}

	template <typename Dtype>
	__global__ void ActivationForward(const int nthreads, const int act_type, const Dtype* src1, const Dtype* src2, Dtype* dst) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			if (act_type == 1)
				dst[i] = Tanh_gpu(src1[i] + src2[i]);
			else if (act_type == 2)
				dst[i] = Relu_gpu(src1[i] + src2[i]);
			else
				dst[i] = Sigmoid_gpu(src1[i] + src2[i]);
		}
	}

	template <typename Dtype>
	__global__ void ActivationBackward(const int nthreads, const int act_type,
		Dtype* top_data_t, Dtype* diff, Dtype* dst) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			dst[i] = diff[i] *
				(act_type == 1 ? d_Tanh_gpu(top_data_t[i])
				: act_type == 2 ? d_Relu_gpu(top_data_t[i]) : d_Sigmoid_gpu(top_data_t[i]));
		}
	}

	template <typename Dtype>
	__global__ void ActivationBackward(const int nthreads, const int act_type, 
		Dtype* top_data_t, Dtype* diff, Dtype* dst1, Dtype* dst2) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			dst1[i] = diff[i] *
				(act_type == 1 ? d_Tanh_gpu(top_data_t[i]) 
				: act_type == 2 ? d_Relu_gpu(top_data_t[i]) : d_Sigmoid_gpu(top_data_t[i]));
			dst2[i] = dst1[i];
		}
	}


	template <typename Dtype>
	void ConvRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_gpu_data();
		int featmap_dim = spatial_dims_ * num_output_;
		const int act_type = act_func_->act_typeid();

		caffe_gpu_set(featmap_dim, Dtype(0.), Y_0_.mutable_gpu_data());
		conv_x_layer_->Forward(conv_x_btm_vec_, conv_x_top_vec_);
		
		for (int t = 0; t < seq_len_; ++t)
		{
			Dtype* H_t = H_t_.mutable_gpu_data() + H_t_.offset(t);
			Dtype* Y_t_1 = t == 0 ? Y_0_.mutable_gpu_data() : top_data + top[0]->offset(t - 1);
			Dtype* Y_t = top_data + top[0]->offset(t);
			const Dtype* x_conv_t = conv_x_top_blob_.gpu_data() + conv_x_top_blob_.offset(t);
			conv_h_btm_blob_.data()->set_gpu_data(Y_t_1);
			conv_h_layer_->Forward(conv_h_btm_vec_, conv_h_top_vec_);
			const Dtype* h_t_data = conv_h_top_blob_.gpu_data();

			ActivationForward<Dtype> << <CAFFE_GET_BLOCKS(featmap_dim), CAFFE_CUDA_NUM_THREADS >> >
				(featmap_dim, act_type, x_conv_t, h_t_data, H_t);
			CUDA_POST_KERNEL_CHECK;

			conv_y_btm_blob_.data()->set_gpu_data(H_t);
			conv_y_layer_->Forward(conv_y_btm_vec_, conv_y_top_vec_);
			const Dtype* conv_y_top_data = conv_y_top_blob_.gpu_data();

			ActivationForward<Dtype> << <CAFFE_GET_BLOCKS(featmap_dim), CAFFE_CUDA_NUM_THREADS >> >
				(featmap_dim, act_type, conv_y_top_data, Y_t);
			CUDA_POST_KERNEL_CHECK;
		}
	}

	template <typename Dtype>
	void ConvRNNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype* top_diff = top[0]->mutable_gpu_diff();
		Dtype* top_data = top[0]->mutable_gpu_data();
		int featmap_dim = spatial_dims_ * num_output_;
		const int act_type = act_func_->act_typeid();

		caffe_gpu_set(Y_0_.count(0), Dtype(0.), Y_0_.mutable_gpu_diff());

		for (int t = seq_len_ - 1; t >= 0; --t)
		{
			Dtype* Y_t_data = top_data + top[0]->offset(t);
			Dtype* Y_t_diff = top_diff + top[0]->offset(t);
			Dtype* H_t_data = H_t_.mutable_gpu_data() + H_t_.offset(t);
			// diff: Y[t] -> H[t]
			Dtype* conv_y_top_diff = conv_y_top_blob_.mutable_gpu_diff();

			ActivationBackward<Dtype> << <CAFFE_GET_BLOCKS(featmap_dim), CAFFE_CUDA_NUM_THREADS >> >
				(featmap_dim, act_type, Y_t_data, Y_t_diff, conv_y_top_diff);
			CUDA_POST_KERNEL_CHECK;

			conv_y_btm_blob_.data()->set_gpu_data(H_t_data);
			conv_y_layer_->Backward(conv_y_top_vec_, vector<bool>{true}, conv_y_btm_vec_);
			Dtype* H_t_diff = conv_y_btm_blob_.mutable_gpu_diff();

			// diff: H[t] -> X[t], Y[t-1]
			Dtype* X_t_diff = conv_x_top_blob_.mutable_gpu_diff() + conv_x_top_blob_.offset(t);
			Dtype* H_t_1_top_diff = conv_h_top_blob_.mutable_gpu_diff();

			ActivationBackward<Dtype> << <CAFFE_GET_BLOCKS(featmap_dim), CAFFE_CUDA_NUM_THREADS >> >
				(featmap_dim, act_type, H_t_data, H_t_diff, X_t_diff, H_t_1_top_diff);
			CUDA_POST_KERNEL_CHECK;

			Dtype* Y_t_1_diff = t == 0 ? Y_0_.mutable_gpu_diff() : top_diff + top[0]->offset(t - 1);
			Dtype* Y_t_1_data = t == 0 ? Y_0_.mutable_gpu_data() : top_data + top[0]->offset(t - 1);
			conv_h_btm_blob_.data()->set_gpu_data(Y_t_1_data);
			conv_h_layer_->Backward(conv_h_top_vec_, vector<bool>{true}, conv_h_btm_vec_);
			caffe_gpu_add(conv_h_btm_blob_.count(0), conv_h_btm_blob_.gpu_diff(), Y_t_1_diff, Y_t_1_diff);

		}
		conv_x_layer_->Backward(conv_x_top_vec_, vector<bool>{propagate_down[0]}, conv_x_btm_vec_);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConvRNNLayer);

}  // namespace caffe