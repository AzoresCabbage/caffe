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
		Dtype* top_data_t, Dtype* h_diff_t, Dtype* x_conv_diff_t, Dtype* h_conv_diff) {
		CUDA_KERNEL_LOOP(i, nthreads) {
			x_conv_diff_t[i] = h_diff_t[i] * 
				(act_type == 1 ? d_Tanh_gpu(top_data_t[i]) 
				: act_type == 2 ? d_Relu_gpu(top_data_t[i]) : d_Sigmoid_gpu(top_data_t[i]));
			h_conv_diff[i] = x_conv_diff_t[i];
		}
	}


	template <typename Dtype>
	void ConvRNNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_gpu_data();
		int featmap_dim = spatial_dims_ * num_output_;
		const int act_type = act_func_->act_typeid();

		caffe_gpu_set(featmap_dim, Dtype(0.), H_0_.mutable_gpu_data());
		conv_x_layer_->Forward(conv_x_btm_vec_, conv_x_top_vec_);
		
		for (int t = 0; t < seq_len_; ++t)
		{
			Dtype* H_t = top_data + top[0]->offset(t);
			Dtype* H_t_1 = t == 0 ? H_0_.mutable_gpu_data() : top_data + top[0]->offset(t - 1);
			const Dtype* x_conv_t = conv_x_top_blob_.gpu_data() + conv_x_top_blob_.offset(t);
			const Dtype* h_conv_t = conv_h_top_blob_.gpu_data();
			conv_h_btm_blob_.data()->set_gpu_data(H_t_1);
			conv_h_layer_->Forward(conv_h_btm_vec_, conv_h_top_vec_);

			ActivationForward<Dtype> << <CAFFE_GET_BLOCKS(featmap_dim), CAFFE_CUDA_NUM_THREADS >> >
				(featmap_dim, act_type, x_conv_t, h_conv_t, H_t);
			CUDA_POST_KERNEL_CHECK;

			if (is_warping_)
			{
				Dtype* warp_flow = t == 0 ? warp_0_.mutable_gpu_data() : bottom[1]->mutable_gpu_data() + bottom[1]->offset(t - 1);
				warp_btm_blob_data_.data()->set_gpu_data(H_t);
				warp_btm_blob_flow_.data()->set_gpu_data(warp_flow);
				warping_layer_->Forward(warp_btm_vec_, warp_top_vec_);
				caffe_gpu_memcpy(featmap_dim, warp_top_blob_.gpu_data(), H_t);
			}
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

		caffe_gpu_set(H_0_.count(), Dtype(0.), H_0_.mutable_gpu_diff());

		for (int t = seq_len_ - 1; t >= 0; --t)
		{
			Dtype* top_data_t = top_data + top[0]->offset(t);
			Dtype* h_data_t_1 = t == 0 ? H_0_.mutable_gpu_data() : top_data + top[0]->offset(t - 1);

			Dtype* h_diff_t = top_diff + top[0]->offset(t);
			Dtype* h_diff_t_1 = t == 0 ? H_0_.mutable_gpu_diff() : top_diff + top[0]->offset(t - 1);
			Dtype* h_conv_diff = conv_h_top_blob_.mutable_gpu_diff();
			Dtype* x_conv_diff_t = conv_x_top_blob_.mutable_gpu_diff() + conv_x_top_blob_.offset(t);
			
			if (is_warping_)
			{
				Dtype* warp_flow = t == 0 ? warp_0_.mutable_gpu_data() : bottom[1]->mutable_gpu_data() + bottom[1]->offset(t - 1);
				warp_btm_blob_data_.data()->set_gpu_data(h_data_t_1);
				warp_btm_blob_flow_.data()->set_gpu_data(warp_flow);
				caffe_gpu_memcpy(featmap_dim, h_diff_t, warp_top_blob_.mutable_gpu_diff());
				warping_layer_->Backward(warp_top_vec_, vector<bool>{true, true}, warp_btm_vec_);
				caffe_gpu_memcpy(featmap_dim, warp_btm_blob_data_.gpu_diff(), h_diff_t);
			}

			ActivationBackward<Dtype> << <CAFFE_GET_BLOCKS(featmap_dim), CAFFE_CUDA_NUM_THREADS >> >
				(featmap_dim, act_type, top_data_t, h_diff_t, x_conv_diff_t, h_conv_diff);
			CUDA_POST_KERNEL_CHECK;

			conv_h_btm_blob_.data()->set_gpu_data(h_data_t_1);
			conv_h_layer_->Backward(conv_h_top_vec_, vector<bool>{true}, conv_h_btm_vec_);
			caffe_gpu_add(featmap_dim, h_diff_t_1, conv_h_btm_blob_.gpu_diff(), h_diff_t_1);
		}
		conv_x_layer_->Backward(conv_x_top_vec_, vector<bool>{propagate_down[0]}, conv_x_btm_vec_);
	}

	INSTANTIATE_LAYER_GPU_FUNCS(ConvRNNLayer);

}  // namespace caffe