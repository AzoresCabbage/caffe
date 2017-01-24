//#include <vector>
//#include <algorithm>
//#include <cmath>
//
//#include "caffe/layers/conv_gru_layer.hpp"
//#include "caffe/util/math_functions.hpp"
//
//namespace caffe {
//
//template <typename Dtype>
//__device__ Dtype sigmoid(const Dtype x) {
//	return Dtype(1) / (Dtype(1) + exp(-x));
//}
//
//template <typename Dtype>
//__device__ Dtype d_sigmoid(const Dtype x) {
//	return x * (1 - x);
//}
//
//template <typename Dtype>
//__device__ Dtype relu(const Dtype x) {
//	return max(x, Dtype(0));
//}
//
//template <typename Dtype>
//__device__ Dtype d_relu(const Dtype x) {
//	return x > 0 ? 1 : 0;
//}
//
//template <typename Dtype>
//__device__ Dtype tanh(const Dtype x) {
//	return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
//}
//
//template <typename Dtype>
//__device__ Dtype d_tanh(const Dtype x) {
//	return 1 - x * x;
//}
//
//template <typename Dtype>
//__device__ Dtype hard_sigmoid(const Dtype x) {
//	return max(min(0.2 * x + 0.5, Dtype(1)), Dtype(0));
//}
//
//template <typename Dtype>
//__device__ Dtype d_hard_sigmoid(const Dtype x) {
//	if (x >= 1 || x <= 0)
//		return 0;
//	else
//		return 0.2;
//}
//
//template <typename Dtype>
//__global__ void TanHForward(const int nthreads, Dtype* data) {
//	CUDA_KERNEL_LOOP(index, nthreads) {
//		data[index] = tanh(data[index]);
//	}
//}
//
//template <typename Dtype>
//__global__ void TanHBackward(const int nthreads, Dtype* diff, Dtype* data) {
//	CUDA_KERNEL_LOOP(index, nthreads) {
//		diff[index] = diff[index] * d_tanh(data[index]);
//	}
//}
//
////nthreads : N * H ; N is seq num
////H : feature_dims = Channel*Height*Width
////hidden_pre_gate: the data after hidden conv, 2 types (Ur, Uz) for C*H*W
////input_pre_gate : the data after input conv. 3 types (Wr, Wz, W) for C*H*W
//template <typename Dtype>
//__global__ void SigmoidForward(const int nthreads, const int H, Dtype* hidden_pre_gate,
//	Dtype* input_pre_gate, Dtype* h_t_1, Dtype* hidden_reset_) {
//	CUDA_KERNEL_LOOP(index, nthreads) {
//        const int n = index / H;
//		const int d = index % H;
//
//		// Rt = sigmoid(Wr*Xt + Ur*H[t-1])
//		input_pre_gate[n * 3 * H + d] += hidden_pre_gate[n * 2 * H + d];
//		input_pre_gate[n * 3 * H + d] = sigmoid(input_pre_gate[n * 3 * H + d]);
//
//		// Zt = sigmoid(Wz*Xt + Uz*H[t-1])
//        input_pre_gate[n * 3 * H + H + d] += hidden_pre_gate[n * 2 * H + H + d];
//        input_pre_gate[n * 3 * H + H + d] = sigmoid(input_pre_gate[n * 3 * H + H + d]);
//
//		// Rt .* H[t-1] for before Ht_candidate conv
//        hidden_reset_[index] = input_pre_gate[n * 3 * H + d] * h_t_1[index];
//	}
//}
//
//template <typename Dtype>
//__global__ void ActivationForward(const int nthreads, const int H, Dtype* hidden_rt_pre_gate,
//	Dtype* input_pre_gate, Dtype* h_t_1, Dtype* h_t) {
//	CUDA_KERNEL_LOOP(index, nthreads) {
//        const int n = index / H;
//		const int d = index % H;
//        input_pre_gate[n * 3 * H + 2 * H + d] += hidden_rt_pre_gate[index];
//		
//		// Yujie: Why use relu here ?
//        //input_pre_gate[n * 3 * H + 2 * H + d] = relu(input_pre_gate[n * 3 * H + 2 * H + d]);
//		input_pre_gate[n * 3 * H + 2 * H + d] = tanh(input_pre_gate[n * 3 * H + 2 * H + d]);
//
//        Dtype z_t = input_pre_gate[n * 3 * H + H + d];
//		h_t[index] = (1 - z_t) * h_t_1[index] + z_t * input_pre_gate[n * 3 * H + 2 * H + d];
//	}
//}
//
//template <typename Dtype>
//__global__ void ActivationBackward(const int nthreads, const int H,
//	const Dtype* gate, Dtype* pre_gate_diff, Dtype* hidden_reset_,
//	const Dtype* h_t_1, Dtype* dh_t_1, const Dtype* dh_t,
//    Dtype* hidden_rt_pre_gate_diff) {
//	CUDA_KERNEL_LOOP(index, nthreads) {
//        const int n = index / H;
//		const int d = index % H;
//		
//		dh_t_1[index] += dh_t[index] * (1 - gate[n * 3 * H + H + d]);
//
//		pre_gate_diff[n * 3 * H + 2 * H + d] = dh_t[index] * gate[n * 3 * H + H + d];
//		// Yujie : Why use relu here?
//		// pre_gate_diff[n * 3 * H + 2 * H + d] *= d_relu(gate[n * 3 * H + 2 * H + d]);
//		pre_gate_diff[n * 3 * H + 2 * H + d] *= d_tanh(gate[n * 3 * H + 2 * H + d]);
//
//		pre_gate_diff[n * 3 * H + H + d] = dh_t[index] * (gate[n * 3 * H + 2 * H + d] - h_t_1[index]);
//		pre_gate_diff[n * 3 * H + H + d] *= d_sigmoid(gate[n * 3 * H + H + d]);
//
//	    hidden_reset_[index] = gate[n * 3 * H + d] * h_t_1[index];
//
//        hidden_rt_pre_gate_diff[index] = pre_gate_diff[n * 3 * H + 2 * H + d];
//    }
//}
//
//template <typename Dtype>
//__global__ void SigmoidBackward(const int nthreads, const int H,
//	const Dtype* gate, Dtype* pre_gate_diff, const Dtype* h_t_1,
//    Dtype* dh_t_1, const Dtype* hidden_rt_diff, Dtype* hidden_pre_gate_diff) {
//	CUDA_KERNEL_LOOP(index, nthreads) {
//        const int n = index / H;
//		const int d = index % H;
//		
//        dh_t_1[index] += hidden_rt_diff[index] * gate[n * 3 * H + d];
//        pre_gate_diff[n * 3 * H + d] = hidden_rt_diff[index] * h_t_1[index];
//        pre_gate_diff[n * 3 * H + d] *= d_sigmoid(gate[n * 3 * H + d]);
//
//        hidden_pre_gate_diff[n * 2 * H + d] = pre_gate_diff[n * 3 * H + d];
//        hidden_pre_gate_diff[n * 2 * H + H + d] = pre_gate_diff[n * 3 * H + H + d];
//    }
//}
//
//template <typename Dtype>
//void ConvGRULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
//	const vector<Blob<Dtype>*>& top) {
//	Dtype* top_data = top[0]->mutable_gpu_data();
//
//	Dtype* input_pre_gate_data = input_pre_gate_.mutable_gpu_data();
//	Dtype* hidden_pre_gate_data = hidden_pre_gate_.mutable_gpu_data();
//	Dtype* hidden_rt_data = hidden_reset_.mutable_gpu_data();
//	Dtype* hidden_rt_pre_gate_data = hidden_rt_pre_gate_.mutable_gpu_data();
//	int feature_dims = H_ * spatial_dims;
//
//
//
//	// Compute input to gate forward propagation
//	conv_input_layer_->Forward(conv_input_bottom_vec_, conv_input_top_vec_);
//
//	// Initialize previous state
//	if (bottom.size() == 2)
//	{
//		h_0_.ShareData(*(bottom[1]));
//		h_0_.ShareDiff(*(bottom[1]));
//	}
//	else
//	{
//		// caffe_gpu_set(h_0_.count(0), Dtype(0.), h_0_.mutable_gpu_data());
//		// set H0 as W*X[t]
//		caffe_copy(h_0_.count(0), input_pre_gate_data + 2 * feature_dims, h_0_.mutable_gpu_data());
//		TanHForward<Dtype> << <CAFFE_GET_BLOCKS(N_ * feature_dims), CAFFE_CUDA_NUM_THREADS >> >(h_0_.count(0), h_0_.mutable_gpu_data());
//		CUDA_POST_KERNEL_CHECK;
//	}
//
//	// Compute recurrent forward propagation
//	for (int tt = 0; tt < T_; ++tt) {
//		int t = tt;
//		if (!forward_direction_) t = T_ - tt - 1;
//
//		Dtype* h_t = top_data + top[0]->count(1) * t;
//		Dtype* input_pre_gate_t = input_pre_gate_data + input_pre_gate_.count(1) * t;
//
//		Dtype* h_t_1 = t > 0 ? (h_t - top[0]->count(1)) : h_0_.mutable_gpu_data();
//
//		if (!forward_direction_){
//			h_t_1 = t < T_ - 1 ? (h_t + top[0]->count(1)) : h_0_.mutable_gpu_data();
//		}
//
//		// Hidden-to-hidden propagation
//		hidden_.data()->set_gpu_data(h_t_1);
//		conv_hidden_layer_->Forward(conv_hidden_bottom_vec_, conv_hidden_top_vec_);
//
//        SigmoidForward<Dtype><<<CAFFE_GET_BLOCKS(N_ * feature_dims), CAFFE_CUDA_NUM_THREADS>>>(N_ * feature_dims, feature_dims,
//	        hidden_pre_gate_data, input_pre_gate_t, h_t_1, hidden_rt_data);
//		CUDA_POST_KERNEL_CHECK;
//
//        conv_tmp_hidden_layer_->Forward(conv_tmp_hidden_bottom_vec_, conv_tmp_hidden_top_vec_);
//
//        ActivationForward<Dtype><<<CAFFE_GET_BLOCKS(N_ * feature_dims), CAFFE_CUDA_NUM_THREADS>>>(N_ * feature_dims, feature_dims,
//	        hidden_rt_pre_gate_data, input_pre_gate_t, h_t_1, h_t);
//		CUDA_POST_KERNEL_CHECK;
//		if (top.size() > 1)
//		{
//			caffe_gpu_memcpy(input_pre_gate_.data()->size(), input_pre_gate_data, top[1]->mutable_gpu_data());
//		}
//	}
//}
//
//template <typename Dtype>
//void ConvGRULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
//	const vector<bool>& propagate_down,
//	const vector<Blob<Dtype>*>& bottom) {
//	const Dtype* gate_data = input_pre_gate_.gpu_data();
//
//	Dtype* top_diff = top[0]->mutable_gpu_diff();
//	Dtype* pre_gate_diff = input_pre_gate_.mutable_gpu_diff();
//	Dtype* hidden_pre_gate_diff = hidden_pre_gate_.mutable_gpu_diff();
//	Dtype* hidden_rt_data = hidden_reset_.mutable_gpu_data();
//	Dtype* hidden_rt_pre_gate_diff = hidden_rt_pre_gate_.mutable_gpu_diff();
//	const Dtype* hidden_rt_diff = hidden_reset_.mutable_gpu_diff();
//	caffe_gpu_set(h_0_.count(0), Dtype(0.), h_0_.mutable_gpu_diff());
//
//	int feature_dims = H_ * spatial_dims;
//
//	for (int tt = T_ - 1; tt >= 0; --tt) {
//		int t = tt;
//		if (!forward_direction_) t = T_ - tt - 1;
//
//		Dtype* dh_t = top_diff + top[0]->count(1) * t;
//		Dtype* pre_gate_diff_t = pre_gate_diff + input_pre_gate_.count(1) * t;
//		const Dtype* gate_t = gate_data + input_pre_gate_.count(1) * t;
//
//		Dtype* dh_t_1 = t > 0 ? top_diff + top[0]->count(1) * (t - 1) : h_0_.mutable_gpu_diff();
//		Dtype* h_t_1 = t > 0 ? (top[0]->mutable_gpu_data() + top[0]->count(1) * (t - 1)) : h_0_.mutable_gpu_data();
//		if (!forward_direction_){
//			dh_t_1 = t < T_ - 1 ? top_diff + top[0]->count(1) * (t + 1) : h_0_.mutable_gpu_diff();
//			h_t_1 = t < T_ - 1 ? (top[0]->mutable_gpu_data() + top[0]->count(1) * (t + 1)) : h_0_.mutable_gpu_data();
//		}
//
//        ActivationBackward<Dtype><<<CAFFE_GET_BLOCKS(N_ * feature_dims), CAFFE_CUDA_NUM_THREADS>>>(
//            N_ * feature_dims, feature_dims, gate_t, pre_gate_diff_t, hidden_rt_data,
//            h_t_1, dh_t_1, dh_t, hidden_rt_pre_gate_diff);
//        CUDA_POST_KERNEL_CHECK;
//
//        conv_tmp_hidden_layer_->Backward(conv_tmp_hidden_top_vec_, vector<bool>{true}, conv_tmp_hidden_bottom_vec_);
//
//        SigmoidBackward<Dtype><<<CAFFE_GET_BLOCKS(N_ * feature_dims), CAFFE_CUDA_NUM_THREADS>>>
//        (N_ * feature_dims, feature_dims, gate_t, pre_gate_diff_t, h_t_1, dh_t_1, hidden_rt_diff, hidden_pre_gate_diff);
//        CUDA_POST_KERNEL_CHECK;
//
//		// Backprop errors to the previous time step
//		hidden_.data()->set_gpu_data(h_t_1);
//		conv_hidden_layer_->Backward(conv_hidden_top_vec_, vector<bool>{true}, conv_hidden_bottom_vec_);
//		const Dtype* hidden_diff_ = hidden_.gpu_diff();
//		caffe_gpu_add<Dtype>(N_ * feature_dims, dh_t_1, hidden_diff_, dh_t_1);
//	}
//	if (bottom.size() == 1)
//	{
//		TanHBackward<Dtype> << <CAFFE_GET_BLOCKS(N_ * feature_dims), CAFFE_CUDA_NUM_THREADS >> >(h_0_.count(0), h_0_.mutable_gpu_diff(), h_0_.mutable_gpu_data());
//		CUDA_POST_KERNEL_CHECK;
//		caffe_gpu_add<Dtype>(h_0_.count(0), h_0_.mutable_gpu_diff(), pre_gate_diff + 2 * feature_dims, pre_gate_diff + 2 * feature_dims);
//	}
//	// Gradient w.r.t. bottom data
//	conv_input_layer_->Backward(conv_input_top_vec_, vector<bool>{propagate_down[0]}, conv_input_bottom_vec_);
//
//}
//
//INSTANTIATE_LAYER_GPU_FUNCS(ConvGRULayer);
//
//}  // namespace caffe