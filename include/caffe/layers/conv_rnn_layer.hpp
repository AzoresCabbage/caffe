/*
*   Convolutional Recurrent Neural Network implementation 
*   Written by Yujie Wang
*   Date: 2016.12.15
*/

#ifndef CAFFE_CONV_RNN_LAYER_HPP_
#define CAFFE_CONV_RNN_LAYER_HPP_

#include <vector>
#include <functional>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

	template <typename Dtype>
	class Activation {
	public:
		virtual Dtype act(Dtype) = 0;
		virtual Dtype d_act(Dtype) = 0;
		virtual int act_typeid() = 0;
	};

	template <typename Dtype>
	class Sigmoid : public Activation <Dtype> {
	public:
		Dtype act(Dtype x) { 
			return Dtype(1.) / (Dtype(1.) + exp(-x)); 
		}
		Dtype d_act(Dtype x) { 
			return x * (Dtype(1) - x); 
		}
		int act_typeid() { return 0; }
	};

	template <typename Dtype>
	class Tanh : public Activation <Dtype> {
	public:
		Dtype act(Dtype x) { 
			return tanh(x);
		}
		Dtype d_act(Dtype x) { 
			return Dtype(1) - x * x;
		}
		int act_typeid() { return 1; }
	};

	template <typename Dtype>
	class Relu : public Activation <Dtype> {
	public:
		Dtype act(Dtype x) { 
			return std::max(x, Dtype(0)); 
		}
		Dtype d_act(Dtype x) { 
			return x > 0 ? Dtype(1) : Dtype(0); 
		}
		int act_typeid() { return 2; }
	};
	/**
	* @brief Convolutional Recurrent Neural Network(ConvRNN) layer.
	* Formula:
	*	H[t] = f(U*X[t] + W*H[t-1] + bu)
	*
	*	*  is the convolution operation
	*   f  is the activation function
	*	X[t] is the input feature map in time t
	*   H[t] is the hidden state
	*   U and W is convolution kernel
	*   bu is bias
	*/

	//TODO(Yujie) : thorough documentation for Forward, Backward, and proto params.
	//TODO(Yujie) : support multi sequence in a single mini batch
	template <typename Dtype>
	class ConvRNNLayer : public Layer<Dtype> {
	public:
		explicit ConvRNNLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ConvRNN"; }

		virtual inline bool AllowForceBackward(const int bottom_index) const {
			return true;
		}

	protected:
		virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
		virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
			const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

		int seq_len_;				// number units of single sequence
		int seq_num_;				// number sequences
		int num_output_;			// number of output channels
		int spatial_dims_;			// Height x Width
		bool channelwise_conv_;     // operate channel wise conv or not
		int act_type_;				// activation type
		bool is_warping_;			// operate warping hidden state or not

		shared_ptr<Activation<Dtype>> act_func_;

		shared_ptr<ConvolutionLayer<Dtype>> conv_x_layer_; // conv layer for input X with bias term as bi, bf, bc, bo
		Blob<Dtype> conv_x_top_blob_; // values that after convolution operation in X[1-t]
		vector<Blob<Dtype>*> conv_x_btm_vec_; // X[1-t]: bottom[0] i.e. input feature map of entire seq
		vector<Blob<Dtype>*> conv_x_top_vec_;

		shared_ptr<ConvolutionLayer<Dtype>> conv_h_layer_; //conv layer for hidden state without bias term
		Blob<Dtype> conv_h_btm_blob_; // H[t-1] blob, its value can be fetched from top[0]
		Blob<Dtype> conv_h_top_blob_; // values that after convolution operation in H[t-1]
		vector<Blob<Dtype>*> conv_h_btm_vec_; // H[t-1] blob vector
		vector<Blob<Dtype>*> conv_h_top_vec_;

		Blob<Dtype> H_0_; // init hidden activation value, 1xCxHxW
	};

}  // namespace caffe

#endif  // CAFFE_CONV_RNN_LAYER_HPP_