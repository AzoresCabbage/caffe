#ifndef CAFFE_CONV_LSTM_LAYER_HPP_
#define CAFFE_CONV_LSTM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"
#include "caffe/layers/eltwise_layer.hpp"

namespace caffe {

	/**
	* @brief Convolutional Long Short Term Memory(ConvLSTM) layer.
	* TODO(Yujie): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class ConvLSTMLayer : public Layer<Dtype> {
	public:
		explicit ConvLSTMLayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ConvLSTM"; }

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

		int seq_len_; // number units of single sequence
		int seq_num_; // number sequences
		int num_output_; // number of output channels
		int spatial_dims; // Height x Width

		Blob<Dtype> H_0_; // init hidden activation value
		Blob<Dtype> C_0_; // init memory cell activation value 

		// conv layer for input x with bias term as Bi, Bf, Bc, Bo
		shared_ptr<ConvolutionLayer<Dtype>> conv_x_layer_;
		Blob<Dtype> conv_x_top_blob_;  // gate values before nonlinearity, also is the conv_x_top_vec_ blob
		vector<Blob<Dtype>*> conv_x_bottom_vec_; // X[t]
		vector<Blob<Dtype>*> conv_x_top_vec_;

		//conv layer for input h without bias term
		shared_ptr<ConvolutionLayer<Dtype>> conv_h_layer_;
		Blob<Dtype> conv_h_top_blob_; // gate values before nonlinearity, also is the conv_hidden_top_vec_ blob
		vector<Blob<Dtype>*> conv_h_bottom_vec_; // H[t-1]
		vector<Blob<Dtype>*> conv_h_top_vec_;

		// matrix for eltwise multiply with memory cell t-1
		Blob<Dtype> Wci_;
		Blob<Dtype> Wcf_;

		// eltwise layer for calc Ct two parts
		shared_ptr<EltwiseLayer<Dtype>> c1_; // f[t] .* C[t-1]
		shared_ptr<EltwiseLayer<Dtype>> c2_; // i[t] .* tanh(Wxi*X[t] + Whi*H[t-1] + bc)
		Blob<Dtype> c1_top_blob_;
		Blob<Dtype> c2_top_blob_;

	};

}  // namespace caffe

#endif  // CAFFE_CONV_LSTM_LAYER_HPP_