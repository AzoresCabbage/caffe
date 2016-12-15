#ifndef CAFFE_CONV_LSTM_LAYER_HPP_
#define CAFFE_CONV_LSTM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

	/**
	* @brief Convolutional Long Short Term Memory(ConvLSTM) layer.
	* Formula:
	*	I[t] = sigmoid(Wxi*X[t] + Whi*H[t-1] + Wci.*C[t-1] + bi)
	*	F[t] = sigmoid(Wxf*X[t] + Whf*H[t-1] + Wcf.*C[t-1] + bf)
	*	C[t] = F[t].*C[t-1] + I[t].*tanh(Wxc*X[t] + Whc*H[t-1] + bc)
	*	O[t] = sigmoid(Wxo*X[t] + Who*H[t-1] + Wco.*C[t] + bo)
	*	H[t] = O[t] * tanh(C[t])
	*
	*	*  means convolution operation
	*   .* means Hadamard procuct
	*	X[t] is the input feature map in time t
	*	I[t], F[t], C[t], O[t] is the input gate, forget gate, memory cell, output gate at time t seperatly
	* source: <Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting>
	*/

	//TODO(Yujie) : thorough documentation for Forward, Backward, and proto params.
	//TODO(Yujie) : support multi sequence in a single mini batch
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
		int spatial_dims_; // Height x Width

		Blob<Dtype> H_0_; // init hidden activation value, 1xCxHxW
		Blob<Dtype> C_0_; // init memory cell activation value , 1xCxHxW

		shared_ptr<ConvolutionLayer<Dtype>> conv_x_layer_; // conv layer for input X with bias term as bi, bf, bc, bo
		Blob<Dtype> conv_x_top_blob_; // values that after convolution operation in X[1-t]
		vector<Blob<Dtype>*> conv_x_btm_vec_; // X[1-t]: bottom[0] i.e. input feature map of entire seq
		vector<Blob<Dtype>*> conv_x_top_vec_;

		shared_ptr<ConvolutionLayer<Dtype>> conv_h_layer_; //conv layer for hidden state without bias term
		Blob<Dtype> conv_h_btm_blob_; // H[t-1] blob, its value can be fetched from top[0]
		Blob<Dtype> conv_h_top_blob_; // values that after convolution operation in H[t-1]
		vector<Blob<Dtype>*> conv_h_btm_vec_; // H[t-1] blob vector
		vector<Blob<Dtype>*> conv_h_top_vec_;

		Blob<Dtype> conv_h_top_t_;

		// matrix for eltwise multiply with memory cell C[t-1], shape: 1xCxHxW
		Blob<Dtype> Wc_;
		Blob<Dtype> Wci_c_t_1_;
		Blob<Dtype> Wcf_c_t_1_;

		Blob<Dtype> wo_ct_tmp_; // tmp value for Wco .* C[t], shape: NxCxHxW

		// intermediate variable that before activation, shape: NxCxHxW
		Blob<Dtype> gate_i_;
		Blob<Dtype> gate_f_;
		Blob<Dtype> gate_c_;
		Blob<Dtype> gate_o_;
	};

}  // namespace caffe

#endif  // CAFFE_CONV_LSTM_LAYER_HPP_