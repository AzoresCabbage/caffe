#ifndef CAFFE_GRU_LAYER_HPP_
#define CAFFE_GRU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

	/**
	* @brief Convolutional Gated Recurrent Unit(GRU) layer.
	* TODO(dox): thorough documentation for Forward, Backward, and proto params.
	*/
	template <typename Dtype>
	class ConvGRULayer : public Layer<Dtype> {
	public:
		explicit ConvGRULayer(const LayerParameter& param)
			: Layer<Dtype>(param) {}
		virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
			const vector<Blob<Dtype>*>& top);

		virtual inline const char* type() const { return "ConvGRU"; }
		virtual inline int ExactNumBottomBlobs() const { return 1; }
		virtual inline int ExactNumTopBlobs() const { return 1; }

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

		int T_; // length of sequence
		int N_; // sequence num
		int H_; // num of hidden units channels
		int C_; // input channels
		int spatial_dims; // HxW
		bool forward_direction_;

		Dtype clipping_threshold_; // threshold for clipped gradient

		Blob<Dtype> h_0_; // previous hidden activation value

		// conv layers
		shared_ptr<ConvolutionLayer<Dtype>> conv_input_layer_;
		Blob<Dtype> input_pre_gate_;  // gate values before nonlinearity
		vector<Blob<Dtype>*> conv_input_bottom_vec_;
		vector<Blob<Dtype>*> conv_input_top_vec_;

		Blob<Dtype> hidden_;

		shared_ptr<ConvolutionLayer<Dtype>> conv_hidden_layer_;
		Blob<Dtype> hidden_pre_gate_;
		vector<Blob<Dtype>*> conv_hidden_bottom_vec_;
		vector<Blob<Dtype>*> conv_hidden_top_vec_;

		Blob<Dtype> hidden_reset_;

		shared_ptr<ConvolutionLayer<Dtype>> conv_tmp_hidden_layer_;
		Blob<Dtype> hidden_rt_pre_gate_;
		vector<Blob<Dtype>*> conv_tmp_hidden_bottom_vec_;
		vector<Blob<Dtype>*> conv_tmp_hidden_top_vec_;
	};

}  // namespace caffe

#endif  // CAFFE_GRU_LAYER_HPP_