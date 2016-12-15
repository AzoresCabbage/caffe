#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/conv_rnn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename Dtype>
	class ConvRNNLayerTest : public GPUDeviceTest<Dtype> {
	protected:
		ConvRNNLayerTest()
			: blob_bottom_(new Blob<Dtype>(vector<int>{ 2, 3, 2, 2 })),
			blob_top_(new Blob<Dtype>()) {
			//// fill the values
			FillerParameter filler_param;
			filler_param.set_min(-0.1);
			filler_param.set_max(0.1);
			UniformFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~ConvRNNLayerTest() { delete blob_bottom_; delete blob_top_; }
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(ConvRNNLayerTest, TestDtypes);

	TYPED_TEST(ConvRNNLayerTest, TestWarpingGredient) {
		LayerParameter layer_param;
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(1);

		ConvRNNParameter* conv_rnn_param = layer_param.mutable_conv_rnn_param();
		conv_rnn_param->set_channelwise(false);
		conv_rnn_param->mutable_x_weight_filler()->set_type("msra");
		conv_rnn_param->mutable_x_bias_filler()->set_type("msra");
		conv_rnn_param->mutable_h_weight_filler()->set_type("msra");
		conv_rnn_param->mutable_h_bias_filler()->set_type("msra");
		conv_rnn_param->set_num_output(3);
		conv_rnn_param->set_act_type(conv_rnn_param->RELU);
		conv_rnn_param->set_warping(true);
		Blob<Dtype> flow(vector<int>{1, 2, 2, 2});
		FillerParameter filler_param;
		filler_param.set_min(-0.1);
		filler_param.set_max(0.1);
		UniformFiller<Dtype> filler(filler_param);
		filler.Fill(&flow);

		this->blob_bottom_vec_.clear();
		this->blob_bottom_vec_.push_back(this->blob_bottom_);
		this->blob_bottom_vec_.push_back(&flow);

		ConvRNNLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_, 0);
	}
}  // namespace caffe