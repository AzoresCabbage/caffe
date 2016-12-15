#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/conv_rnn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

	template <typename TypeParam>
	class ConvRNNLayerTest : public CPUDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
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

	TYPED_TEST_CASE(ConvRNNLayerTest, TestDtypesAndDevices);

	TYPED_TEST(ConvRNNLayerTest, TestSigmoidGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(1);
		// conv_param->add_pad(1);

		ConvRNNParameter* conv_rnn_param = layer_param.mutable_conv_rnn_param();
		conv_rnn_param->set_channelwise(false);
		conv_rnn_param->mutable_x_weight_filler()->set_type("msra");
		conv_rnn_param->mutable_x_bias_filler()->set_type("msra");
		conv_rnn_param->mutable_h_weight_filler()->set_type("msra");
		conv_rnn_param->mutable_h_bias_filler()->set_type("msra");
		conv_rnn_param->set_num_output(3);
		conv_rnn_param->set_act_type(conv_rnn_param->DEFAULT);
		conv_rnn_param->set_warping(false);

		this->blob_bottom_vec_.clear();
		this->blob_bottom_vec_.push_back(this->blob_bottom_);
		ConvRNNLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_, 0);
	}

	TYPED_TEST(ConvRNNLayerTest, TestTanhGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(1);
		// conv_param->add_pad(1);

		ConvRNNParameter* conv_rnn_param = layer_param.mutable_conv_rnn_param();
		conv_rnn_param->set_channelwise(false);
		conv_rnn_param->mutable_x_weight_filler()->set_type("msra");
		conv_rnn_param->mutable_x_bias_filler()->set_type("msra");
		conv_rnn_param->mutable_h_weight_filler()->set_type("msra");
		conv_rnn_param->mutable_h_bias_filler()->set_type("msra");
		conv_rnn_param->set_num_output(3);
		conv_rnn_param->set_act_type(conv_rnn_param->TANH);
		conv_rnn_param->set_warping(false);

		this->blob_bottom_vec_.clear();
		this->blob_bottom_vec_.push_back(this->blob_bottom_);
		ConvRNNLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_, 0);
	}

	TYPED_TEST(ConvRNNLayerTest, TestChannelwiseGradient) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(1);
		// conv_param->add_pad(1);

		ConvRNNParameter* conv_rnn_param = layer_param.mutable_conv_rnn_param();
		conv_rnn_param->set_channelwise(true);
		conv_rnn_param->mutable_x_weight_filler()->set_type("msra");
		conv_rnn_param->mutable_x_bias_filler()->set_type("msra");
		conv_rnn_param->mutable_h_weight_filler()->set_type("msra");
		conv_rnn_param->mutable_h_bias_filler()->set_type("msra");
		conv_rnn_param->set_num_output(3);
		conv_rnn_param->set_act_type(conv_rnn_param->DEFAULT);
		conv_rnn_param->set_warping(false);

		this->blob_bottom_vec_.clear();
		this->blob_bottom_vec_.push_back(this->blob_bottom_);
		ConvRNNLayer<Dtype> layer(layer_param);
		GradientChecker<Dtype> checker(1e-2, 1e-3);
		checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
			this->blob_top_vec_, 0);
	}

	TYPED_TEST(ConvRNNLayerTest, TestChannelwise) {
		typedef typename TypeParam::Dtype Dtype;
		LayerParameter layer_param;
		ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
		conv_param->add_kernel_size(1);

		ConvRNNParameter* conv_rnn_param = layer_param.mutable_conv_rnn_param();
		conv_rnn_param->set_channelwise(true);
		conv_rnn_param->mutable_x_weight_filler()->set_type("constant");
		conv_rnn_param->mutable_x_weight_filler()->set_value(1);
		conv_rnn_param->mutable_x_bias_filler()->set_type("constant");
		conv_rnn_param->mutable_x_bias_filler()->set_value(0);
		conv_rnn_param->mutable_h_weight_filler()->set_type("constant");
		conv_rnn_param->mutable_h_weight_filler()->set_value(1);
		conv_rnn_param->mutable_h_bias_filler()->set_type("constant");
		conv_rnn_param->mutable_h_bias_filler()->set_value(0);
		conv_rnn_param->set_num_output(3);
		conv_rnn_param->set_act_type(conv_rnn_param->RELU);
		conv_rnn_param->set_warping(false);

		Dtype* btm_data = this->blob_bottom_->mutable_cpu_data();
		for (int i = 0; i < this->blob_bottom_->num(); ++i)
		{
			for (int j = 0; j < this->blob_bottom_->count(1); ++j)
			{
				btm_data[i*this->blob_bottom_->count(1) + j] = i + 1;
			}
		}

		this->blob_bottom_vec_.clear();
		this->blob_bottom_vec_.push_back(this->blob_bottom_);
		ConvRNNLayer<Dtype> layer(layer_param);
		layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		Dtype sum = 0;
		Dtype* top_data = this->blob_top_->mutable_cpu_data();
		for (int i = this->blob_top_->count() - this->blob_top_->count(1); i < this->blob_top_->count(); ++i)
		{
			sum += abs(3 - top_data[i]);
		}
		std::cout << sum << std::endl;

		conv_rnn_param->set_channelwise(false);
		ConvRNNLayer<Dtype> layer1(layer_param);
		layer1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
		layer1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
		sum = 0;
		top_data = this->blob_top_->mutable_cpu_data();
		for (int i = this->blob_top_->count() - this->blob_top_->count(1); i < this->blob_top_->count(); ++i)
		{
			sum += abs(15 - top_data[i]);
		}
		std::cout << sum << std::endl;
	}
}  // namespace caffe