#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/conv_gru_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

#ifndef CPU_ONLY
	extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
#endif

	template <typename TypeParam>
	class GRULayerTest : public MultiDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		// note that H0_ dim must be [1, num_output of gru, btm_x, btm_y]
		GRULayerTest()
			: blob_bottom_(new Blob<Dtype>(vector<int>{ 5, 3, 4, 4 })),
			H0_(new Blob<Dtype>(vector<int>{ 1, 1, 4, 4 })),
			blob_top_(new Blob<Dtype>()) {
			// fill the values
			FillerParameter filler_param;
			filler_param.set_min(-0.1);
			filler_param.set_max(0.1);
			UniformFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			blob_top_vec_.push_back(blob_top_);

			filler.Fill(H0_);
		}
		virtual ~GRULayerTest() { delete blob_bottom_; delete blob_top_; }
		Blob<Dtype>* blob_bottom_;
		Blob<Dtype>* H0_;
		Blob<Dtype>* blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(GRULayerTest, TestDtypesAndDevices);

	TYPED_TEST(GRULayerTest, TestBottom2Default) {
		typedef typename TypeParam::Dtype Dtype;
		bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
		IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
		if (Caffe::mode() == Caffe::CPU ||
			sizeof(Dtype) == 4 || IS_VALID_CUDA) {
			LayerParameter layer_param;
			ConvGRUParameter* gru_parm = layer_param.mutable_conv_gru_param();
			gru_parm->set_num_output(1);
			ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
			conv_param->add_kernel_size(3);
			conv_param->mutable_weight_filler()->set_type("uniform");
			conv_param->mutable_weight_filler()->set_min(-0.01);
			conv_param->mutable_weight_filler()->set_max(0.01);
			conv_param->mutable_bias_filler()->set_type("constant");
			conv_param->mutable_bias_filler()->set_value(0);
			conv_param->add_pad(1);
			this->blob_bottom_vec_.clear();
			this->blob_bottom_vec_.push_back(this->blob_bottom_);
			this->blob_bottom_vec_.push_back(this->H0_);
			ConvGRULayer<Dtype> layer(layer_param);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				this->blob_top_vec_, 0);
		}
		else {
			LOG(ERROR) << "Skipping test due to old architecture.";
		}
	}

	TYPED_TEST(GRULayerTest, TestBottom1Default) {
		typedef typename TypeParam::Dtype Dtype;
		bool IS_VALID_CUDA = false;
#ifndef CPU_ONLY
		IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
#endif
		if (Caffe::mode() == Caffe::CPU ||
			sizeof(Dtype) == 4 || IS_VALID_CUDA) {
			LayerParameter layer_param;
			ConvGRUParameter* gru_parm = layer_param.mutable_conv_gru_param();
			gru_parm->set_num_output(1);
			ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
			conv_param->add_kernel_size(3);
			conv_param->mutable_weight_filler()->set_type("uniform");
			conv_param->mutable_weight_filler()->set_min(-0.01);
			conv_param->mutable_weight_filler()->set_max(0.01);
			conv_param->mutable_bias_filler()->set_type("constant");
			conv_param->mutable_bias_filler()->set_value(0);
			conv_param->add_pad(1);
			this->blob_bottom_vec_.clear();
			this->blob_bottom_vec_.push_back(this->blob_bottom_);
			ConvGRULayer<Dtype> layer(layer_param);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				this->blob_top_vec_, 0);
		}
		else {
			LOG(ERROR) << "Skipping test due to old architecture.";
		}
	}
}  // namespace caffe