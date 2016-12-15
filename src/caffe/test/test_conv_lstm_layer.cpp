#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/filler.hpp"
#include "caffe/layers/conv_lstm_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

//#ifndef CPU_ONLY
//	extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;
//#endif

	template <typename TypeParam>
	class ConvLSTMLayerTest : public CPUDeviceTest<TypeParam> {
		typedef typename TypeParam::Dtype Dtype;
	protected:
		ConvLSTMLayerTest()
			: blob_bottom_(new Blob<Dtype>(vector<int>{ 2, 3, 2, 2 })),
			blob_top_(new Blob<Dtype>()) {
			//fill the values
			FillerParameter filler_param;
			filler_param.set_min(-0.1);
			filler_param.set_max(0.1);
			UniformFiller<Dtype> filler(filler_param);
			filler.Fill(this->blob_bottom_);
			//Dtype* pdata = this->blob_bottom_->mutable_cpu_data();
			//pdata[0] = 1;
			//pdata[1] = 1;
			//pdata[2] = 0;
			//pdata[3] = 0;
			blob_top_vec_.push_back(blob_top_);
		}
		virtual ~ConvLSTMLayerTest() { delete blob_bottom_; delete blob_top_; }
		Blob<Dtype>* const blob_bottom_;
		Blob<Dtype>* const blob_top_;
		vector<Blob<Dtype>*> blob_bottom_vec_;
		vector<Blob<Dtype>*> blob_top_vec_;
	};

	TYPED_TEST_CASE(ConvLSTMLayerTest, TestDtypesAndDevices);

	TYPED_TEST(ConvLSTMLayerTest, TestGradientDefault) {
		typedef typename TypeParam::Dtype Dtype;
		bool IS_VALID_CUDA = false;
//#ifndef CPU_ONLY
//		IS_VALID_CUDA = CAFFE_TEST_CUDA_PROP.major >= 2;
//#endif
		if (Caffe::mode() == Caffe::CPU ||
			sizeof(Dtype) == 4 || IS_VALID_CUDA) {
			LayerParameter layer_param;
			ConvolutionParameter* conv_param = layer_param.mutable_convolution_param();
			conv_param->set_num_output(2);
			conv_param->add_kernel_size(1);
			/*conv_param->mutable_weight_filler()->set_type("constant");
			conv_param->mutable_weight_filler()->set_value(0.1);*/
			conv_param->mutable_weight_filler()->set_type("uniform");
			conv_param->mutable_weight_filler()->set_min(-0.01);
			conv_param->mutable_weight_filler()->set_max(0.01);
			conv_param->mutable_bias_filler()->set_type("constant");
			conv_param->mutable_bias_filler()->set_value(0);
			//conv_param->add_pad(1);
			this->blob_bottom_vec_.clear();
			this->blob_bottom_vec_.push_back(this->blob_bottom_);
			ConvLSTMLayer<Dtype> layer(layer_param);
			GradientChecker<Dtype> checker(1e-2, 1e-3);
			checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
				this->blob_top_vec_, 0);
		}
		else {
			LOG(ERROR) << "Skipping test due to old architecture.";
		}
	}

}  // namespace caffe