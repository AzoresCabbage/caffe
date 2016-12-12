#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_lstm_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

	template <typename Dtype>
	inline Dtype sigmoid(Dtype x) {
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	inline Dtype hard_sigmoid(Dtype x) {
		return std::max<Dtype>(std::min<Dtype>(0.2 * x + 0.5, 1), 0);
	}

	template <typename Dtype>
	inline Dtype d_hard_sigmoid(Dtype x) {
		if (x >= 1 || x <= 0)
			return 0;
		else
			return 0.2;
	}

	template <typename Dtype>
	void ConvLSTMLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		// TODO(Yujie) : support multiple sequences in one mini batch

		// Currently, only support one bottom input: X[t]
		CHECK_EQ(bottom.size(), 1)
				<< "ConvLSTM layer exactly have one bottom X[t]";
		//CHECK_GE(bottom->size(), 1) 
		//	<< "ConvLSTM layer at least have one bottom. Bottom in order X[t], optional ( H[t-1], C[t-1] )";
		//CHECK_LE(bottom->size(), 3)
		//	<< "ConvLSTM layer at most have three bottoms. Bottom in order X[t], optional ( H[t-1], C[t-1] )";

		// Input shape should like this:
		// [0] : seq length * seq num(set as 1 now)
		// [1] : input_channel
		// [2] : Height
		// [3] : Width
		seq_num_ = 1;
		seq_len_ = bottom[0]->shape(0);
		spatial_dims = bottom[0]->count(2); //H*W
		num_output_ = this->layer_param_.convolution_param().num_output(); // number of hidden units channels
		const FillerParameter& weight_filler = this->layer_param_.convolution_param().weight_filler();
		const FillerParameter& bias_filler = this->layer_param_.convolution_param().bias_filler();

		// All units' shape in ConvLSTM should like this:
		// [0]: seq length
		// [1]: num_output
		// [2]: Height
		// [3]: Width
		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}
		unit_shape[1] = num_output_;
		H_0_.Reshape(unit_shape);
		C_0_.Reshape(unit_shape);

		//Set convolution_param
		LayerParameter conv_layer_param(this->layer_param_);
		ParamSpec* param_weight = conv_layer_param.add_param();
		param_weight->set_lr_mult(1);
		ParamSpec* param_bias = conv_layer_param.add_param();
		param_bias->set_lr_mult(2);

		ConvolutionParameter* conv_x_param = conv_layer_param.mutable_convolution_param();
		conv_x_param->mutable_weight_filler()->CopyFrom(weight_filler);
		conv_x_param->mutable_bias_filler()->CopyFrom(bias_filler);
		int kernel_size = conv_x_param->kernel_size().Get(0);
		int dilation = 1;
		if (conv_x_param->dilation().size() != 0)
			dilation = conv_x_param->dilation().Get(0);
		CHECK_EQ(kernel_size % 2, 1);
		conv_x_param->clear_stride();
		conv_x_param->clear_pad();
		conv_x_param->add_pad(dilation * ((kernel_size - 1) / 2));
		conv_x_param->set_axis(1);
		//input X should contribute to input_gate(I), forget_gate(F), memory cell(C), output_gate(O)
		conv_x_param->set_num_output(num_output_ * 4);
		conv_x_param->set_bias_term(true); // for Bi, Bf, Bc, Bo

		//Set up conv_x_layer_
		conv_x_bottom_vec_.clear();
		conv_x_bottom_vec_.push_back(bottom[0]);
		conv_x_top_vec_.clear();
		conv_x_top_vec_.push_back(&conv_x_top_blob_);
		conv_x_layer_.reset(new ConvolutionLayer<Dtype>(conv_layer_param));
		conv_x_layer_->SetUp(conv_x_bottom_vec_, conv_x_top_vec_);

		//Set up conv_h_layer_
		LayerParameter hidden_conv_param(conv_layer_param);
		ConvolutionParameter* conv_h_param = hidden_conv_param.mutable_convolution_param();
		conv_h_param->mutable_weight_filler()->CopyFrom(weight_filler);
		conv_h_param->mutable_bias_filler()->CopyFrom(bias_filler);
		conv_h_param->set_axis(1);
		conv_h_param->set_bias_term(false);
		// H[t-1] should contribute to input_gate(I), forget_gate(F), memory cell(C), output_gate(O)
		conv_h_param->set_num_output(num_output_ * 4);

		conv_h_bottom_vec_.clear();
		conv_h_bottom_vec_.push_back(&H_0_);
		conv_h_top_vec_.clear();
		conv_h_top_vec_.push_back(&conv_h_top_blob_);
		conv_h_layer_.reset(new ConvolutionLayer<Dtype>(hidden_conv_param));
		conv_h_layer_->SetUp(conv_h_bottom_vec_, conv_h_top_vec_);

		shared_ptr<Filler<Dtype> > matrix_filler( GetFiller<Dtype>(weight_filler) );
		matrix_filler->Fill(&Wci_);
		matrix_filler->Fill(&Wcf_);

		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
			if (conv_x_layer_->blobs()[0]->shape() != this->blobs_[0]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[0]->shape()";
			if (conv_x_layer_->blobs()[1]->shape() != this->blobs_[1]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[1]->shape()";
			if (conv_h_layer_->blobs()[0]->shape() != this->blobs_[2]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[2]->shape()";
			if (Wci_.shape() != this->blobs_[3]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[3]->shape()";
			if (Wcf_.shape() != this->blobs_[4]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[4]->shape()";

			conv_x_layer_->blobs()[0]->ShareData(*(this->blobs_[0]));
			conv_x_layer_->blobs()[1]->ShareData(*(this->blobs_[1]));
			conv_h_layer_->blobs()[0]->ShareData(*(this->blobs_[2]));
			Wci_.ShareData(*(this->blobs_[3]));
			Wcf_.ShareData(*(this->blobs_[4]));

			conv_x_layer_->blobs()[0]->ShareDiff(*(this->blobs_[0]));
			conv_x_layer_->blobs()[1]->ShareDiff(*(this->blobs_[1]));
			conv_h_layer_->blobs()[0]->ShareDiff(*(this->blobs_[2]));
			Wci_.ShareDiff(*(this->blobs_[3]));
			Wcf_.ShareDiff(*(this->blobs_[4]));
		}
		else {
			this->blobs_.resize(5);

			this->blobs_[0].reset(new Blob<Dtype>(conv_x_layer_->blobs()[0]->shape())); // weight for input conv
			this->blobs_[1].reset(new Blob<Dtype>(conv_x_layer_->blobs()[1]->shape())); // bias for input conv
			this->blobs_[2].reset(new Blob<Dtype>(conv_h_layer_->blobs()[0]->shape())); // weight for H[t-1] conv
			this->blobs_[3].reset(new Blob<Dtype>(Wci_.shape()));
			this->blobs_[4].reset(new Blob<Dtype>(Wcf_.shape()));

			this->blobs_[0]->ShareData(*(conv_x_layer_->blobs()[0]));
			this->blobs_[1]->ShareData(*(conv_x_layer_->blobs()[1]));
			this->blobs_[2]->ShareData(*(conv_h_layer_->blobs()[0]));
			this->blobs_[3]->ShareData(Wci_);
			this->blobs_[4]->ShareData(Wcf_);

			this->blobs_[0]->ShareDiff(*(conv_x_layer_->blobs()[0]));
			this->blobs_[1]->ShareDiff(*(conv_x_layer_->blobs()[1]));
			this->blobs_[2]->ShareDiff(*(conv_h_layer_->blobs()[0]));
			this->blobs_[3]->ShareDiff(Wci_);
			this->blobs_[4]->ShareDiff(Wcf_);
		}
		this->param_propagate_down_.resize(this->blobs_.size(), true);
	}

	template <typename Dtype>
	void ConvLSTMLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (this->blobs_[0]->data() != conv_x_layer_->blobs()[0]->data()){
			LOG(INFO) << "share data/diff with blobs_[0]";
			conv_x_layer_->blobs()[0]->ShareData(*(this->blobs_[0]));
			conv_x_layer_->blobs()[0]->ShareDiff(*(this->blobs_[0]));
		}
		if (this->blobs_[1]->data() != conv_x_layer_->blobs()[1]->data()){
			LOG(INFO) << "share data/diff with blobs_[1]";
			conv_x_layer_->blobs()[1]->ShareData(*(this->blobs_[1]));
			conv_x_layer_->blobs()[1]->ShareDiff(*(this->blobs_[1]));
		}
		if (this->blobs_[2]->data() != conv_h_layer_->blobs()[0]->data()){
			LOG(INFO) << "share data/diff with blobs_[2]";
			conv_h_layer_->blobs()[0]->ShareData(*(this->blobs_[2]));
			conv_h_layer_->blobs()[0]->ShareDiff(*(this->blobs_[2]));
		}
		if (this->blobs_[3]->data() != Wci_.data()){
			LOG(INFO) << "share data/diff with blobs_[3]";
			Wci_.ShareData(*(this->blobs_[3]));
			Wci_.ShareDiff(*(this->blobs_[3]));
		}
		if (this->blobs_[4]->data() != Wcf_.data()){
			LOG(INFO) << "share data/diff with blobs_[4]";
			Wcf_.ShareData(*(this->blobs_[4]));
			Wcf_.ShareDiff(*(this->blobs_[4]));
		}

		seq_len_ = bottom[0]->shape(0); // seq len
		spatial_dims = bottom[0]->count(2); //H*W

		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}
		unit_shape[1] = num_output_;
		H_0_.Reshape(unit_shape);
		C_0_.Reshape(unit_shape);
		top[0]->Reshape(unit_shape);
		Wci_.Reshape(unit_shape);
		Wcf_.Reshape(unit_shape);

		conv_x_layer_->Reshape(conv_x_bottom_vec_, conv_x_top_vec_);
		conv_h_layer_->Reshape(conv_h_bottom_vec_, conv_h_top_vec_);
	}

	template <typename Dtype>
	void ConvLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		caffe_set(H_0_.count(), Dtype(0.), H_0_.mutable_cpu_data());
		caffe_set(C_0_.count(), Dtype(0.), C_0_.mutable_cpu_data());

		// X[t] -> Wxi*X[t], Wxf*X[t], Wxc*X[t], Wxo*X[t]
		conv_x_layer_->Forward(conv_x_bottom_vec_, conv_x_top_vec_);
		
		for (int t = 0; t < seq_len_; ++t)
		{
			// H[t] -> Whi*H[t], Whf*H[t], Whc*H[t], Who*H[t]
			conv_h_layer_->Forward(conv_h_bottom_vec_, conv_h_top_vec_);

		}
	}

	template <typename Dtype>
	void ConvLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

	}

#ifdef CPU_ONLY
	STUB_GPU(ConvLSTMLayer);
#endif

	INSTANTIATE_CLASS(ConvLSTMLayer);
	REGISTER_LAYER_CLASS(ConvLSTM);

}  // namespace caffe