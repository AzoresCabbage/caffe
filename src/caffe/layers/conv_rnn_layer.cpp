#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_rnn_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

	template <typename Dtype>
	void ConvRNNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		// TODO(Yujie) : support multiple sequences in one mini batch
		
		// Input shape should like this:
		// [0] : seq length * seq num(set as 1 now)
		// [1] : input_channel
		// [2] : Height
		// [3] : Width
		seq_num_ = 1;
		seq_len_ = bottom[0]->shape(0);
		spatial_dims_ = bottom[0]->count(2); //H*W

		channelwise_conv_ = this->layer_param_.conv_rnn_param().channelwise();
		const FillerParameter& x_weight_filler = this->layer_param_.conv_rnn_param().x_weight_filler();
		const FillerParameter& x_bias_filler = this->layer_param_.conv_rnn_param().x_bias_filler();
		const FillerParameter& h_weight_filler = this->layer_param_.conv_rnn_param().h_weight_filler();
		const FillerParameter& h_bias_filler = this->layer_param_.conv_rnn_param().h_bias_filler();
		num_output_ = this->layer_param_.conv_rnn_param().num_output();
		act_type_ = this->layer_param_.conv_rnn_param().act_type();
		is_warping_ = this->layer_param_.conv_rnn_param().warping();

		// check
		if (channelwise_conv_) {
			CHECK(num_output_ == bottom[0]->channels()) << "If choose channel wise conv, the output channels must equal input channels.";
		}
		if (is_warping_)
		{
			CHECK_EQ(bottom.size(), 2)
				<< "ConvRNN layer exactly have two bottom X[t] and warp[t] when using warpping";
			CHECK_EQ(bottom[0]->num(), bottom[1]->num() + 1)
				<< "constraint corrupt: In sequence, x.num == warp.num + 1";
			CHECK_EQ(2, bottom[1]->channels())
				<< "Warping optical flow's dimension should be 2";
			CHECK_EQ(bottom[0]->height(), bottom[1]->height())
				<< "height not compatible between X and Warp";
			CHECK_EQ(bottom[0]->width(), bottom[1]->width())
				<< "width not compatible between X and Warp";
		}
		else
		{
			CHECK_EQ(bottom.size(), 1)
				<< "ConvRNN layer exactly have one bottom X[t] when without warpping";
		}

		if (act_type_ == 1)
			act_func_.reset(new Tanh<Dtype>());
		else if (act_type_ == 2)
			act_func_.reset(new Relu<Dtype>());
		else
			act_func_.reset(new Sigmoid<Dtype>());

		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}
		// [0]: 1
		// [1]: num_output
		// [2]: Height
		// [3]: Width
		unit_shape[0] = 1;
		unit_shape[1] = num_output_;
		H_0_.Reshape(unit_shape);
		conv_h_btm_blob_.Reshape(unit_shape);

		//Set convolution_param
		LayerParameter conv_layer_param(this->layer_param_);
		conv_layer_param.add_param()->set_lr_mult(1);
		conv_layer_param.add_param()->set_lr_mult(2);

		ConvolutionParameter* conv_x_param = conv_layer_param.mutable_convolution_param();
		conv_x_param->mutable_weight_filler()->CopyFrom(x_weight_filler);
		conv_x_param->mutable_bias_filler()->CopyFrom(x_bias_filler);
		int kernel_size = conv_x_param->kernel_size().Get(0);
		int dilation = 1;
		if (conv_x_param->dilation().size() != 0)
			dilation = conv_x_param->dilation().Get(0);
		CHECK_EQ(kernel_size % 2, 1);
		conv_x_param->clear_stride();
		conv_x_param->clear_pad();
		conv_x_param->add_pad(dilation * ((kernel_size - 1) / 2));
		conv_x_param->set_axis(1);
		if (channelwise_conv_)
			conv_x_param->set_group(num_output_);
		else
			conv_x_param->clear_group();
		conv_x_param->set_num_output(num_output_);
		conv_x_param->set_bias_term(true); // for bu

		//Set up conv_x_layer_
		conv_x_btm_vec_.clear();
		conv_x_btm_vec_.push_back(bottom[0]);
		conv_x_top_vec_.clear();
		conv_x_top_vec_.push_back(&conv_x_top_blob_);
		conv_x_layer_.reset(new ConvolutionLayer<Dtype>(conv_layer_param));
		conv_x_layer_->SetUp(conv_x_btm_vec_, conv_x_top_vec_);

		//Set up conv_h_layer_
		LayerParameter hidden_conv_param(conv_layer_param);
		ConvolutionParameter* conv_h_param = hidden_conv_param.mutable_convolution_param();
		conv_h_param->mutable_weight_filler()->CopyFrom(h_weight_filler);
		conv_h_param->mutable_bias_filler()->CopyFrom(h_bias_filler);
		conv_h_param->set_bias_term(false);

		conv_h_btm_vec_.clear();
		conv_h_btm_vec_.push_back(&conv_h_btm_blob_);
		conv_h_top_vec_.clear();
		conv_h_top_vec_.push_back(&conv_h_top_blob_);
		conv_h_layer_.reset(new ConvolutionLayer<Dtype>(hidden_conv_param));
		conv_h_layer_->SetUp(conv_h_btm_vec_, conv_h_top_vec_);

		// Set up Warping layer if necessary
		if (is_warping_)
		{
			vector<int> warp_shape{ 1, 2, H_0_.height(), H_0_.width() };
			warping_layer_.reset(new WarpingLayer<Dtype>(this->layer_param_));
			warp_btm_blob_flow_.Reshape(warp_shape);
			warp_btm_blob_data_.Reshape(unit_shape);
			warp_btm_vec_.clear();
			warp_btm_vec_.push_back(&warp_btm_blob_data_);
			warp_btm_vec_.push_back(&warp_btm_blob_flow_);
			warp_top_vec_.clear();
			warp_top_vec_.push_back(&warp_top_blob_);
			warping_layer_->SetUp(warp_btm_vec_, warp_top_vec_);
			warp_0_.Reshape(vector<int>{1,2,H_0_.height(),H_0_.width()});
		}

		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
			if (conv_x_layer_->blobs()[0]->shape() != this->blobs_[0]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[0]->shape()";
			if (conv_x_layer_->blobs()[1]->shape() != this->blobs_[1]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[1]->shape()";
			if (conv_h_layer_->blobs()[0]->shape() != this->blobs_[2]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[2]->shape()";

			conv_x_layer_->blobs()[0]->ShareData(*(this->blobs_[0]));
			conv_x_layer_->blobs()[1]->ShareData(*(this->blobs_[1]));
			conv_h_layer_->blobs()[0]->ShareData(*(this->blobs_[2]));

			conv_x_layer_->blobs()[0]->ShareDiff(*(this->blobs_[0]));
			conv_x_layer_->blobs()[1]->ShareDiff(*(this->blobs_[1]));
			conv_h_layer_->blobs()[0]->ShareDiff(*(this->blobs_[2]));
		}
		else {
			this->blobs_.resize(3);

			this->blobs_[0].reset(new Blob<Dtype>(conv_x_layer_->blobs()[0]->shape())); // weight for X conv
			this->blobs_[1].reset(new Blob<Dtype>(conv_x_layer_->blobs()[1]->shape())); // bias for X conv
			this->blobs_[2].reset(new Blob<Dtype>(conv_h_layer_->blobs()[0]->shape())); // weight for H conv

			this->blobs_[0]->ShareData(*(conv_x_layer_->blobs()[0]));
			this->blobs_[1]->ShareData(*(conv_x_layer_->blobs()[1]));
			this->blobs_[2]->ShareData(*(conv_h_layer_->blobs()[0]));

			this->blobs_[0]->ShareDiff(*(conv_x_layer_->blobs()[0]));
			this->blobs_[1]->ShareDiff(*(conv_x_layer_->blobs()[1]));
			this->blobs_[2]->ShareDiff(*(conv_h_layer_->blobs()[0]));
		}
		this->param_propagate_down_.resize(this->blobs_.size(), true);
	}

	template <typename Dtype>
	void ConvRNNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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

		seq_len_ = bottom[0]->shape(0); // seq len
		spatial_dims_ = bottom[0]->count(2); //H*W

		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}
		// [0]: seq length
		// [1]: num_output
		// [2]: Height
		// [3]: Width
		unit_shape[1] = num_output_;
		top[0]->Reshape(unit_shape);

		unit_shape[0] = 1;
		H_0_.Reshape(unit_shape);
		conv_h_btm_blob_.Reshape(unit_shape);

		if (is_warping_)
		{
			vector<int> warp_shape{ 1, 2, H_0_.height(), H_0_.width() };
			warp_btm_blob_flow_.Reshape(warp_shape);
			warp_btm_blob_data_.Reshape(unit_shape);
			warp_0_.Reshape(warp_shape);
			warping_layer_->Reshape(warp_btm_vec_, warp_top_vec_);
		}

		conv_x_layer_->Reshape(conv_x_btm_vec_, conv_x_top_vec_);
		conv_h_layer_->Reshape(conv_h_btm_vec_, conv_h_top_vec_);
	}

	template <typename Dtype>
	void ConvRNNLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_cpu_data();
		int featmap_dim = spatial_dims_ * num_output_;

		caffe_set(H_0_.count(0), Dtype(0.), H_0_.mutable_cpu_data());
		// For all input X: X[t] -> U*X[t] in conv_x_top_blob_
		conv_x_layer_->Forward(conv_x_btm_vec_, conv_x_top_vec_);

		for (int t = 0; t < seq_len_; ++t)
		{
			Dtype* H_t = top_data + top[0]->offset(t);
			Dtype* H_t_1 = t == 0 ? H_0_.mutable_cpu_data() : top_data + top[0]->offset(t - 1);
			const Dtype* x_conv_t = conv_x_top_blob_.cpu_data() + conv_x_top_blob_.offset(t);
			conv_h_btm_blob_.set_cpu_data(H_t_1);
			conv_h_layer_->Forward(conv_h_btm_vec_, conv_h_top_vec_);
			caffe_add(featmap_dim, conv_h_top_blob_.cpu_data(), x_conv_t, H_t);
			for (int i = 0; i < featmap_dim; ++i)
			{
				H_t[i] = act_func_->act(H_t[i]);
			}
			if (is_warping_)
			{
				Dtype* warp_flow = t == 0 ? warp_0_.mutable_cpu_data() : bottom[1]->mutable_cpu_data() + bottom[1]->offset(t - 1);
				warp_btm_blob_data_.set_cpu_data(H_t);
				warp_btm_blob_flow_.set_cpu_data(warp_flow);
				warping_layer_->Forward(warp_btm_vec_, warp_top_vec_);
				caffe_copy(featmap_dim, warp_top_blob_.cpu_data(), H_t);
			}
		}
	}

	template <typename Dtype>
	void ConvRNNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype* top_diff = top[0]->mutable_cpu_diff();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int featmap_dim = spatial_dims_ * num_output_;

		caffe_set(H_0_.count(0), Dtype(0.), H_0_.mutable_cpu_diff());

		for (int t = seq_len_ - 1; t >= 0; --t)
		{
			Dtype* top_data_t = top_data + top[0]->offset(t);
			Dtype* h_data_t_1 = t == 0 ? H_0_.mutable_cpu_data() : top_data + top[0]->offset(t - 1);

			Dtype* h_diff_t = top_diff + top[0]->offset(t);
			Dtype* h_diff_t_1 = t == 0 ? H_0_.mutable_cpu_diff() : top_diff + top[0]->offset(t - 1);
			Dtype* h_conv_diff = conv_h_top_blob_.mutable_cpu_diff();
			Dtype* x_conv_diff_t = conv_x_top_blob_.mutable_cpu_diff() + conv_x_top_blob_.offset(t);
			
			if (is_warping_)
			{
				Dtype* warp_flow = t == 0 ? warp_0_.mutable_cpu_data() : bottom[1]->mutable_cpu_data() + bottom[1]->offset(t - 1);
				warp_btm_blob_data_.set_cpu_data(h_data_t_1);
				warp_btm_blob_flow_.set_cpu_data(warp_flow);
				caffe_copy(featmap_dim, h_diff_t, warp_top_blob_.mutable_cpu_diff());
				warping_layer_->Backward(warp_top_vec_, vector<bool>{true, true}, warp_btm_vec_);
				caffe_copy(featmap_dim, warp_btm_blob_data_.cpu_diff(), h_diff_t);
			}
			
			for (int i = 0; i < featmap_dim; ++i)
			{
				x_conv_diff_t[i] = h_diff_t[i] * act_func_->d_act(top_data_t[i]);
				h_conv_diff[i] = x_conv_diff_t[i];
			}
			conv_h_btm_blob_.set_cpu_data(h_data_t_1);
			conv_h_layer_->Backward(conv_h_top_vec_, vector<bool>{true}, conv_h_btm_vec_);
			caffe_add(featmap_dim, h_diff_t_1, conv_h_btm_blob_.cpu_diff(), h_diff_t_1);
		}
		conv_x_layer_->Backward(conv_x_top_vec_, vector<bool>{propagate_down[0]}, conv_x_btm_vec_);
	}

#ifdef CPU_ONLY
	STUB_GPU(ConvRNNLayer);
#endif

	INSTANTIATE_CLASS(ConvRNNLayer);
	REGISTER_LAYER_CLASS(ConvRNN);

}  // namespace caffe