#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/conv_gru_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

	template <typename Dtype>
	inline Dtype sigmoid(Dtype x) {
		return 1. / (1. + exp(-x));
	}

	template <typename Dtype>
	inline Dtype d_sigmoid(Dtype x) {
		return x * (1 - x);
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
	Dtype tanh(const Dtype x) {
		return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
	}

	template <typename Dtype>
	inline Dtype d_tanh(const Dtype x) {
		return 1 - x * x;
	}

	template <typename Dtype>
	void ConvGRULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		CHECK_LE(top.size(), 2) << "ConvLayer must have one or three[hidden, gate(r,z)] top blob";
		CHECK_LE(bottom.size(), 2) << "ConvLayer must have one or two bottom for X and H0";

		clipping_threshold_ = this->layer_param_.conv_gru_param().clipping_threshold();
		BN_term = this->layer_param_.conv_gru_param().bn_term();
		const FillerParameter& weight_filler = this->layer_param_.conv_gru_param().weight_filler();
		const FillerParameter& bias_filler = this->layer_param_.conv_gru_param().bias_filler();
		// Input shape should like this:
		// btm[0] : seq length
		// btm[1] : input channel
		// btm[2] : H
		// btm[3] : W
		seq_len_ = bottom[0]->shape(0); // seq length
		spatial_dims_ = bottom[0]->count(2); //H*W
		num_output_ = this->layer_param_.conv_gru_param().num_output(); // number of hidden units channels
		forward_direction_ = this->layer_param_.conv_gru_param().forward_direction();

		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}
		unit_shape[1] = num_output_;
		Uh_h_.Reshape(unit_shape);
		unit_shape[0] = 1;
		// 0: 1
		// 1: num_output
		// 2: H
		// 3: W
		h_0_.Reshape(unit_shape);
		conv_h_btm_blob_.Reshape(unit_shape);

		//Set convolution_param
		LayerParameter conv_layer_param(this->layer_param_);

		ConvolutionParameter* ptr_conv_param = conv_layer_param.mutable_convolution_param();
		ptr_conv_param->mutable_weight_filler()->CopyFrom(weight_filler);
		ptr_conv_param->mutable_bias_filler()->CopyFrom(bias_filler);
		ptr_conv_param->set_bias_term(true);
		CHECK_EQ(ptr_conv_param->kernel_size().size(), 1);
		int kernel_size = ptr_conv_param->kernel_size().Get(0);
		int dilation = 1;
		if (ptr_conv_param->dilation().size() != 0)
			dilation = ptr_conv_param->dilation().Get(0);
		CHECK_EQ(kernel_size % 2, 1);
		ptr_conv_param->clear_stride();
		ptr_conv_param->clear_pad();
		ptr_conv_param->add_pad(dilation * ((kernel_size - 1) / 2));
		ptr_conv_param->set_axis(1); // see input shape above
		//input X should contribute to reset_gate(R), update_gate(Z) and candidate_act in order
		ptr_conv_param->set_num_output(num_output_ * 3);
		//ptr_conv_param->set_num_output(num_output_);

		//Set up conv_x_layer_
		conv_x_bottom_vec_.clear();
		conv_x_bottom_vec_.push_back(bottom[0]);
		conv_x_top_vec_.clear();
		conv_x_top_vec_.push_back(&conv_x_top_blob_);
		conv_x_layer_.reset(new ConvolutionLayer<Dtype>(conv_layer_param));
		conv_x_layer_->SetUp(conv_x_bottom_vec_, conv_x_top_vec_);

		//Set up conv_h_layer_
		LayerParameter hidden_conv_param(conv_layer_param);
		ptr_conv_param = hidden_conv_param.mutable_convolution_param();
		ptr_conv_param->mutable_weight_filler()->CopyFrom(weight_filler);
		ptr_conv_param->mutable_bias_filler()->CopyFrom(bias_filler);
		ptr_conv_param->set_axis(1); // see unit shape above
		ptr_conv_param->set_bias_term(false);
		// H[t-1] should contribute to Ur, Uz, Uh
		ptr_conv_param->set_num_output(num_output_ * 3);

		conv_h_bottom_vec_.clear();
		conv_h_bottom_vec_.push_back(&conv_h_btm_blob_);
		conv_h_top_vec_.clear();
		conv_h_top_vec_.push_back(&conv_h_top_blob_);
		conv_h_layer_.reset(new ConvolutionLayer<Dtype>(hidden_conv_param));
		conv_h_layer_->SetUp(conv_h_bottom_vec_, conv_h_top_vec_);

		if (BN_term)
		{
			LayerParameter batch_norm_param(this->layer_param_);
			batch_norm_param.add_param()->set_lr_mult(0);
			batch_norm_param.add_param()->set_lr_mult(0);
			batch_norm_param.add_param()->set_lr_mult(0);

			batch_norm_x_btm_vec_.clear();
			batch_norm_x_btm_vec_.push_back(&conv_x_top_blob_);
			batch_norm_x_top_vec_.clear();
			batch_norm_x_top_vec_.push_back(&conv_x_top_blob_);
			batch_norm_x_layer_.reset(new BatchNormLayer<Dtype>(batch_norm_param));
			batch_norm_x_layer_->SetUp(batch_norm_x_btm_vec_, batch_norm_x_top_vec_);

			batch_norm_h_btm_vec_.clear();
			batch_norm_h_btm_vec_.push_back(&conv_h_top_blob_);
			batch_norm_h_top_vec_.clear();
			batch_norm_h_top_vec_.push_back(&conv_h_top_blob_);
			batch_norm_h_layer_.reset(new BatchNormLayer<Dtype>(batch_norm_param));
			batch_norm_h_layer_->SetUp(batch_norm_h_btm_vec_, batch_norm_h_top_vec_);

			LayerParameter scale_param(this->layer_param_);
			scale_x_btm_vec_.clear();
			scale_x_btm_vec_.push_back(&conv_x_top_blob_);
			scale_x_top_vec_.clear();
			scale_x_top_vec_.push_back(&conv_x_top_blob_);
			scale_x_layer_.reset(new ScaleLayer<Dtype>(scale_param));
			scale_x_layer_->SetUp(conv_x_top_vec_, conv_x_top_vec_);

			scale_h_btm_vec_.clear();
			scale_h_btm_vec_.push_back(&conv_h_top_blob_);
			scale_h_top_vec_.clear();
			scale_h_top_vec_.push_back(&conv_h_top_blob_);
			scale_h_layer_.reset(new ScaleLayer<Dtype>(scale_param));
			scale_h_layer_->SetUp(scale_h_btm_vec_, scale_h_top_vec_);
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

			this->blobs_[0].reset(new Blob<Dtype>(conv_x_layer_->blobs()[0]->shape())); // weight for input conv
			this->blobs_[1].reset(new Blob<Dtype>(conv_x_layer_->blobs()[1]->shape())); // bias for input conv
			this->blobs_[2].reset(new Blob<Dtype>(conv_h_layer_->blobs()[0]->shape())); // weight for H[t-1] conv
			
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
	void ConvGRULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
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
		unit_shape[1] = num_output_;
		top[0]->Reshape(unit_shape);
		Uh_h_.Reshape(unit_shape);

		unit_shape[0] = 1;
		h_0_.Reshape(unit_shape);
		conv_h_btm_blob_.Reshape(unit_shape);

		conv_x_layer_->Reshape(conv_x_bottom_vec_, conv_x_top_vec_);
		conv_h_layer_->Reshape(conv_h_bottom_vec_, conv_h_top_vec_);

		if (BN_term)
		{
			batch_norm_x_layer_->Reshape(batch_norm_x_btm_vec_, batch_norm_x_top_vec_);
			scale_x_layer_->Reshape(scale_x_btm_vec_, scale_x_top_vec_);
			batch_norm_h_layer_->Reshape(batch_norm_h_btm_vec_, batch_norm_h_top_vec_);
			scale_h_layer_->Reshape(scale_h_btm_vec_, scale_h_top_vec_);
		}
		if (top.size() > 1)
		{
			top[1]->ReshapeLike(conv_x_top_blob_);
		}
		if (bottom.size() == 2)
		{
			h_0_.ShareData(*(bottom[1]));
			h_0_.ShareDiff(*(bottom[1]));
		}
	}

	template <typename Dtype>
	void ConvGRULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		int feature_dims = num_output_ * spatial_dims_; // one input's size
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* conv_x_top_data = conv_x_top_blob_.mutable_cpu_data();
		Dtype* conv_h_top_data = conv_h_top_blob_.mutable_cpu_data();
		Dtype* btm_data = bottom[0]->mutable_cpu_data();
		// Compute input to gate forward propagation
		// W*Xt, Wz*Xt, Wr*Xt
		conv_x_layer_->Forward(conv_x_bottom_vec_, conv_x_top_vec_);
		//caffe_copy(conv_x_top_blob_.count(), btm_data, conv_x_top_blob_.mutable_cpu_data());
		if (BN_term)
		{
			batch_norm_x_layer_->Forward(batch_norm_x_btm_vec_, batch_norm_x_top_vec_);
			scale_x_layer_->Forward(scale_x_btm_vec_, scale_x_top_vec_);
		}
		//caffe_copy(top[0]->count(), conv_x_top_data, top_data);

		// Initialize previous state
		
		if (bottom.size() == 1)
		{
			caffe_set(h_0_.count(0), Dtype(0.), h_0_.mutable_cpu_data());
		}
		
		// Compute recurrent forward propagation
		for (int tt = 0; tt < seq_len_; ++tt) {
			int t = tt;
		
			Dtype* conv_x_top_t_data = conv_x_top_data + conv_x_top_blob_.offset(t);
			Dtype* Wr_x_t_data = conv_x_top_t_data + 0 * feature_dims;
			Dtype* Wz_x_t_data = conv_x_top_t_data + 1 * feature_dims;
			Dtype* Wh_x_t_data = conv_x_top_t_data + 2 * feature_dims;
		
			Dtype* h_t_data = top_data + top[0]->offset(t);
			Dtype* h_t_1_data = t > 0 ? (top_data + top[0]->offset(t - 1)) : h_0_.mutable_cpu_data();
		
			conv_h_btm_blob_.set_cpu_data(h_t_1_data);
			// Ur*H[t-1], Uz*H[t-1], Uh*H[t-1]
			conv_h_layer_->Forward(conv_h_bottom_vec_, conv_h_top_vec_);
			if (BN_term)
			{
				batch_norm_h_layer_->Forward(batch_norm_h_btm_vec_, batch_norm_h_top_vec_);
				scale_h_layer_->Forward(scale_h_btm_vec_, scale_h_top_vec_);
			}
		
			Dtype* Ur_h_t_1_data = conv_h_top_data + 0 * feature_dims;
			Dtype* Uz_h_t_1_data = conv_h_top_data + 1 * feature_dims;
			Dtype* Uh_h_t_1_data = conv_h_top_data + 2 * feature_dims;
			
			caffe_copy(feature_dims, Uh_h_t_1_data, Uh_h_.mutable_cpu_data() + Uh_h_.offset(t));
		
			// Wr*Xr + Ur*H[t-1] ; Wz*Xt + Uz*H[t-1]
			caffe_add(feature_dims, Wr_x_t_data, Ur_h_t_1_data, Wr_x_t_data);
			caffe_add(feature_dims, Wz_x_t_data, Uz_h_t_1_data, Wz_x_t_data);
		
			// reset_gate or Rt = sigmoid(Wr*Xr + Ur*H[t-1])
			// and
			// update_gate or Zt = sigmoid(Wz*Xt + Uz*H[t-1])
			for (int d = 0; d < feature_dims; ++d) {
				Wr_x_t_data[d] = sigmoid(Wr_x_t_data[d]);
				Wz_x_t_data[d] = sigmoid(Wz_x_t_data[d]);
			}
		
			// Wh_data = tanh(W*X + U*(Rt .* H[t-1]))
			for (int d = 0; d < feature_dims; ++d) {
				// Apply nonlinearity
				Wh_x_t_data[d] = tanh(Wh_x_t_data[d] + Wr_x_t_data[d] * Uh_h_t_1_data[d]);
			}
		
			for (int d = 0; d < feature_dims; ++d) {
				h_t_data[d] = (1 - Wz_x_t_data[d]) * h_t_1_data[d]
					+ Wz_x_t_data[d] * Wh_x_t_data[d];
			}
		} // for tt
		if (top.size() > 1)
		{
			caffe_copy(conv_x_top_blob_.count(0), conv_x_top_blob_.cpu_data(), top[1]->mutable_cpu_data());
		}
	}

	template <typename Dtype>
	void ConvGRULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		int feature_dims = num_output_ * spatial_dims_;
		Dtype* top_data = top[0]->mutable_cpu_data();
		Dtype* top_diff = top[0]->mutable_cpu_diff();

		Dtype* conv_x_top_data = conv_x_top_blob_.mutable_cpu_data();
		Dtype* conv_x_top_diff = conv_x_top_blob_.mutable_cpu_diff();

		Dtype* conv_h_top_data = conv_h_top_blob_.mutable_cpu_data();
		Dtype* conv_h_top_diff = conv_h_top_blob_.mutable_cpu_diff();


		caffe_set(h_0_.count(0), Dtype(0.), h_0_.mutable_cpu_diff());
		
		for (int tt = seq_len_ - 1; tt >= 0; --tt) {
			int t = tt;
		
			Dtype* conv_x_top_t_data = conv_x_top_data + conv_x_top_blob_.offset(t);
			Dtype* Wr_x_t_data = conv_x_top_t_data + 0 * feature_dims;
			Dtype* Wz_x_t_data = conv_x_top_t_data + 1 * feature_dims;
			Dtype* Wh_x_t_data = conv_x_top_t_data + 2 * feature_dims;
		
			Dtype* conv_x_top_t_diff = conv_x_top_diff + conv_x_top_blob_.offset(t);
			Dtype* Wr_x_t_diff = conv_x_top_t_diff + 0 * feature_dims;
			Dtype* Wz_x_t_diff = conv_x_top_t_diff + 1 * feature_dims;
			Dtype* Wh_x_t_diff = conv_x_top_t_diff + 2 * feature_dims;
		
		
			Dtype* h_t_diff = top_diff + top[0]->offset(t);
		
			Dtype* h_t_1_data = t > 0 ? top_data + top[0]->offset(t - 1) : h_0_.mutable_cpu_data();
			Dtype* h_t_1_diff = t > 0 ? top_diff + top[0]->offset(t - 1) : h_0_.mutable_cpu_diff();
			
			Dtype* Ur_h_t_1_data = conv_h_top_data + 0 * feature_dims;
			Dtype* Uz_h_t_1_data = conv_h_top_data + 1 * feature_dims;
			Dtype* Uh_h_t_1_data = conv_h_top_data + 2 * feature_dims;
			Dtype* Ur_h_t_1_diff = conv_h_top_diff + 0 * feature_dims;
			Dtype* Uz_h_t_1_diff = conv_h_top_diff + 1 * feature_dims;
			Dtype* Uh_h_t_1_diff = conv_h_top_diff + 2 * feature_dims;
			for (int d = 0; d < feature_dims; ++d) {
				// top_diff -> H[t-1]
				h_t_1_diff[d] += h_t_diff[d] * (1 - Wz_x_t_data[d]);
				// top_diff -> Ht_candidate
				Dtype h_candidate_diff = h_t_diff[d] * Wz_x_t_data[d];
				// top_diff -> gate_z
				Dtype gate_z_t_diff = h_t_diff[d] * (Wh_x_t_data[d] - h_t_1_data[d]);
				// gate z -> Wz*X and Uz*H[t-1]
				Wz_x_t_diff[d] = Uz_h_t_1_diff[d] = gate_z_t_diff * d_sigmoid(Wz_x_t_data[d]);
				// h candidate -> Wh*X[t] and R[t] and Uh*H[t-1]
				Wh_x_t_diff[d] = h_candidate_diff * d_tanh(Wh_x_t_data[d]);
				
				Dtype gate_r_t_diff = Wh_x_t_diff[d] * (Uh_h_.mutable_cpu_data() + Uh_h_.offset(t))[d];
				Uh_h_t_1_diff[d] = Wh_x_t_diff[d] * Wr_x_t_data[d];
				// gate r -> Wr*X and Ur*H[t-1]
				Wr_x_t_diff[d] = Ur_h_t_1_diff[d] = gate_r_t_diff * d_sigmoid(Wr_x_t_data[d]);
			}
			conv_h_btm_blob_.set_cpu_data(h_t_1_data);
			if (BN_term)
			{
				scale_h_layer_->Backward(scale_h_top_vec_, vector<bool>{true}, scale_h_btm_vec_);
				batch_norm_h_layer_->Backward(batch_norm_h_top_vec_, vector<bool>{true}, batch_norm_h_btm_vec_);
			}
			conv_h_layer_->Backward(conv_h_top_vec_, vector<bool>{true}, conv_h_bottom_vec_);
			const Dtype* hidden_diff_ = conv_h_btm_blob_.cpu_diff();
			caffe_add(feature_dims, h_t_1_diff, hidden_diff_, h_t_1_diff);
		}

		// caffe_copy(top[0]->count(), top_diff, conv_x_top_diff);
		if (BN_term)
		{
			scale_x_layer_->Backward(scale_x_top_vec_, propagate_down, scale_x_btm_vec_);
			batch_norm_x_layer_->Backward(batch_norm_x_top_vec_, propagate_down, batch_norm_x_btm_vec_);
		}
		// Gradient w.r.t. bottom data 
		// accumulated all diff from input_pre_gate(conv) -> btm data
		// At the same time, calc all gradient for Wr, Wz, W
		conv_x_layer_->Backward(conv_x_top_vec_, propagate_down, conv_x_bottom_vec_);
		// caffe_copy(conv_x_top_blob_.count(), conv_x_top_diff, bottom[0]->mutable_cpu_diff());
	}

#ifdef CPU_ONLY
	STUB_GPU(ConvGRULayer);
#endif

	INSTANTIATE_CLASS(ConvGRULayer);
	REGISTER_LAYER_CLASS(ConvGRU);

}  // namespace caffe