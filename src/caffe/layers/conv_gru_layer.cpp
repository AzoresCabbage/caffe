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
	void ConvGRULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		clipping_threshold_ = this->layer_param_.conv_gru_param().clipping_threshold();
		// Input shape should like this:
		// btm[0] : seq length * seq num(1)
		// btm[1] : channel; 
		// btm[2] : H;
		// btm[3] : W
		T_ = bottom[0]->shape(0); // seq length
		H_ = this->layer_param_.conv_gru_param().num_output(); // number of hidden units channels
		C_ = bottom[0]->shape(1); // input channels
		forward_direction_ = this->layer_param_.conv_gru_param().forward_direction();
		N_ = 1; // seq num(1)

		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}
		unit_shape[0] = T_;
		// All units' shape in convGRU should like this, except input:
		// 0: seq length
		// 1: num_output
		// 2: H
		// 3: W
		h_0_.Reshape(unit_shape);
		hidden_.Reshape(unit_shape);
		hidden_reset_.Reshape(unit_shape);

		//Set convolution_param
		LayerParameter conv_layer_param(this->layer_param_);
		ParamSpec* param_weight = conv_layer_param.add_param();
		param_weight->set_lr_mult(1);
		ParamSpec* param_bias = conv_layer_param.add_param();
		param_bias->set_lr_mult(2);

		ConvolutionParameter* input_conv_param = conv_layer_param.mutable_convolution_param();
		CHECK_EQ(input_conv_param->kernel_size().size(), 1);
		int kernel_size = input_conv_param->kernel_size().Get(0);
		int dilation = 1;
		if (input_conv_param->dilation().size() != 0)
			dilation = input_conv_param->dilation().Get(0);
		CHECK_EQ(kernel_size % 2, 1); // Yujie: why kernel_size must be odd ?
		input_conv_param->clear_stride();
		input_conv_param->clear_pad();
		input_conv_param->add_pad(dilation * ((kernel_size - 1) / 2));
		input_conv_param->set_axis(1); // see input shape above
		//input X should contribute to reset_gate(R), update_gate(Z) and candidate_act in order
		input_conv_param->set_num_output(H_ * 3);

		//Set up conv_input_layer_
		conv_input_bottom_vec_.clear();
		conv_input_bottom_vec_.push_back(bottom[0]);
		conv_input_top_vec_.clear();
		conv_input_top_vec_.push_back(&input_pre_gate_);
		conv_input_layer_.reset(new ConvolutionLayer<Dtype>(conv_layer_param));
		conv_input_layer_->SetUp(conv_input_bottom_vec_, conv_input_top_vec_);

		//Set up conv_hidden_layer_
		LayerParameter hidden_conv_param(conv_layer_param);
		hidden_conv_param.mutable_convolution_param()->set_axis(1); // see unit shape above
		hidden_conv_param.mutable_convolution_param()->set_bias_term(false);
		// H[t-1] should contribute to Ur, Uz. Due to U * (Rt .* H[t-1]), so we need to calc it in conv_tmp_hidden_layer_
		hidden_conv_param.mutable_convolution_param()->set_num_output(H_ * 2);
		hidden_conv_param.clear_param();

		conv_hidden_bottom_vec_.clear();
		conv_hidden_bottom_vec_.push_back(&hidden_);
		conv_hidden_top_vec_.clear();
		conv_hidden_top_vec_.push_back(&hidden_pre_gate_);
		conv_hidden_layer_.reset(new ConvolutionLayer<Dtype>(hidden_conv_param));
		conv_hidden_layer_->SetUp(conv_hidden_bottom_vec_, conv_hidden_top_vec_);

		//Set up conv_tmp_hidden_layer_
		LayerParameter tmp_hidden_conv_param(hidden_conv_param);
		tmp_hidden_conv_param.mutable_convolution_param()->set_num_output(H_);
		tmp_hidden_conv_param.clear_param();

		conv_tmp_hidden_bottom_vec_.clear();
		conv_tmp_hidden_bottom_vec_.push_back(&hidden_reset_);
		conv_tmp_hidden_top_vec_.clear();
		conv_tmp_hidden_top_vec_.push_back(&hidden_rt_pre_gate_);
		conv_tmp_hidden_layer_.reset(new ConvolutionLayer<Dtype>(tmp_hidden_conv_param));
		conv_tmp_hidden_layer_->SetUp(conv_tmp_hidden_bottom_vec_, conv_tmp_hidden_top_vec_);

		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
			if (conv_input_layer_->blobs()[0]->shape() != this->blobs_[0]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[0]->shape()";
			if (conv_input_layer_->blobs()[1]->shape() != this->blobs_[1]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[1]->shape()";
			if (conv_hidden_layer_->blobs()[0]->shape() != this->blobs_[2]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[2]->shape()";
			if (conv_tmp_hidden_layer_->blobs()[0]->shape() != this->blobs_[3]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[3]->shape()";

			conv_input_layer_->blobs()[0]->ShareData(*(this->blobs_[0]));
			conv_input_layer_->blobs()[1]->ShareData(*(this->blobs_[1]));
			conv_hidden_layer_->blobs()[0]->ShareData(*(this->blobs_[2]));
			conv_tmp_hidden_layer_->blobs()[0]->ShareData(*(this->blobs_[3]));

			conv_input_layer_->blobs()[0]->ShareDiff(*(this->blobs_[0]));
			conv_input_layer_->blobs()[1]->ShareDiff(*(this->blobs_[1]));
			conv_hidden_layer_->blobs()[0]->ShareDiff(*(this->blobs_[2]));
			conv_tmp_hidden_layer_->blobs()[0]->ShareDiff(*(this->blobs_[3]));
		}
		else {
			this->blobs_.resize(4);

			this->blobs_[0].reset(new Blob<Dtype>(conv_input_layer_->blobs()[0]->shape())); // weight for input conv
			this->blobs_[1].reset(new Blob<Dtype>(conv_input_layer_->blobs()[1]->shape())); // bias for input conv
			this->blobs_[2].reset(new Blob<Dtype>(conv_hidden_layer_->blobs()[0]->shape())); // weight for H[t-1] conv
			this->blobs_[3].reset(new Blob<Dtype>(conv_tmp_hidden_layer_->blobs()[0]->shape())); // weight for U * (Rt .* H[t-1])

			this->blobs_[0]->ShareData(*(conv_input_layer_->blobs()[0]));
			this->blobs_[1]->ShareData(*(conv_input_layer_->blobs()[1]));
			this->blobs_[2]->ShareData(*(conv_hidden_layer_->blobs()[0]));
			this->blobs_[3]->ShareData(*(conv_tmp_hidden_layer_->blobs()[0]));

			this->blobs_[0]->ShareDiff(*(conv_input_layer_->blobs()[0]));
			this->blobs_[1]->ShareDiff(*(conv_input_layer_->blobs()[1]));
			this->blobs_[2]->ShareDiff(*(conv_hidden_layer_->blobs()[0]));
			this->blobs_[3]->ShareDiff(*(conv_tmp_hidden_layer_->blobs()[0]));
		}
		this->param_propagate_down_.resize(this->blobs_.size(), true);
	}

	template <typename Dtype>
	void ConvGRULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		if (this->blobs_[0]->data() != conv_input_layer_->blobs()[0]->data()){
			LOG(INFO) << "share data/diff with blobs_[0]";
			conv_input_layer_->blobs()[0]->ShareData(*(this->blobs_[0]));
			conv_input_layer_->blobs()[0]->ShareDiff(*(this->blobs_[0]));
		}
		if (this->blobs_[1]->data() != conv_input_layer_->blobs()[1]->data()){
			LOG(INFO) << "share data/diff with blobs_[1]";
			conv_input_layer_->blobs()[1]->ShareData(*(this->blobs_[1]));
			conv_input_layer_->blobs()[1]->ShareDiff(*(this->blobs_[1]));
		}
		if (this->blobs_[2]->data() != conv_hidden_layer_->blobs()[0]->data()){
			LOG(INFO) << "share data/diff with blobs_[2]";
			conv_hidden_layer_->blobs()[0]->ShareData(*(this->blobs_[2]));
			conv_hidden_layer_->blobs()[0]->ShareDiff(*(this->blobs_[2]));
		}
		if (this->blobs_[3]->data() != conv_tmp_hidden_layer_->blobs()[0]->data()){
			LOG(INFO) << "share data/diff with blobs_[3]";
			conv_tmp_hidden_layer_->blobs()[0]->ShareData(*(this->blobs_[3]));
			conv_tmp_hidden_layer_->blobs()[0]->ShareDiff(*(this->blobs_[3]));
		}

		T_ = bottom[0]->shape(0); // seq len
		spatial_dims = bottom[0]->count(2); //H*W
		// Figure out the dimensions
		CHECK_EQ(bottom[0]->shape(1), C_) << "Input size "
			"incompatible with inner product parameters.";

		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}
		unit_shape[0] = T_;
		h_0_.Reshape(unit_shape);
		hidden_.Reshape(unit_shape);
		hidden_reset_.Reshape(unit_shape);

		vector<int> original_top_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			original_top_shape.push_back(bottom[0]->shape()[i]);
		}
		top[0]->Reshape(original_top_shape);

		conv_input_layer_->Reshape(conv_input_bottom_vec_, conv_input_top_vec_);
		conv_hidden_layer_->Reshape(conv_hidden_bottom_vec_, conv_hidden_top_vec_);
		conv_tmp_hidden_layer_->Reshape(conv_tmp_hidden_bottom_vec_, conv_tmp_hidden_top_vec_);
	}

	template <typename Dtype>
	void ConvGRULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		Dtype* top_data = top[0]->mutable_cpu_data();

		Dtype* input_pre_gate_data = input_pre_gate_.mutable_cpu_data(); // conv_input_top_vec
		Dtype* hidden_pre_gate_data = hidden_pre_gate_.mutable_cpu_data(); // conv_hidden_top_vec
		Dtype* hidden_rt_data = hidden_reset_.mutable_cpu_data(); // conv_tmp_hidden_bottom_vec
		Dtype* hidden_rt_pre_gate_data = hidden_rt_pre_gate_.mutable_cpu_data(); // conv_tmp_hidden_top_vec
		int feature_dims = H_ * spatial_dims; // one input's size

		// Initialize previous state
		caffe_set(h_0_.count(0), Dtype(0.), h_0_.mutable_cpu_data());

		// Compute input to gate forward propagation
		// W*Xt, Wz*Xt, Wr*Xt
		conv_input_layer_->Forward(conv_input_bottom_vec_, conv_input_top_vec_);

		// Compute recurrent forward propagation
		for (int tt = 0; tt < T_; ++tt) {
			int t = tt;
			if (!forward_direction_) t = T_ - tt - 1;

			Dtype* h_t = top_data + top[0]->count(1) * t; // position that store current feature map in same seq order
			Dtype* input_pre_gate_t = input_pre_gate_data + input_pre_gate_.count(1) * t; // W[i]*Xt

			Dtype* h_t_1 = t > 0 ? (h_t - top[0]->count(1)) : h_0_.mutable_cpu_data(); // position that store prev data

			if (!forward_direction_){
				h_t_1 = t < T_ - 1 ? (h_t + top[0]->count(1)) : h_0_.mutable_cpu_data();
			}

			// Hidden-to-hidden propagation
			hidden_.set_cpu_data(h_t_1); // conv_hidden_bottom_vec
			// Ur*H[t-1], Uz*H[t-1]
			conv_hidden_layer_->Forward(conv_hidden_bottom_vec_, conv_hidden_top_vec_);

			for (int n = 0; n < N_; ++n) {
				Dtype* input_pre_gate_t_n = input_pre_gate_t + n * 3 * feature_dims;
				Dtype* hidden_pre_gate_data_n = hidden_pre_gate_data + n * 2 * feature_dims;
				Dtype* hidden_rt_data_n = hidden_rt_data + n * feature_dims;
				Dtype* h_t_1_n = h_t_1 + n * feature_dims;

				// Wr*Xr + Ur*H[t-1] ; Wz*Xt + Uz*H[t-1]
				caffe_add(2 * feature_dims, input_pre_gate_t_n, hidden_pre_gate_data_n, input_pre_gate_t_n);
				// reset_gate or Rt = sigmoid(Wr*Xr + Ur*H[t-1]) ; update_gate or Zt = sigmoid(Wz*Xt + Uz*H[t-1])
				for (int d = 0; d < 2 * feature_dims; ++d) {
					// Apply nonlinearity
					input_pre_gate_t_n[d] = hard_sigmoid(input_pre_gate_t_n[d]);
				}
				//Rt .* H[t-1]
				for (int d = 0; d < feature_dims; ++d){
					hidden_rt_data_n[d] = input_pre_gate_t_n[d] * h_t_1_n[d];
				}
			}
			
			//U*(Rt .* H[t-1])
			conv_tmp_hidden_layer_->Forward(conv_tmp_hidden_bottom_vec_, conv_tmp_hidden_top_vec_);

			for (int n = 0; n < N_; ++n) {
				Dtype* input_pre_gate_t_n = input_pre_gate_t + n * 3 * feature_dims + 2 * feature_dims;
				Dtype* hidden_rt_pre_gate_data_n = hidden_rt_pre_gate_data + n * feature_dims;
				// W*X + U*(Rt .* H[t-1])
				caffe_add(feature_dims, input_pre_gate_t_n,	hidden_rt_pre_gate_data_n, input_pre_gate_t_n);
				// tanh(W*X + U*(Rt .* H[t-1]))
				for (int d = 0; d < feature_dims; ++d) {
					// Apply nonlinearity
					input_pre_gate_t_n[d] = tanh(input_pre_gate_t_n[d]);
				}
			}

			for (int n = 0; n < N_; ++n) {
				Dtype* z_t_n = input_pre_gate_t + n * 3 * feature_dims + feature_dims;
				Dtype* h_t_1_n = h_t_1 + n * feature_dims;
				Dtype* h_t_n = h_t + n * feature_dims;
				Dtype* tmp_h_t_n = input_pre_gate_t + n * 3 * feature_dims + 2 * feature_dims;
				// Yujie : Any effect? in the paper, the equation is: H[t] = (1-Zt) .* H[t-1] + Zt .* H[candidate]
				for (int d = 0; d < feature_dims; ++d) {
					h_t_n[d] = z_t_n[d] * h_t_1_n[d] + (1 - z_t_n[d]) * tmp_h_t_n[d];
				}
			}
		} // for tt
	}

	template <typename Dtype>
	void ConvGRULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {

		// After forward, all input_pre_gate -> Act(W[i]*Xt + U[i]*H[t-1])
		const Dtype* gate_data = input_pre_gate_.cpu_data();

		Dtype* top_diff = top[0]->mutable_cpu_diff();
		Dtype* pre_gate_diff = input_pre_gate_.mutable_cpu_diff(); // conv_input_top_vec diff
		Dtype* hidden_pre_gate_diff = hidden_pre_gate_.mutable_cpu_diff(); // conv_hidden_top_vec diff
		Dtype* hidden_rt_data = hidden_reset_.mutable_cpu_data(); // conv_tmp_hidden_bottom_vec data
		Dtype* hidden_rt_pre_gate_diff = hidden_rt_pre_gate_.mutable_cpu_diff(); // conv_tmp_hidden_top_vec diff
		const Dtype* hidden_rt_diff = hidden_reset_.mutable_cpu_diff(); // conv_tmp_hidden_bottom_vec diff
		caffe_set(h_0_.count(0), Dtype(0.), h_0_.mutable_cpu_diff());

		int feature_dims = H_ * spatial_dims;

		for (int tt = T_ - 1; tt >= 0; --tt) {
			int t = tt;
			if (!forward_direction_) t = T_ - tt - 1;

			Dtype* dh_t = top_diff + top[0]->count(1) * t; // top diff of this seq order
			Dtype* pre_gate_diff_t = pre_gate_diff + input_pre_gate_.count(1) * t; // input_conv top diff
			const Dtype* gate_t = gate_data + input_pre_gate_.count(1) * t; // input_conv top data

			// bottom diff of this seq order
			Dtype* dh_t_1 = t > 0 ? top_diff + top[0]->count(1) * (t - 1) : h_0_.mutable_cpu_diff();
			
			// bottom data of this seq order
			Dtype* h_t_1 = t > 0 ? (top[0]->mutable_cpu_data() + top[0]->count(1) * (t - 1)) : h_0_.mutable_cpu_data();
			if (!forward_direction_){
				dh_t_1 = t < T_ - 1 ? top_diff + top[0]->count(1) * (t + 1) : h_0_.mutable_cpu_diff();
				h_t_1 = t < T_ - 1 ? (top[0]->mutable_cpu_data() + top[0]->count(1) * (t + 1)) : h_0_.mutable_cpu_data();
			}

			for (int n = 0; n < N_; ++n) {
				Dtype* pre_gate_diff_t_n = pre_gate_diff_t + n * 3 * feature_dims;
				const Dtype* gate_t_n = gate_t + n * 3 * feature_dims;
				Dtype* dh_t_1_n = dh_t_1 + n * feature_dims;
				Dtype* dh_t_n = dh_t + n * feature_dims;
				Dtype* h_t_1_n = h_t_1 + n * feature_dims;
				Dtype* hidden_rt_data_n = hidden_rt_data + n * feature_dims;
				Dtype* hidden_rt_pre_gate_diff_n = hidden_rt_pre_gate_diff + n * feature_dims;

				for (int d = 0; d < feature_dims; ++d) {
					// diff: top_diff -> (1-Zt) .* H[t-1] -> H[t-1]
					// becasue in forward, the weight for H[t-1] is Zt, so 1-Zt -> Zt here actually
					dh_t_1_n[d] += dh_t_n[d] * gate_t_n[d + feature_dims];

					// diff: top_diff -> after Ht_candidate -> Ht_candidate -> after input_conv
					pre_gate_diff_t_n[d + 2 * feature_dims] = dh_t_n[d] * (1 - gate_t_n[d + feature_dims]);
					pre_gate_diff_t_n[d + 2 * feature_dims] *= 1 - gate_t_n[d + 2 * feature_dims] * gate_t_n[d + 2 * feature_dims];

					// diff: top_diff -> Zt and 1-Zt -> after input_conv
					pre_gate_diff_t_n[d + feature_dims] = dh_t_n[d] * (h_t_1_n[d] - gate_t_n[d + 2 * feature_dims]);
					pre_gate_diff_t_n[d + feature_dims] *= d_hard_sigmoid(gate_t_n[d + feature_dims]);

					// Yujie: why should we do such work again? 
					//        in forward, this is done and not changed
					hidden_rt_data_n[d] = gate_t_n[d] * h_t_1_n[d];
					
					// Ht_candidate = W*Xt + U(Rt .* H[t-1])
					// gradient for W*Xt is equal to U(Rt .* H[t-1])
					// diff: Ht_candidate -> after_tmp_conv(hidden_reset_pre_gate)
					hidden_rt_pre_gate_diff_n[d] = pre_gate_diff_t_n[d + 2 * feature_dims];
				}
			}

			// hidden_reset_pre_gate -> hidden_reset_
			conv_tmp_hidden_layer_->Backward(conv_tmp_hidden_top_vec_, vector<bool>{true}, conv_tmp_hidden_bottom_vec_);

			for (int n = 0; n < N_; ++n) {
				Dtype* pre_gate_diff_t_n = pre_gate_diff_t + n * 3 * feature_dims;
				const Dtype* gate_t_n = gate_t + n * 3 * feature_dims;
				Dtype* dh_t_1_n = dh_t_1 + n * feature_dims;
				Dtype* h_t_1_n = h_t_1 + n * feature_dims;
				const Dtype* hidden_rt_diff_n = hidden_rt_diff + n * feature_dims;
				Dtype* hidden_pre_gate_diff_n = hidden_pre_gate_diff + n * feature_dims;

				for (int d = 0; d < feature_dims; ++d) {
					// diff: before_tmp_conv(Rt .* H[t-1]) -> H[t-1]
					dh_t_1_n[d] += hidden_rt_diff_n[d] * gate_t_n[d];
					
					// diff: Rt .* H[t-1] -> Rt -> after_input_conv
					pre_gate_diff_t_n[d] = hidden_rt_diff_n[d] * h_t_1_n[d];
					pre_gate_diff_t_n[d] *= d_hard_sigmoid(gate_t_n[d]);

					// data is sum relation between W[i]*Xt and U[i]*H[t-1]
					// so diffs are equal for W[i]*Xt and U[i]*H[t-1]
					// diff: Rt .* H[t-1] -> Rt -> after_hidden_conv
					hidden_pre_gate_diff_n[d] = pre_gate_diff_t_n[d];
					// diff: Zt -> after_hidden_conv
					hidden_pre_gate_diff_n[d + feature_dims] = pre_gate_diff_t_n[d + feature_dims];
				}
			}

			// Backprop output errors to the previous time step
			// hidden_pre_gate_(after_hidden_conv) -> hidden(H[t-1])
			hidden_.set_cpu_data(h_t_1);
			conv_hidden_layer_->Backward(conv_hidden_top_vec_, vector<bool>{true}, conv_hidden_bottom_vec_);
			const Dtype* hidden_diff_ = hidden_.cpu_diff();
			caffe_add(N_ * feature_dims, dh_t_1, hidden_diff_, dh_t_1);
		}

		// Gradient w.r.t. bottom data 
		// accumulated all diff from input_pre_gate(conv) -> btm data
		// At the same time, calc all gradient for Wr, Wz, W
		conv_input_layer_->Backward(conv_input_top_vec_, vector<bool>{propagate_down[0]}, conv_input_bottom_vec_);
	}

#ifdef CPU_ONLY
	STUB_GPU(ConvGRULayer);
#endif

	INSTANTIATE_CLASS(ConvGRULayer);
	REGISTER_LAYER_CLASS(ConvGRU);

}  // namespace caffe