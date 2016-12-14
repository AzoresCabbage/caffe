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
	Dtype d_sigmoid(const Dtype x) {
		return x * (Dtype(1) - x);
	}

	template <typename Dtype>
	Dtype tanh(const Dtype x) {
		return Dtype(2) * sigmoid(Dtype(2) * x) - Dtype(1);
	}

	template <typename Dtype>
	Dtype d_tanh(const Dtype x) {
		return Dtype(1) - x * x;
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
		spatial_dims_ = bottom[0]->count(2); //H*W
		num_output_ = this->layer_param_.convolution_param().num_output(); // number of hidden units channels
		const FillerParameter& weight_filler = this->layer_param_.convolution_param().weight_filler();
		const FillerParameter& bias_filler = this->layer_param_.convolution_param().bias_filler();


		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}		
		// n * (4*num_output) * h * w
		unit_shape[1] = 4 * num_output_;
		conv_h_top_t_.Reshape(unit_shape);

		// [0]: seq length
		// [1]: num_output
		// [2]: Height
		// [3]: Width
		unit_shape[1] = num_output_;
		gate_i_.Reshape(unit_shape);
		gate_f_.Reshape(unit_shape);
		gate_c_.Reshape(unit_shape);
		gate_o_.Reshape(unit_shape);
		wo_ct_tmp_.Reshape(unit_shape);


		// [0]: 1
		// [1]: num_output
		// [2]: Height
		// [3]: Width
		unit_shape[0] = 1;
		H_0_.Reshape(unit_shape);
		C_0_.Reshape(unit_shape);
		Wci_c_t_1_.Reshape(unit_shape);
		Wcf_c_t_1_.Reshape(unit_shape);
		conv_h_btm_blob_.Reshape(unit_shape);


		// for eltwise C[t-1] -> Wci .* C[t-1] , Wcf .* C[t-1] and Wco .* C[t]
		// [0]: 1
		// [1]: 3 * num_output
		// [2]: Height
		// [3]: Width
		unit_shape[1] = 3 * unit_shape[1];
		Wc_.Reshape(unit_shape);


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
		conv_x_param->set_bias_term(true); // for bi, bf, bc, bo

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
		conv_h_param->mutable_weight_filler()->CopyFrom(weight_filler);
		conv_h_param->mutable_bias_filler()->CopyFrom(bias_filler);
		conv_h_param->set_axis(1);
		conv_h_param->set_bias_term(false);
		// H[t-1] should contribute to input_gate(I), forget_gate(F), memory cell(C), output_gate(O)
		conv_h_param->set_num_output(num_output_ * 4);

		conv_h_btm_vec_.clear();
		conv_h_btm_vec_.push_back(&conv_h_btm_blob_);
		conv_h_top_vec_.clear();
		conv_h_top_vec_.push_back(&conv_h_top_blob_);
		conv_h_layer_.reset(new ConvolutionLayer<Dtype>(hidden_conv_param));
		conv_h_layer_->SetUp(conv_h_btm_vec_, conv_h_top_vec_);

		// fill Wci, Wcf, Wco
		shared_ptr<Filler<Dtype> > matrix_filler( GetFiller<Dtype>(weight_filler) );
		matrix_filler->Fill(&Wc_);

		// Check if we need to set up the weights
		if (this->blobs_.size() > 0) {
			LOG(INFO) << "Skipping parameter initialization";
			if (conv_x_layer_->blobs()[0]->shape() != this->blobs_[0]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[0]->shape()";
			if (conv_x_layer_->blobs()[1]->shape() != this->blobs_[1]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[1]->shape()";
			if (conv_h_layer_->blobs()[0]->shape() != this->blobs_[2]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[2]->shape()";
			if (Wc_.shape() != this->blobs_[3]->shape())
				LOG(ERROR) << "incompatible with this->blobs_[3]->shape()";

			conv_x_layer_->blobs()[0]->ShareData(*(this->blobs_[0]));
			conv_x_layer_->blobs()[1]->ShareData(*(this->blobs_[1]));
			conv_h_layer_->blobs()[0]->ShareData(*(this->blobs_[2]));
			Wc_.ShareData(*(this->blobs_[3]));

			conv_x_layer_->blobs()[0]->ShareDiff(*(this->blobs_[0]));
			conv_x_layer_->blobs()[1]->ShareDiff(*(this->blobs_[1]));
			conv_h_layer_->blobs()[0]->ShareDiff(*(this->blobs_[2]));
			Wc_.ShareDiff(*(this->blobs_[3]));
		}
		else {
			this->blobs_.resize(4);

			this->blobs_[0].reset(new Blob<Dtype>(conv_x_layer_->blobs()[0]->shape())); // weight for input conv
			this->blobs_[1].reset(new Blob<Dtype>(conv_x_layer_->blobs()[1]->shape())); // bias for input conv
			this->blobs_[2].reset(new Blob<Dtype>(conv_h_layer_->blobs()[0]->shape())); // weight for H[t-1] conv
			this->blobs_[3].reset(new Blob<Dtype>(Wc_.shape()));

			this->blobs_[0]->ShareData(*(conv_x_layer_->blobs()[0]));
			this->blobs_[1]->ShareData(*(conv_x_layer_->blobs()[1]));
			this->blobs_[2]->ShareData(*(conv_h_layer_->blobs()[0]));
			this->blobs_[3]->ShareData(Wc_);

			this->blobs_[0]->ShareDiff(*(conv_x_layer_->blobs()[0]));
			this->blobs_[1]->ShareDiff(*(conv_x_layer_->blobs()[1]));
			this->blobs_[2]->ShareDiff(*(conv_h_layer_->blobs()[0]));
			this->blobs_[3]->ShareDiff(Wc_);
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
		if (this->blobs_[3]->data() != Wc_.data()){
			LOG(INFO) << "share data/diff with blobs_[3]";
			Wc_.ShareData(*(this->blobs_[3]));
			Wc_.ShareDiff(*(this->blobs_[3]));
		}

		seq_len_ = bottom[0]->shape(0); // seq len
		spatial_dims_ = bottom[0]->count(2); //H*W

		vector<int> unit_shape;
		for (int i = 0; i < bottom[0]->shape().size(); ++i) {
			unit_shape.push_back(bottom[0]->shape()[i]);
		}
		// n * (4*num_output) * h * w
		unit_shape[1] = 4 * num_output_;
		conv_h_top_t_.Reshape(unit_shape);

		// [0]: seq length
		// [1]: num_output
		// [2]: Height
		// [3]: Width
		unit_shape[1] = num_output_;
		gate_i_.Reshape(unit_shape);
		gate_f_.Reshape(unit_shape);
		gate_c_.Reshape(unit_shape);
		gate_o_.Reshape(unit_shape);
		wo_ct_tmp_.Reshape(unit_shape);

		top[0]->Reshape(unit_shape);

		unit_shape[0] = 1;
		H_0_.Reshape(unit_shape);
		C_0_.Reshape(unit_shape);
		Wci_c_t_1_.Reshape(unit_shape);
		Wcf_c_t_1_.Reshape(unit_shape);
		conv_h_btm_blob_.Reshape(unit_shape);

		// for eltwise C[t-1] -> Wci .* C[t-1] , Wcf .* C[t-1] and Wco .* C[t]
		// [0]: 1
		// [1]: 3 * num_output
		// [2]: Height
		// [3]: Width
		unit_shape[1] = 3 * unit_shape[1];
		Wc_.Reshape(unit_shape);

		conv_x_layer_->Reshape(conv_x_btm_vec_, conv_x_top_vec_);
		conv_h_layer_->Reshape(conv_h_btm_vec_, conv_h_top_vec_);
	}

	template <typename Dtype>
	void ConvLSTMLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {

		Dtype* top_data = top[0]->mutable_cpu_data();
		int featmap_dim = spatial_dims_ * num_output_;

		caffe_set(H_0_.count(), Dtype(0.), H_0_.mutable_cpu_data());
		caffe_set(C_0_.count(), Dtype(0.), C_0_.mutable_cpu_data());

		// For all input X: X[t] -> Wxi*X[t], Wxf*X[t], Wxc*X[t], Wxo*X[t] in conv_x_top_blob_
		conv_x_layer_->Forward(conv_x_btm_vec_, conv_x_top_vec_);
		
		for (int t = 0; t < seq_len_; ++t)
		{
			Dtype* gate_i_t_data = gate_i_.mutable_cpu_data() + gate_i_.offset(t);
			Dtype* gate_f_t_data = gate_f_.mutable_cpu_data() + gate_f_.offset(t);
			Dtype* gate_c_t_data = gate_c_.mutable_cpu_data() + gate_c_.offset(t);
			Dtype* gate_o_t_data = gate_o_.mutable_cpu_data() + gate_o_.offset(t);
			
			const Dtype* X_w_t = conv_x_top_blob_.cpu_data() + conv_x_top_blob_.offset(t);
			const Dtype* x_wi_t = X_w_t;
			const Dtype* x_wf_t = X_w_t + featmap_dim * 1;
			const Dtype* x_wc_t = X_w_t + featmap_dim * 2;
			const Dtype* x_wo_t = X_w_t + featmap_dim * 3;
			
			Dtype* C_t_1 = t == 0 ? C_0_.mutable_cpu_data() : gate_c_t_data - gate_c_.count(1);
			//  for eltwise C[t-1] -> Wci .* C[t-1], Wcf .* C[t-1] in c_i_top_blob_ and c_f_top_blob_
			Dtype* wci_t_1 = Wci_c_t_1_.mutable_cpu_data();
			Dtype* wcf_t_1 = Wcf_c_t_1_.mutable_cpu_data();
			Dtype* wc_data = Wc_.mutable_cpu_data();
			Dtype* wci_data = wc_data;
			Dtype* wcf_data = wc_data + spatial_dims_;
			Dtype* wco_data = wc_data + spatial_dims_ * 2;
			caffe_mul(featmap_dim, C_t_1, wci_data, wci_t_1);
			caffe_mul(featmap_dim, C_t_1, wcf_data, wcf_t_1);


			Dtype* H_t = top_data + top[0]->offset(t);
			Dtype* H_t_1 = t == 0 ? H_0_.mutable_cpu_data() : top_data + top[0]->offset(t - 1);
			Dtype* H_w_t_1 = conv_h_top_blob_.mutable_cpu_data();
			Dtype* h_wi_t_1 = H_w_t_1;
			Dtype* h_wf_t_1 = H_w_t_1 + featmap_dim * 1;
			Dtype* h_wc_t_1 = H_w_t_1 + featmap_dim * 2;
			Dtype* h_wo_t_1 = H_w_t_1 + featmap_dim * 3;
			Dtype* wo_ct_t = wo_ct_tmp_.mutable_cpu_data() + wo_ct_tmp_.offset(t);

			// H[t-1] -> Whi*H[t-1], Whf*H[t-1], Whc*H[t-1], Who*H[t-1] in conv_h_top_blob_
			conv_h_btm_blob_.set_cpu_data(H_t_1);
			conv_h_layer_->Forward(conv_h_btm_vec_, conv_h_top_vec_);

			// cache Whc*H[t-1] for backward
			Dtype* wh_data = conv_h_top_t_.mutable_cpu_data() + conv_h_top_t_.offset(t);
			caffe_copy(conv_h_top_blob_.count(), H_w_t_1, wh_data);

			// I[t] = Wxi*X[t] + Whi*H[t-1] + Wci.*C[t-1] + bi(bias term of conv_x)
			caffe_add(featmap_dim, x_wi_t, h_wi_t_1, gate_i_t_data);
			caffe_add(featmap_dim, gate_i_t_data, wci_t_1, gate_i_t_data);

			// F[t] = Wxf*X[t] + Whf*H[t-1] + Wcf.*C[t-1] + bf(bias term of conv_x)
			caffe_add(featmap_dim, x_wf_t, h_wf_t_1, gate_f_t_data);
			caffe_add(featmap_dim, gate_f_t_data, wcf_t_1, gate_f_t_data);

			// C[t] = sigmoid(F[t]).*C[t-1] + sigmoid(I[t]).*tanh(Wxc*X[t] + Whc*H[t-1] + bc(bias term of conv_x))
			for (int i = 0; i < featmap_dim; ++i)
			{
				gate_c_t_data[i] = sigmoid(gate_f_t_data[i]) * C_t_1[i]
					+ sigmoid(gate_i_t_data[i]) * tanh(x_wc_t[i] + h_wc_t_1[i]);
			}

			// O[t] = (Wxo*X[t] + Who*H[t-1] + Wco .* C[t] + bo(bias term of conv_x))
			caffe_mul(featmap_dim, wco_data, gate_c_t_data, wo_ct_t);
			caffe_add(featmap_dim, x_wo_t, h_wo_t_1, gate_o_t_data);
			caffe_add(featmap_dim, gate_o_t_data, wo_ct_t, gate_o_t_data);

			// H[t] = sigmoid(O[t]) .* tanh(C[t])
			for (int i = 0; i < featmap_dim; ++i)
			{
				H_t[i] = sigmoid(gate_o_t_data[i]) * tanh(gate_c_t_data[i]);
			}
		}
	}

	template <typename Dtype>
	void ConvLSTMLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down,
		const vector<Blob<Dtype>*>& bottom) {
		Dtype* top_diff = top[0]->mutable_cpu_diff();
		Dtype* top_data = top[0]->mutable_cpu_data();
		int featmap_dim = spatial_dims_ * num_output_;

		caffe_set(H_0_.count(), Dtype(0.), H_0_.mutable_cpu_diff());
		caffe_set(C_0_.count(), Dtype(0.), C_0_.mutable_cpu_diff());

		for (int t = seq_len_ - 1; t >= 0; --t)
		{
			Dtype* gate_i_t_data = gate_i_.mutable_cpu_data() + gate_i_.offset(t);
			Dtype* gate_f_t_data = gate_f_.mutable_cpu_data() + gate_f_.offset(t);
			Dtype* gate_c_t_data = gate_c_.mutable_cpu_data() + gate_c_.offset(t);
			Dtype* gate_o_t_data = gate_o_.mutable_cpu_data() + gate_o_.offset(t);

			Dtype* gate_i_t_diff = gate_i_.mutable_cpu_diff() + gate_i_.offset(t);
			Dtype* gate_f_t_diff = gate_f_.mutable_cpu_diff() + gate_f_.offset(t);
			Dtype* gate_c_t_diff = gate_c_.mutable_cpu_diff() + gate_c_.offset(t);
			Dtype* gate_o_t_diff = gate_o_.mutable_cpu_diff() + gate_o_.offset(t);

			// diff: H[t] -> O[t] and C[t], note that H[t] = O[t].*tanh(C[t]), O[t] = sigmoid(...)
			// but gate_o here not perform sigmoid
			Dtype* H_t_diff = top_diff + top[0]->offset(t);
			for (int i = 0; i < featmap_dim; ++i)
			{
				gate_o_t_diff[i] = H_t_diff[i] * d_sigmoid(sigmoid(gate_o_t_data[i])) * tanh(gate_c_t_data[i]);
				gate_c_t_diff[i] = H_t_diff[i] * sigmoid(gate_o_t_data[i]) * d_tanh(tanh(gate_c_t_data[i]));
			}

			// conv X diff
			Dtype* X_w_t_diff = conv_x_top_blob_.mutable_cpu_diff() + conv_x_top_blob_.offset(t);
			Dtype* x_wi_t_diff = X_w_t_diff;
			Dtype* x_wf_t_diff = X_w_t_diff + featmap_dim * 1;
			Dtype* x_wc_t_diff = X_w_t_diff + featmap_dim * 2;
			Dtype* x_wo_t_diff = X_w_t_diff + featmap_dim * 3;
			// conv X data
			Dtype* X_w_t_data = conv_x_top_blob_.mutable_cpu_data() + conv_x_top_blob_.offset(t);
			Dtype* x_wc_t_data = X_w_t_data + featmap_dim * 2;

			// conv H data
			Dtype* H_w_t_1_data = conv_h_top_blob_.mutable_cpu_data();
			// restore conv_h_top_ at t
			caffe_copy(conv_h_top_blob_.count(), conv_h_top_t_.mutable_cpu_data() + conv_h_top_t_.offset(t), H_w_t_1_data);
			Dtype* h_wc_t_1_data = H_w_t_1_data + featmap_dim * 2;
			// conv H diff
			Dtype* H_w_t_1_diff = conv_h_top_blob_.mutable_cpu_diff();
			Dtype* h_wi_t_1_diff = H_w_t_1_diff;
			Dtype* h_wf_t_1_diff = H_w_t_1_diff + featmap_dim * 1;
			Dtype* h_wc_t_1_diff = H_w_t_1_diff + featmap_dim * 2;
			Dtype* h_wo_t_1_diff = H_w_t_1_diff + featmap_dim * 3;


			// diff: O[t] -> Wxo*X[t] and bo , Who*H[t-1] , Wco.*C[t]
			// diff: Wco.*C[t] -> C[t], Wco
			Dtype* wc_data = Wc_.mutable_cpu_data();
			Dtype* wci_data = wc_data;
			Dtype* wcf_data = wc_data + spatial_dims_;
			Dtype* wco_data = wc_data + spatial_dims_ * 2;
			Dtype* wc_diff = Wc_.mutable_cpu_diff();
			Dtype* wci_diff = wc_diff;
			Dtype* wcf_diff = wc_diff + spatial_dims_;
			Dtype* wco_diff = wc_diff + spatial_dims_ * 2;
			for (int i = 0; i < featmap_dim; ++i)
			{
				x_wo_t_diff[i] = gate_o_t_diff[i];
				h_wo_t_1_diff[i] = gate_o_t_diff[i];

				gate_c_t_diff[i] += gate_o_t_diff[i] * wco_data[i];
				wco_diff[i] += gate_o_t_diff[i] * gate_c_t_data[i];
			}

			// diff: C[t] -> F[t], C[t-1], I[t], Wxc*X[t] and bc, Whc*H[t-1]
			Dtype* C_t_1_diff = t == 0 ? C_0_.mutable_cpu_diff() : gate_c_t_diff - gate_c_.count(1);
			Dtype* C_t_1_data = t == 0 ? C_0_.mutable_cpu_data() : gate_c_t_data - gate_c_.count(1);
			for (int i = 0; i < featmap_dim; ++i)
			{
				gate_f_t_diff[i] = gate_c_t_diff[i] * C_t_1_data[i] * d_sigmoid(sigmoid(gate_f_t_data[i]));
				C_t_1_diff[i] = gate_c_t_diff[i] * sigmoid(gate_f_t_data[i]);
				gate_i_t_diff[i] = gate_c_t_diff[i] * d_sigmoid(sigmoid(gate_i_t_data[i])) * tanh(x_wc_t_data[i] + h_wc_t_1_data[i]);
				x_wc_t_diff[i] = gate_c_t_diff[i] * sigmoid(gate_i_t_data[i]) * d_tanh(tanh(x_wc_t_data[i] + h_wc_t_1_data[i]));
				
				h_wc_t_1_diff[i] = x_wc_t_diff[i];
			}

			// diff: I[t] -> Wxi*X[t] and bi, Whi*H[t-1], Wci*C[t-1]
			// diff: F[t] -> Wxf*X[t] and bf, Whf*H[t-1], Wcf*C[t-1]
			// diff: Wci*C[t-1] -> C[t-1], Wci
			// diff: Wcf*C[t-1] -> C[t-1], Wcf
			for (int i = 0; i < featmap_dim; ++i)
			{
				x_wi_t_diff[i] = gate_i_t_diff[i];
				h_wi_t_1_diff[i] = gate_i_t_diff[i];

				x_wf_t_diff[i] = gate_f_t_diff[i];
				h_wf_t_1_diff[i] = gate_f_t_diff[i];

				C_t_1_diff[i] += gate_i_t_diff[i] * wci_data[i] + gate_f_t_diff[i] * wcf_data[i];
				wci_diff[i] += gate_i_t_diff[i] * C_t_1_data[i];
				wcf_diff[i] += gate_f_t_diff[i] * C_t_1_data[i];
			}


			Dtype* H_t_1_diff = t == 0 ? H_0_.mutable_cpu_diff() : top_diff + top[0]->offset(t - 1);
			Dtype* H_t_1_data = t == 0 ? H_0_.mutable_cpu_data() : top_data + top[0]->offset(t - 1);

			conv_h_btm_blob_.set_cpu_data(H_t_1_data);
			conv_h_layer_->Backward(conv_h_top_vec_, vector<bool>{true}, conv_h_btm_vec_);

			// add conv_btm_blob to H[t-1] diff
			const Dtype* conv_h_t_1_diff = conv_h_btm_blob_.cpu_diff();
			caffe_add(conv_h_btm_blob_.count(), H_t_1_diff, conv_h_t_1_diff, H_t_1_diff);
		}
		conv_x_layer_->Backward(conv_x_top_vec_, vector<bool>{propagate_down[0]}, conv_x_btm_vec_);
	}

#ifdef CPU_ONLY
	STUB_GPU(ConvLSTMLayer);
#endif

	INSTANTIATE_CLASS(ConvLSTMLayer);
	REGISTER_LAYER_CLASS(ConvLSTM);

}  // namespace caffe