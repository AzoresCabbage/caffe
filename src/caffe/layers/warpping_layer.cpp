#include <vector>

#include "caffe/layers/warping_layer.hpp"

namespace caffe {

	template <typename Dtype>
	void WarpingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		top[0]->ReshapeLike(*bottom[0]);
		CHECK(bottom[1]->shape(0) == bottom[0]->shape(0)) << "Batch size should be same";
		CHECK(bottom[1]->shape(1) == 2) << "Displacement dims should be 2";
		CHECK(bottom[1]->shape(2) == bottom[0]->shape(2) && bottom[1]->shape(3) == bottom[0]->shape(3)) << "Spatial dims should be same";
	}

	template <typename Dtype>
	void WarpingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		Dtype* top_data = top[0]->mutable_cpu_data();
		const Dtype* Displacement_data = bottom[1]->cpu_data();

		int count = bottom[0]->count();
		int num = bottom[0]->shape(0);
		int channels = bottom[0]->shape(1);
		int height = bottom[0]->shape(2);
		int width = bottom[0]->shape(3);
		int spatial_dim = height * width;

		bool pad_zero = this->layer_param_.warping_param().pad_method() == WarpingParameter_PadMethod_ZERO;

		for (int index = 0; index < count; ++index) {
			int cn = index / spatial_dim;
			int n = cn / channels; //num

			int s = index % spatial_dim;
			int h = s / width;
			int w = s % width;


			Dtype botx = w + Displacement_data[n * 2 * spatial_dim + h * width + w];
			Dtype boty = h + Displacement_data[n * 2 * spatial_dim + spatial_dim + h * width + w];

			if (!pad_zero){
				botx = max(min(botx, Dtype(width - 1)), Dtype(0));
				boty = max(min(boty, Dtype(height - 1)), Dtype(0));
			}

			int botidxoffcn = cn * spatial_dim;

			float accum_value = 0;

			int floor_botx = floor(botx);
			int floor_boty = floor(boty);
			for (int by = floor_boty; by <= floor_boty + 1; by++)  {
				for (int bx = floor_botx; bx <= floor_botx + 1; bx++)  {
					if (bx < 0 || bx >(width - 1) || by < 0 || by >(height - 1))
						continue;
					float weight = (1.0f - abs((float)bx - botx)) * (1.0f - abs((float)by - boty));
					float sample = bottom_data[botidxoffcn + by * width + bx];
					accum_value += sample * weight;
				}
			}
			top_data[index] = accum_value;
		}
	}

	template <typename Dtype>
	void WarpingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		const Dtype* bottom_data = bottom[0]->cpu_data();
		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* data_diff = bottom[0]->mutable_cpu_diff();

		int count = bottom[0]->count();
		int num = bottom[0]->shape(0);
		int channels = bottom[0]->shape(1);
		int height = bottom[0]->shape(2);
		int width = bottom[0]->shape(3);

		int spatial_dim = height * width;

		const Dtype*  Displacement_data = bottom[1]->cpu_data();
		Dtype* Displacement_diff = bottom[1]->mutable_cpu_diff();

		caffe_set(bottom[0]->count(), Dtype(0), data_diff);
		caffe_set(bottom[1]->count(), Dtype(0), Displacement_diff);

		if (!propagate_down[0] && !propagate_down[1])
			return;

		bool pad_zero = this->layer_param_.warping_param().pad_method() == WarpingParameter_PadMethod_ZERO;

		for (int index = 0; index < num * height * width; ++index) {

			int n = index / spatial_dim;
			int s = index % spatial_dim;
			int h = s / width;
			int w = s % width;

			Dtype botx = w + Displacement_data[n * 2 * spatial_dim + h * width + w];
			Dtype boty = h + Displacement_data[n * 2 * spatial_dim + spatial_dim + h * width + w];

			if (!pad_zero){
				botx = max(min(botx, Dtype(width - 1)), Dtype(0));
				boty = max(min(boty, Dtype(height - 1)), Dtype(0));
			}

			int floor_botx = floor(botx);
			int floor_boty = floor(boty);
			for (int by = floor_boty; by <= floor_boty + 1; by++)  {
				for (int bx = floor_botx; bx <= floor_botx + 1; bx++)  {
					if (bx < 0 || bx >(width - 1) || by < 0 || by >(height - 1))
						continue;
					float weightx = 1.0f - abs((float)bx - botx);
					float weighty = 1.0f - abs((float)by - boty);

					for (int c = 0; c < channels; c++) {
						int botidxoffcn = (n * channels + c) * spatial_dim;

						if (propagate_down[0])
						{
							*(data_diff + botidxoffcn + by * width + bx) += top_diff[botidxoffcn + h * width + w] * weightx * weighty;
						}
						
						if (!propagate_down[1])
							continue;

						Dtype diff_sample = top_diff[botidxoffcn + h * width + w] * bottom_data[botidxoffcn + by * width + bx];
						if (pad_zero || (botx > Dtype(0) && botx < Dtype(width - 1))){
							if (bx > botx)
								Displacement_diff[n * 2 * spatial_dim + h * width + w] += diff_sample * weighty;
							else
								Displacement_diff[n * 2 * spatial_dim + h * width + w] -= diff_sample * weighty;
						}
						if (pad_zero || (boty > Dtype(0) && boty < Dtype(height - 1))){
							if (by > boty)
								Displacement_diff[n * 2 * spatial_dim + spatial_dim + h * width + w] += diff_sample * weightx;
							else
								Displacement_diff[n * 2 * spatial_dim + spatial_dim + h * width + w] -= diff_sample * weightx;
						}
					}
				}
			}
		}
	}


#ifdef CPU_ONLY
	STUB_GPU(WarpingLayer);
#endif

	INSTANTIATE_CLASS(WarpingLayer);
	REGISTER_LAYER_CLASS(Warping);

}  // namespace caffe
