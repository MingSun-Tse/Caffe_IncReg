#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) {
      loss[index] = 0;
      counts[index] = 0;
    } else {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s],
                      Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_); // Calculate prob_
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  const int dim = prob_.count() / outer_num_;
  const int nthreads = outer_num_ * inner_num_;
  // Since this memory is not used for anything until it is overwritten
  // on the backward pass, we use it here to avoid having to allocate new GPU
  // memory to accumulate intermediate results in the kernel.
  Dtype* loss_data = bottom[0]->mutable_gpu_diff();
  // Similarly, this memory is never used elsewhere, and thus we can use it
  // to avoid having to allocate additional GPU memory.
  Dtype* counts = prob_.mutable_gpu_diff();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>>(nthreads, prob_data, label, loss_data,
      outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts); // Calculate cross entropy loss
  Dtype loss;
  caffe_gpu_asum(nthreads, loss_data, &loss); // Get total loss
  Dtype valid_count = -1;
  // Only launch another CUDA kernel if we actually need the count of valid
  // outputs.
  if (normalization_ == LossParameter_NormalizationMode_VALID &&
      has_ignore_label_) {
    caffe_gpu_asum(nthreads, counts, &valid_count);
  }
  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_,
                                                        valid_count);
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
__global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, Dtype* counts) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) { /// @mingsuntse, nthreads = batch_size, here 
    const int n = index / spatial_dim; // n = {0, 1, ..., batch_size - 1}
    const int s = index % spatial_dim; // s = 0 here
    const int label_value = static_cast<int>(label[n * spatial_dim + s]); // n * spatial_dim + s = index, why do not use index directly?
    if (has_ignore_label_ && label_value == ignore_label_) {
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
      counts[index] = 0;
    } else {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
      
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff(); // bottom[0] is logits, bottom[1] is labels
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_gpu_memcpy(prob_.count() * sizeof(Dtype), prob_data, bottom_diff); // Initialize to p, note that, dLoss/dzi = pi - qi
    
    /*
    // second diff of logits = pi + pi * pi
    caffe_gpu_mul(prob_.count(),
                  prob_data,
                  prob_data,
                  bottom[0]->mutable_gpu_secdiff());
    caffe_gpu_add(prob_.count(),
                  prob_data,
                  bottom[0]->gpu_secdiff(),
                  bottom[0]->mutable_gpu_secdiff()); /// @mingsuntse
    */              
    const Dtype* label = bottom[1]->gpu_data();
    const int dim = prob_.count() / outer_num_;
    const int nthreads = outer_num_ * inner_num_;
    // Since this memory is never used for anything else,
    // we use to to avoid allocating new GPU memory.
    Dtype* counts = prob_.mutable_gpu_diff();
    // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, top_data, label, bottom_diff,
        outer_num_, dim, inner_num_, has_ignore_label_, ignore_label_, counts); // get bottom_diff
        
    /* Check
    const Dtype* label_cpu = bottom[1]->cpu_data();
    std::cout << nthreads
          << " " << prob_.count()      // = batch_size * num_class, bottom[0] has the same dimension
          << " " << bottom[1]->count() // = batch_size, because bottom[1] is labels, each exampple has one label, whose range = {0, num_class - 1}
          << " " << outer_num_         // = batch_size
          << " " << dim                // = num_class
          << " " << inner_num_         // spatial_dim, = 1 here. Any occasions that inner_num_ != 1 ?
          << std::endl;
    for (int i = 0; i < bottom[1]->count(); ++i) {
        std::cout << label_cpu[i] << " ";
    }
    std::cout << std::endl;
    */
        
    Dtype valid_count = -1;
    // Only launch another CUDA kernel if we actually need the count of valid
    // outputs.
    if (normalization_ == LossParameter_NormalizationMode_VALID &&
        has_ignore_label_) {
      caffe_gpu_asum(nthreads, counts, &valid_count);
    }
    const Dtype loss_weight = top[0]->cpu_diff()[0] / get_normalizer(normalization_, valid_count); // normalizer = bacth_size for convnet exp
    // std::cout << "valid_count: " << valid_count // -1 for convnet exp
               // << "  loss_weight: " << loss_weight << std::endl; 
    // loss_weight is the multiplier for a loss term in total loss, usually used in the multi-loss task
    caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);
    // caffe_gpu_scal(prob_.count(), loss_weight , bottom[0]->mutable_gpu_secdiff()); /// @mingsuntse
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossLayer);

}  // namespace caffe
