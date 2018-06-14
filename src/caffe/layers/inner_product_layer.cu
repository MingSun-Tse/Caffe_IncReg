#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
  this->PruneForward(); /// @mingsuntse, for pruning
  
  const Dtype* weight = this->blobs_[0]->gpu_data();
  if (M_ == 1) {
    caffe_gpu_gemv<Dtype>(CblasNoTrans, N_, K_, (Dtype)1.,
                         weight, bottom_data, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_axpy<Dtype>(N_, bias_multiplier_.cpu_data()[0],
                            this->blobs_[1]->gpu_data(), top_data);
  } else {
    caffe_gpu_gemm<Dtype>(CblasNoTrans,
                          transpose_ ? CblasNoTrans : CblasTrans,
                          M_, N_, K_, (Dtype)1.,
                          bottom_data, weight, (Dtype)0., top_data);
    if (bias_term_)
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
                            bias_multiplier_.gpu_data(),
                            this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }
    // Restore weights when using ProbPrune
    if (this->IF_restore) {
        caffe_gpu_memcpy(this->blobs_[0]->count(),
                         this->blobs_backup_[0]->gpu_data(),
                         this->blobs_[0]->mutable_gpu_data());
    }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff    = top[0]->gpu_diff();
    // const Dtype* top_secdiff = top[0]->gpu_secdiff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    
    /*
    caffe_gpu_mul(bottom[0]->count(),
                  bottom_data,
                  bottom_data,
                  bottom[0]->mutable_gpu_secdata()); // x^2
    const Dtype* bottom_secdata = bottom[0]->gpu_secdata();
    */
    
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
      /*
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_secdata, top_secdiff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_secdiff()); /// @mingsuntse
      */
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
      /*
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_secdiff, bottom_secdata,
          (Dtype)1., this->blobs_[0]->mutable_gpu_secdiff()); /// @mingsuntse
      */
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff    = top[0]->gpu_diff();
    // const Dtype* top_secdiff = top[0]->gpu_secdiff();
    
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
    /*
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_secdiff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_secdiff()); // TODO(mingsuntse): Seems wrong, should be 'bias_multiplier_.gpu_data() * bias_multiplier_.gpu_data()'
    */
  }
  
  this->PruneBackward(top);
  
  if (propagate_down[0]) {
    const Dtype* top_diff    = top[0]->gpu_diff();
    // const Dtype* top_secdiff = top[0]->gpu_secdiff();
    /*
    caffe_gpu_mul(this->blobs_[0]->count(),
                  this->blobs_[0]->gpu_data(),
                  this->blobs_[0]->gpu_data(),
                  this->blobs_[0]->mutable_gpu_secdata()); // w^2
    */
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
      /*
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_secdiff, this->blobs_[0]->gpu_secdata(),
          (Dtype)0., bottom[0]->mutable_gpu_secdiff());
      */
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
      /*
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_secdiff, this->blobs_[0]->gpu_secdata(),
         (Dtype)0., bottom[0]->mutable_gpu_secdiff());
      */
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
