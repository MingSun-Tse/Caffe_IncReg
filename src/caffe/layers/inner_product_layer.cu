#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#define LAYER_PRINTED 3
#define SHOW_INTERVAL 10
#define SHOW_NUM_LAYER 5

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
    // ------------------------------------------------
    // Added by WANGHUAN for pruning
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const string layer_name = this->layer_param_.name();
    const string mthd = APP::prune_method;
    const int L = APP::layer_index[layer_name];
    
    /// IF_mask
    const bool IF_prune       = mthd != "None";
    const bool IF_enough_iter = (APP::step_ - 1) >= APP::prune_begin_iter;
    const bool IF_pruned      = APP::pruned_ratio[L] > 0;
    this->IF_mask             = IF_prune && (IF_enough_iter || IF_pruned);
    
    if (this->phase_ == TRAIN) {
        if (this->IF_mask) {
            // UpdateNumPrunedCol/Row
            // TODO
            UpdatePrunedRatio();
            
            // check if prune finished, get into here ONLY once
            if (APP::iter_prune_finished[L] == INT_MAX) {
                Dtype pruned_ratio;
                if (APP::prune_unit == "Weight")   { pruned_ratio = APP::pruned_ratio[L];     }
                else if (APP::prune_unit == "Row") { pruned_ratio = APP::pruned_ratio_row[L]; }
                else if (APP::prune_unit == "Col") { pruned_ratio = APP::pruned_ratio_col[L]; }
                const bool layer_finish     = pruned_ratio >= APP::prune_ratio[L]; /// layer pruning target achieved
                const bool net_finish_speed = APP::IF_speedup_achieved;   /// net pruning target of speed achieved
                const bool net_finish_param = APP::IF_compRatio_achieved; /// net pruning target of compression achieved
                
                if (layer_finish || net_finish_speed || net_finish_param) {
                    APP::iter_prune_finished[L] = APP::step_ - 1;

                    // print to log
                    char rlayer[10];
                    char rrow[10];
                    char rcol[10];
                    sprintf(rlayer, "%6.4f", APP::pruned_ratio[L]);
                    sprintf(rrow,   "%6.4f", APP::pruned_ratio_row[L]);
                    sprintf(rcol,   "%6.4f", APP::pruned_ratio_col[L]);
                    cout << layer_name << " prune finished!" 
                         << "  step: " << APP::step_
                         << "  net speedup: " << APP::speedup
                         << "  net compRatio: " << APP::compRatio
                         << "  pruned_ratio: " << rlayer
                         << "  pruned_ratio_row: " << rrow
                         << "  pruned_ratio_col: " << rcol 
                         << "  prune_ratio: " << APP::prune_ratio[L] << endl;
                    IF_alpf();
                }
            }
        }
        
        // Print, before masked
        if (L == LAYER_PRINTED && APP::step_ % SHOW_INTERVAL == 0 && APP::inner_iter == 0) {
            Print(L, 'f');
        }
        
        // Update masks
        if (this->IF_mask && APP::iter_prune_finished[L] == INT_MAX) {
            if (mthd == "Reg_Weight") {
                PruneMinimals();
            }
            UpdatePrunedRatio();
        }
        
        
        // After update, print current pruning state
        if (mthd != "None" && L < SHOW_NUM_LAYER && APP::inner_iter == 0) {
               cout << layer_name << "  IF_mask: " << this->IF_mask 
                 << "  pruned_ratio: " << APP::pruned_ratio[L] 
                 << "  prune_ratio: " << APP::prune_ratio[L] << endl;
        }
        
    }
  // ------------------------------------------------
  
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
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_gpu_diff());
    }
    // -------------------------------------------------
    const int count = this->blobs_[0]->count();
    const int L = APP::layer_index[this->layer_param_.name()];
    Dtype* muweight_diff = this->blobs_[0]->mutable_cpu_diff();
    
    // Print
    if (L == LAYER_PRINTED && APP::step_ % SHOW_INTERVAL == 0 && APP::inner_iter == 0) {
        Print(L, 'b');
    }
    
    for (int i = 0; i < count; ++i) {
        muweight_diff[i] *= APP::masks[L][i];
    }
    

    
    // -------------------------------------------------
    
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
          (Dtype)0., bottom[0]->mutable_gpu_diff());
    } else {
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
         (Dtype)1., top_diff, this->blobs_[0]->gpu_data(),
         (Dtype)0., bottom[0]->mutable_gpu_diff());
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
