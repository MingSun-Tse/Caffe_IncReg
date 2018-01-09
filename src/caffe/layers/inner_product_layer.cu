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
    const string mthd = APP<Dtype>::prune_method;
    const int L = APP<Dtype>::layer_index[layer_name];
    
    /// IF_prune
    const bool IF_want_prune  = mthd != "None" && APP<Dtype>::prune_ratio[L] > 0; // if you want to prune, you must specify a meaningful prune_method and give a positive prune_ratio
    const bool IF_been_pruned = APP<Dtype>::pruned_ratio[L] > 0; // for a pruned layer, continue to prune
    const bool IF_enough_iter = APP<Dtype>::step_ >= APP<Dtype>::prune_begin_iter+1; // for a raw layer, if iter is enough, then prune
    this->IF_prune = IF_want_prune && (IF_been_pruned || IF_enough_iter);
    
    if (this->phase_ == TRAIN) {
        if (this->IF_prune) {
            if (APP<Dtype>::IF_update_row_col) {
                // UpdateNumPrunedRow/Col
                // Note that, UpdateNumPrunedRow/Col before pruning, 
                // so that when calculating score, the zombie weights will not be counted.
                if (APP<Dtype>::prune_unit == "Col" && L != APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt -1) {
                    if (APP<Dtype>::step_-1 - APP<Dtype>::iter_prune_finished[L+1] <= 1) {
                        UpdateNumPrunedRow();
                    }
                } else if (APP<Dtype>::prune_unit == "Row" && mthd != "TP_Row" && APP<Dtype>::pruned_rows.size()) {
                    UpdateNumPrunedCol();
                }
                UpdatePrunedRatio();
            }
            
            // check if prune finished, get into here ONLY once
            if (APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
                Dtype pruned_ratio;
                if (APP<Dtype>::prune_unit == "Weight")   { pruned_ratio = APP<Dtype>::pruned_ratio[L];     }
                else if (APP<Dtype>::prune_unit == "Row") { pruned_ratio = APP<Dtype>::pruned_ratio_row[L]; }
                else if (APP<Dtype>::prune_unit == "Col") { pruned_ratio = APP<Dtype>::pruned_ratio_col[L]; }
                const bool layer_finish     = pruned_ratio >= APP<Dtype>::prune_ratio[L]; /// layer pruning target achieved
                const bool net_finish_speed = APP<Dtype>::IF_speedup_achieved;   /// net pruning target of speed achieved
                const bool net_finish_param = APP<Dtype>::IF_compRatio_achieved; /// net pruning target of compression achieved
                
                if (layer_finish || net_finish_speed || net_finish_param) {
                    APP<Dtype>::iter_prune_finished[L] = APP<Dtype>::step_ - 1;

                    // print to log
                    char rlayer[10];
                    char rrow[10];
                    char rcol[10];
                    sprintf(rlayer, "%6.4f", APP<Dtype>::pruned_ratio[L]);
                    sprintf(rrow,   "%6.4f", APP<Dtype>::pruned_ratio_row[L]);
                    sprintf(rcol,   "%6.4f", APP<Dtype>::pruned_ratio_col[L]);
                    cout << layer_name << " prune finished!" 
                         << "  step: " << APP<Dtype>::step_
                         << "  net speedup: " << APP<Dtype>::speedup
                         << "  net compRatio: " << APP<Dtype>::compRatio
                         << "  pruned_ratio: " << rlayer
                         << "  pruned_ratio_row: " << rrow
                         << "  pruned_ratio_col: " << rcol 
                         << "  prune_ratio: " << APP<Dtype>::prune_ratio[L] << endl;
                    IF_alpf();
                }
            }
        }
        
        // Print, before masked
        if (L == LAYER_PRINTED && APP<Dtype>::step_ % SHOW_INTERVAL == 0 && APP<Dtype>::inner_iter == 0) {
            Print(L, 'f');
        }
        
        // Update masks
        if (this->IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
            if (APP<Dtype>::prune_coremthd == "Reg") {
                PruneMinimals();
            }
            UpdatePrunedRatio();
            if (L == APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt - 1) {
                APP<Dtype>::pruned_rows.clear();
            }
        }
        
        
        // After update, print current pruning state
        if (mthd != "None" && L < SHOW_NUM_LAYER && APP<Dtype>::inner_iter == 0) {
               cout << layer_name << "  IF_prune: " << this->IF_prune 
                 << "  pruned_ratio: " << APP<Dtype>::pruned_ratio[L] 
                 << "  prune_ratio: " << APP<Dtype>::prune_ratio[L] << endl;
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
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    if (APP<Dtype>::prune_method != "None" && APP<Dtype>::pruned_ratio[L] > 0) {
        // Print
        if (L == LAYER_PRINTED && APP<Dtype>::step_ % SHOW_INTERVAL == 0 && APP<Dtype>::inner_iter == 0) {
            Print(L, 'b');
        }
        
        const int count = this->blobs_[0]->count();
        Dtype* muweight_diff = this->blobs_[0]->mutable_cpu_diff();
        for (int i = 0; i < count; ++i) {
            muweight_diff[i] *= APP<Dtype>::masks[L][i];
        }
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
