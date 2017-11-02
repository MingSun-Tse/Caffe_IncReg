#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#define SHOW_INTERVAL 10
#define SHOW_NUM_LAYER 5

using namespace std;
namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
          
    /// ADDED BY WANGHUAN -----------------------------------
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    const string mthd = APP::prune_method;
    const int L = APP::layer_index[layer_name];
    this->IF_restore = false;
    
    
    /// IF_mask
    const bool IF_prune       = mthd != "None";
    const bool IF_enough_iter = (APP::step_ - 1) >= APP::prune_begin_iter;
    const bool IF_pruned      = APP::pruned_ratio[L] > 0;
    this->IF_mask             = IF_prune && (IF_enough_iter || IF_pruned);
    
    
    if (this->phase_ == TRAIN) {
        if (this->IF_mask) {
            
            // UpdateNumPrunedRow/Col
            // Note that, UpdateNumPrunedRow/Col before pruning, 
            // so that when calculating score, the zombie weights will not be counted.
            if ((mthd == "PPc" || mthd == "CP") && L != APP::layer_cnt-1) {
                if (APP::step_ - 1 - APP::iter_prune_finished[L + 1] <= 1) {
                    UpdateNumPrunedRow();
                }
            } else if ((mthd == "PPr" || mthd == "FP" || mthd == "TP") && L != 0) {
                UpdateNumPrunedCol();
            }
            UpdatePrunedRatio();
            
            // Given pruned_ratio, judge whether prune finished for current layer
            // Get into here ONLY once
            if (APP::iter_prune_finished[L] == INT_MAX) {
                const Dtype pruned_ratio = (mthd == "PPr" || mthd == "FP" || mthd == "TP") 
                                ? APP::pruned_ratio_row[L] : APP::pruned_ratio_col[L];
                const bool layer_finish = (pruned_ratio >= APP::prune_ratio[L]); /// layer pruning target achieved
                const bool net_finish   = APP::IF_speedup_achieved; /// net pruning target achieved
                if (layer_finish || net_finish) {
                    if (mthd.substr(0, 2) == "PP") { CleanWorkForPP(); } /// last time, do some clean work
                    APP::iter_prune_finished[L] = APP::step_ - 1;
                    cout << layer_name << " prune finished!" 
                         << "  step: " << APP::step_ 
                         << "  pruned_ratio: "     << APP::pruned_ratio[L]
                         << "  pruned_ratio_col: " << APP::pruned_ratio_col[L] << endl;
                    IF_alpf();
                }
            }
        }
        
        // Print and check
        if (mthd != "None" && L < SHOW_NUM_LAYER && APP::inner_iter == 0) {
            cout << layer_name << "  IF_mask: " << this->IF_mask 
                 << "  pruned_ratio: " << APP::pruned_ratio[L]
                 << "  prune_ratio: "  << APP::prune_ratio[L]
                 << "  num_pruned_col: " << APP::num_pruned_col[L]
                 << "  num_pruned_row: " << APP::num_pruned_row[L] << endl;
        }
        if (L == 1 && APP::step_ % SHOW_INTERVAL == 0 && APP::inner_iter == 0) {
            Print(L, 'f');
        }

        // Update masks and apply masks
        if (this->IF_mask && APP::iter_prune_finished[L] == INT_MAX) {
            if (mthd == "Prune" && APP::criteria == "L2-norm") { 
                /// UpdateMasks(); 
            } else if (mthd == "FP" && (APP::step_ - 1) % APP::prune_interval == 0) {
                FilterPrune(); 
            } else if (mthd == "PPc" && IF_hppf()) {
                ProbPruneCol();
            } else if (mthd == "PPr" && IF_hppf()) {
                ProbPruneRow();
            }  else if (mthd == "TP") {
                for (int i = 0; i < count; ++i) {
                    muweight[i] *= APP::masks[L][i]; /// explictly prune, because seems TP is wrong somewhere.
                }  
            }
        }
        
        // Weight logging
        if (APP::num_log) {
            const int num_log = APP::log_index[L].size();
            for (int k = 0; k < num_log; ++k) {
                const int index = APP::log_index[L][k];
                Dtype sum = 0;
                for (int i = 0; i < num_row; ++i) {
                    sum += fabs(muweight[i * num_col + index]);
                }
                sum /= num_row;
                APP::log_weight[L][k].push_back(sum);
            }
        }
    } else {
        if (this->IF_mask && APP::iter_prune_finished[L] == INT_MAX && mthd == "PP") {
            Dtype rands[num_col];
            caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
            for (int i = 0; i < count; ++i) {
                APP::masks[L][i] = rands[i % num_col] < APP::history_prob[L][i % num_col] ? 1 : 0; /// generate masks
            }              
            for (int i = 0; i < count; ++i) { 
                this->weight_backup[i] = muweight[i]; /// backup weights
            } 
            this->IF_restore = true;
            for (int i = 0; i < count; ++i) { 
                muweight[i] *= APP::masks[L][i]; /// apply masks
            } 
        }
    }
  /// ------------------------------------------------------
  
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        Dtype* top_data = top[i]->mutable_gpu_data();
        for (int n = 0; n < this->num_; ++n) {
            this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight,
            top_data + n * this->top_dim_);
            if (this->bias_term_) {
                const Dtype* bias = this->blobs_[1]->gpu_data();
                this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }
    /// this->bottom_dim_: bottom feature map size, input
    /// this->top_dim_: top feature map size, output
    /// this->num_: batch size
    
    /// Print feature map to check --------
    /// If row 3 and 8 are pruned in previous layer, then channel 3 and 8 will be only biases in this layer's feature map.
    /**
    if (!APP::IN_TEST && L == 0) {
        cout << "bottom.size(): " << bottom.size() << endl;
        for (int i = 0; i < bottom.size(); ++i) {
            const Dtype* top_data = top[i]->cpu_data();
            const int channel = top[i]->shape()[1];
            const int width   = top[i]->shape()[2];
            const int height  = top[i]->shape()[3];
            cout << "channel: " << channel << " " << width << " " <<  height << endl;
            
            vector<Dtype> sum(channel, 0);
            for (int c = 0; c < channel; ++c) {
                for (int w = 0 ; w < width; ++w) {
                    for (int h = 0; h < height; ++h) {
                        sum[c] += fabs(top_data[0 + c * width * height + w * height + h]);
                    }
                }
            }
            for (int c = 0; c < channel; ++c) {
                cout << sum[c] << "  ";
            }
            cout << endl;
        }
    }
    */
    /// -----------------------------------
    
    
    
    /// Restore weights ----------------
    if (this->IF_restore) {
        /// cout << layer_name << ": restore weights! " << endl;
        this->blobs_[0]->mutable_cpu_data();
        /// this->blobs_[0]->gpu_data(); 
        /// Interesting! If the above line is added, something like "control" seems to transfer from cpu to gpu. 
        /// Then modifying cpu weights won't affect their gpu counterparts.
        for (int i = 0; i < count; ++i) {
            muweight[i] = this->weight_backup[i];
        }
        
        /**
        /// ========================
        /// Chech restore
        cout << "weights from cpu:" << endl;
        for (int i = 0; i < 20; ++i) {
            cout << muweight[i] << " ";
        }
        cout << endl;

        Dtype weight_cpu[count];
        const Dtype* weight_gpu = this->blobs_[0]->gpu_data();
        cout << "weights copied from gpu:" << endl;
        cudaMemcpy(weight_cpu, weight_gpu, sizeof(Dtype) * count, cudaMemcpyDeviceToHost);
        for (int i = 0; i < 20; ++i) {
            cout << weight_cpu[i] << " ";
        }
        cout << endl;
        /// ========================
        */
    }
    /// --------------------------------
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
  
/// ADDED BY WANGHUAN ------------------------------------------
    Dtype* muweight_diff = this->blobs_[0]->mutable_cpu_diff();      
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int L = APP::layer_index[this->layer_param_.name()];

    /// Print and check
    if (L == 1 && APP::step_ % SHOW_INTERVAL == 0 && APP::inner_iter == 0) {
        Print(L, 'b');
    }
    
    /// Diff log
    if (APP::num_log) {
        const int num_log = APP::log_index[L].size();
        for (int i = 0; i < num_log; ++i) {
            const int index = APP::log_index[L][i];
            Dtype sum = 0;
            for (int r = 0; r < num_row; ++r) {
                sum += fabs(muweight_diff[r * num_col + index]);
            }
            sum /= num_row;
            APP::log_diff[L][i].push_back(sum);
        }
    }
    
    if (this->IF_mask) {
        for (int j = 0; j < count; ++j) { 
            muweight_diff[j] *= APP::masks[L][j]; 
        }
        if (APP::iter_prune_finished[L] == INT_MAX) {
            if (APP::prune_method == "Prune" && APP::criteria == "diff") {
            } else if (APP::prune_method == "TP" && (APP::step_ - 1) % APP::prune_interval == 0) {
                TaylorPrune(top);
            }
        }
    }
/// ------------------------------------------------------------- 
  
  
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
