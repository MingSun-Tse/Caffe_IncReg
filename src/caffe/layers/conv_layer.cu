#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#define SHOW_INTERVAL 10
#define SHOW_NUM_LAYER 5
#define LAYER_PRINTED 0

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
    const string mthd = APP<Dtype>::prune_method;
    const int L = APP<Dtype>::layer_index[layer_name];
    this->IF_restore = false;
    
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
                if (APP<Dtype>::prune_unit == "Col" && L != APP<Dtype>::conv_layer_cnt-1) {
                    if (APP<Dtype>::step_-1 - APP<Dtype>::iter_prune_finished[L+1] <= 1) {
                        UpdateNumPrunedRow();
                    }
                } else if (APP<Dtype>::prune_unit == "Row" && mthd != "TP_Row" && L != 0 && APP<Dtype>::pruned_rows.size()) {
                    UpdateNumPrunedCol();
                } /// Note we don't update column for TP, because their method didn't mention this.
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
                    
                    char* mthd = new char[strlen(APP<Dtype>::prune_method.c_str()) + 1];
                    strcpy(mthd, APP<Dtype>::prune_method.c_str());
                    const string mthd_ = strtok(mthd, "_"); // mthd is like "Reg_Col", the first split is `Reg`
                    if (mthd_ == "SPP") { CleanWorkForPP(); } // last time, do some clean work
                    
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
        
        // Print and check, before update probs
        // put this outside, to print even when we do not prune
        if (L == LAYER_PRINTED && APP<Dtype>::step_ % SHOW_INTERVAL == 0 && APP<Dtype>::inner_iter == 0) {
            Print(L, 'f');
        }

        // Update masks and apply masks
        if (this->IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
            if (mthd == "FP" && (APP<Dtype>::step_ - 1) % APP<Dtype>::prune_interval == 0) {
                FilterPrune(); 
            } else if (mthd.substr(0, 3) == "PPc" && IF_hppf()) {
                if (APP<Dtype>::prune_interval) {
                    ProbPruneCol(APP<Dtype>::prune_interval);
                } else {
                    ProbPruneCol();
                }
            } else if (mthd == "PPr" && IF_hppf()) {
                ProbPruneRow();
            } else if (mthd == "Reg_Col" || mthd == "Reg_Weight") {
                PruneMinimals();
            }
            UpdatePrunedRatio();
            if (L == APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt - 1) {
                APP<Dtype>::pruned_rows.clear();
            }
        }
        
        
        // Print 
        if (mthd != "None" && L < SHOW_NUM_LAYER && APP<Dtype>::inner_iter == 0) {
            cout << layer_name << "  IF_prune: " << this->IF_prune 
                 << "  pruned_ratio: " << APP<Dtype>::pruned_ratio[L];
            if (mthd == "PPr" || mthd == "FP" || mthd == "TP") {
                cout << "  pruned_ratio_col: " << APP<Dtype>::num_pruned_col[L] * 1.0 / num_col << "(" << APP<Dtype>::num_pruned_col[L] << ")"
                     << "  pruned_ratio_row: " << APP<Dtype>::num_pruned_row[L] * 1.0 / num_row << "(" << APP<Dtype>::num_pruned_row[L] << ")";
            } else {
                cout << "  pruned_ratio_row: " << APP<Dtype>::num_pruned_row[L] * 1.0 / num_row << "(" << APP<Dtype>::num_pruned_row[L] << ")"
                     << "  pruned_ratio_col: " << APP<Dtype>::num_pruned_col[L] * 1.0 / num_col << "(" << APP<Dtype>::num_pruned_col[L] << ")";
            }
            cout << "  prune_ratio: "  << APP<Dtype>::prune_ratio[L] 
                 << "  reg: " << APP<Dtype>::reg_to_distribute[L] 
                 << "/" << ceil(APP<Dtype>::prune_ratio[L] * num_col) * APP<Dtype>::target_reg << endl; 
        }
        
        
        // Weight logging
        if (APP<Dtype>::num_log) {
            const int num_log = APP<Dtype>::log_index[L].size();
            for (int k = 0; k < num_log; ++k) {
                const int index = APP<Dtype>::log_index[L][k];
                Dtype sum = 0;
                for (int i = 0; i < num_row; ++i) {
                    sum += fabs(muweight[i * num_col + index]);
                }
                sum /= num_row;
                APP<Dtype>::log_weight[L][k].push_back(sum);
            }
        }
    } else {
        if (this->IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX && mthd.substr(0, 2) == "PP") {
            Dtype rands[num_col];
            caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
            for (int i = 0; i < count; ++i) {
                APP<Dtype>::masks[L][i] = rands[i % num_col] < APP<Dtype>::history_prob[L][i % num_col] ? 1 : 0; /// generate masks
            }              
            for (int i = 0; i < count; ++i) { 
                this->weight_backup[i] = muweight[i]; /// backup weights
            } 
            this->IF_restore = true;
            for (int i = 0; i < count; ++i) { 
                muweight[i] *= APP<Dtype>::masks[L][i]; /// apply masks
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
    if (!APP<Dtype>::IN_TEST && L == 0) {
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
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];

    /// Diff log
    if (APP<Dtype>::num_log) {
        const int num_log = APP<Dtype>::log_index[L].size();
        for (int i = 0; i < num_log; ++i) {
            const int index = APP<Dtype>::log_index[L][i];
            Dtype sum = 0;
            for (int r = 0; r < num_row; ++r) {
                sum += fabs(muweight_diff[r * num_col + index]);
            }
            sum /= num_row;
            APP<Dtype>::log_diff[L][i].push_back(sum);
        }
    }
    
    // TaylorPrune
    if (this->IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
        if (APP<Dtype>::prune_method == "TP" && (APP<Dtype>::step_ - 1) % APP<Dtype>::prune_interval == 0) {
            TaylorPrune(top);
        }
    }
    
    /// Print and check
    if (L == LAYER_PRINTED && APP<Dtype>::step_ % SHOW_INTERVAL == 0 && APP<Dtype>::inner_iter == 0) {
        Print(L, 'b');
    }
    
    if (APP<Dtype>::prune_method != "None" && APP<Dtype>::pruned_ratio[L] > 0) { 
        for (int j = 0; j < count; ++j) { 
            muweight_diff[j] *= APP<Dtype>::masks[L][j]; 
        }
        
        // Trying: update this to GPU code
        /* 
        caffe_gpu_mul(this->blobs_[0]->count(), 
                      this->blobs_[0]->gpu_diff(), 
                      &(APP<Dtype>::masks[L][0]), 
                      this->blobs_[0]->mutable_gpu_diff());
        
        cout << this->layer_param_.name() << " - weight_diff:" << endl;
        for (int j = 0; j < 20;  ++j) {
            cout << muweight_diff[j] << endl;
        }
        */
    }
/// ------------------------------------------------------------- 
  
  
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
