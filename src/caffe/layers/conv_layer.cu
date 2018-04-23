#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"

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
    const bool IF_prune = IF_want_prune && (IF_been_pruned || IF_enough_iter);
    
    if (this->phase_ == TRAIN && APP<Dtype>::inner_iter == 0) {
        // For a layer which doesn't want to prune, it still should UpdateNumPrunedCol/Row because of neighbour layer
        if (mthd != "None" && (IF_been_pruned || IF_enough_iter)) { 
            if (APP<Dtype>::IF_update_row_col && APP<Dtype>::IF_update_row_col_layer[L]) {
                // Note that, UpdateNumPrunedRow/Col before pruning, so that when calculating score, the zombie weights will not be counted.
                // The last conv and last fc layer need not updating num of pruned row.
                // In fact, the last conv should be updated row and the first fc should be updated col, but for simplicity, which are ignored for now.
                if (APP<Dtype>::prune_unit == "Col" && L != APP<Dtype>::conv_layer_cnt-1) { 
                    if (APP<Dtype>::step_-1 - APP<Dtype>::iter_prune_finished[L+1] <= 1) {
                        UpdateNumPrunedRow();
                    }
                } else if (APP<Dtype>::prune_unit == "Row" && mthd != "TP_Row" && APP<Dtype>::pruned_rows.size()) {
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
                    // if (APP<Dtype>::prune_coremthd.substr(0, 2) == "PP") { CleanWorkForPP(); } // last time, do some clean work
                    
                    // print when finished
                    char rlayer[10], rrow[10], rcol[10];
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
        if (L == APP<Dtype>::show_layer && APP<Dtype>::step_ % APP<Dtype>::show_interval == 0) {
            Print(L, 'f');
        }

        // Update masks and apply masks
        if (IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
            if (mthd == "FP_Row" && (APP<Dtype>::step_ - 1) % APP<Dtype>::prune_interval == 0) {
                FilterPrune(); 
            } else if (mthd == "PP_Col" && IF_hppf()) {
                ProbPruneCol(APP<Dtype>::prune_interval);
            } else if (mthd == "PP_Row" && IF_hppf()) {
                ProbPruneRow(APP<Dtype>::prune_interval);
            } else if (APP<Dtype>::prune_coremthd.substr(0, 3) == "Reg") {
                PruneMinimals();
            } else if (mthd == "PP-chl_Col" && IF_hppf()) {
                ProbPruneCol_chl(APP<Dtype>::prune_interval);
            } else {
                LOG(INFO) << "Wrong: unknown prune_method";
                exit(1);
            }
            UpdatePrunedRatio();
            if (L == APP<Dtype>::conv_layer_cnt - 1) { // To avoid the first fc from updating col
                APP<Dtype>::pruned_rows.clear();
            }
        }
        
        // Print weight magnitude
    if (APP<Dtype>::num_log > 0) {
        if (APP<Dtype>::prune_unit == "Col") {
            cout << "ave-magnitude_col " << APP<Dtype>::step_ << " " << layer_name << ":";
            for (int j = 0; j < num_col; ++j) {
                Dtype sum = 0;
                for (int i = 0; i < num_row; ++i) {
                    sum += fabs(muweight[i*num_col + j]);
                }
                cout << " " << sum;
            }
            cout << endl;
        } else if (APP<Dtype>::prune_unit == "Row") {
            cout << "ave-magnitude_row " << APP<Dtype>::step_ << " " << layer_name << ":";
            for (int i = 0; i < num_row; ++i) {
                Dtype sum = 0;
                for (int j = 0; j < num_col; ++j) {
                    sum += fabs(muweight[i*num_col + j]);
                }
                cout << " " << sum;
            }
            cout << endl;
        }
    }
        // Summary print 
        if (mthd != "None" && L < APP<Dtype>::show_num_layer) {
            cout << layer_name << "  IF_prune: " << IF_prune 
                 << "  pruned_ratio: " << APP<Dtype>::pruned_ratio[L];
            cout << "  pruned_ratio_row: " << APP<Dtype>::num_pruned_row[L] * 1.0 / num_row << "(" << APP<Dtype>::num_pruned_row[L] << ")"
                 << "  pruned_ratio_col: " << APP<Dtype>::num_pruned_col[L] * 1.0 / num_col << "(" << APP<Dtype>::num_pruned_col[L] << ")";
            cout << "  prune_ratio: "  << APP<Dtype>::prune_ratio[L] << endl;
        }
        
    } else if (this->phase_ == TEST && IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX && APP<Dtype>::prune_coremthd.substr(0, 2) == "PP") {
        if (APP<Dtype>::mask_generate_mechanism == "group-wise") {
            // use the old mask-generating mechanism
            const int num_unit = (APP<Dtype>::prune_unit == "Row") ? num_row : num_col;
            Dtype rands[num_unit];
            caffe_rng_uniform(num_unit, (Dtype)0, (Dtype)1, rands);
            for (int i = 0; i < count; ++i) {
                const int row_index = i / num_col;
                const int col_index = i % num_col;
                const bool cond1 = (APP<Dtype>::prune_unit == "Row") ? rands[row_index] < APP<Dtype>::history_prob[L][row_index]
                                                                     : rands[col_index] < APP<Dtype>::history_prob[L][col_index];
                const bool cond2 = (APP<Dtype>::prune_unit == "Row") ? !APP<Dtype>::IF_col_pruned[L][col_index][0]
                                                                     : !APP<Dtype>::IF_row_pruned[L][row_index];
                APP<Dtype>::masks[L][i] = (cond1 && cond2) ? 1 : 0;
                this->weight_backup[i] = muweight[i]; // backup weights
                muweight[i] *= APP<Dtype>::masks[L][i];
            }
        } else if (APP<Dtype>::mask_generate_mechanism == "element-wise") {
            // use the new mask-generating mechanism (1)
            Dtype rands[count/10];
            for (int i = 0; i < count; ++i) {
        if (i % (count/10) == 0) {
            caffe_rng_uniform(count/10, (Dtype)0, (Dtype)1, rands);
        }
                const int row_index = i / num_col;
                const int col_index = i % num_col;
                const bool cond1 = (APP<Dtype>::prune_unit == "Row") ? rands[i%(count/10)] < APP<Dtype>::history_prob[L][row_index]
                                                                     : rands[i%(count/10)] < APP<Dtype>::history_prob[L][col_index];
                const bool cond2 = (APP<Dtype>::prune_unit == "Row") ? !APP<Dtype>::IF_col_pruned[L][col_index][0]
                                                                     : !APP<Dtype>::IF_row_pruned[L][row_index];
                APP<Dtype>::masks[L][i] = (cond1 && cond2) ? 1 : 0;
                this->weight_backup[i] = muweight[i]; // backup weights
                muweight[i] *= APP<Dtype>::masks[L][i];
            }
        } else if (APP<Dtype>::mask_generate_mechanism == "channel-wise") {
            // new mask-generating mechanism (2)
            assert(APP<Dtype>::prune_unit != "Col");
            const int num = this->blobs_[0]->count(0, 2); // number of channel
            const int kernel_spatial_size = this->blobs_[0]->count(2);
            Dtype rands[num];
            caffe_rng_uniform(num, (Dtype)0, (Dtype)1, rands);
            for (int i = 0; i < count; ++i) {
                const int row_index = i / num_col;
                const int col_index = i % num_col;
                const int chl_index = i / kernel_spatial_size; // channel index
                const bool cond1 = rands[chl_index] < APP<Dtype>::history_prob[L][row_index];
                const bool cond2 = !APP<Dtype>::IF_col_pruned[L][col_index][0];
                APP<Dtype>::masks[L][i] = (cond1 && cond2) ? 1 : 0;
                this->weight_backup[i] = muweight[i]; // backup weights
                muweight[i] *= APP<Dtype>::masks[L][i];
            }
        } else {
            LOG(INFO) << "Wrong, unknown mask_generate_mechanism";
            exit(1);
        }
        this->IF_restore = true;
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
    
    /*
    // ProbPruneRow-2, use feature map to measure importance
    if (this->phase_ == TRAIN && APP<Dtype>::inner_iter == 0) {
        if (IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
            if (mthd == "PP-fm_Row") {
                ProbPruneRow_fm(top, APP<Dtype>::prune_interval);
            }
            UpdatePrunedRatio();
            if (L == APP<Dtype>::conv_layer_cnt - 1) { // To avoid the first fc from updating col
                APP<Dtype>::pruned_rows.clear();
            }
        }
    }
    */
    
    /*
    this->bottom_dim_: bottom feature map size, input
    this->top_dim_: top feature map size, output
    this->num_: batch size
    */
    
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
    if (APP<Dtype>::prune_method == "TP_Row" && (APP<Dtype>::step_ - 1) % APP<Dtype>::prune_interval == 0) {
        const bool IF_want_prune  = APP<Dtype>::prune_method != "None" && APP<Dtype>::prune_ratio[L] > 0;
        const bool IF_been_pruned = APP<Dtype>::pruned_ratio[L] > 0;
        const bool IF_enough_iter = APP<Dtype>::step_ >= APP<Dtype>::prune_begin_iter+1;
        const bool IF_prune = IF_want_prune && (IF_been_pruned || IF_enough_iter);
        if (IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
            TaylorPrune(top);
        }
    }
    
    // Print and check
    if (L == APP<Dtype>::show_layer && APP<Dtype>::step_ % APP<Dtype>::show_interval == 0 && APP<Dtype>::inner_iter == 0) {
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
