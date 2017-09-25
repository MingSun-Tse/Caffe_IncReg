#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#define SHOW_INTERVAL 10

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
    this->IF_restore = false;
    
    /// IF_mask
    const bool IF_prune       = APP::prune_method != "None";
    const bool IF_enough_iter = (APP::step_ - 1) >= APP::prune_begin_iter;
    const bool IF_pruned      = this->pruned_ratio > 0;
    const bool IF_mask        = IF_prune && (IF_enough_iter || IF_pruned);
    
    if (this->phase_ == TRAIN) {
        if (IF_mask) {
            // UpdateNumPrunedRow();
            // UpdateNumPrunedCol();
            this->pruned_ratio = 1 - (1 - this->num_pruned_col * 1.0 / num_col) * (1 - this->num_pruned_row * 1.0 / num_row);
            if (!APP::IF_prune_finished[this->layer_index]) {
                if (this->pruned_ratio >= this->prune_ratio) {
                    if (APP::prune_method == "PP") { CleanWorkForPP(); } // last time, do some clean work
                    APP::IF_prune_finished[this->layer_index] = true;
                    cout << layer_name << " prune finished!" 
                         << "  step: " << APP::step_ 
                         << "  pruned_ratio: " << this->pruned_ratio << endl;
                }
            }
        }
        
        /// Print and check
        if (APP::prune_method != "None" && this->layer_index < 5 && APP::inner_iter == 0) {
            cout << layer_name << "  IF_mask: " << IF_mask 
                 << "  pruned_ratio: " << this->pruned_ratio
                 << "  prune_ratio: " << this->prune_ratio 
                 << "  num_pruned_col: " << this->num_pruned_col
                 << "  num_pruned_row: " << this->num_pruned_row << endl;
        }
        
        /// Print and check (before pruning)
        if (this->layer_index == 1 && APP::step_ % SHOW_INTERVAL == 0 && APP::inner_iter == 0) {
            /// cout.setf(std::ios::left);
            cout.width(5);  cout << "Index" << "   ";
            cout.width(18); cout << "WeightBeforeMasked" << "   ";
            cout.width(4);  cout << "Mask" << "   ";
            cout.width(4);  cout << "Prob" << endl;
            for (int i = 0; i < 20; ++i) {
                cout.width(3);  cout << "#";
                cout.width(2);  cout << i+1 << "   ";
                cout.width(18); cout << muweight[i] << "   ";
                cout.width(4);  cout << this->masks_[i] << "   ";
                cout.width(4);  cout << APP::history_prob[this->layer_index][i] << endl;
            }
        }

        /// Update masks and apply masks
        if (IF_mask && this->pruned_ratio < this->prune_ratio) {
            if (APP::prune_method == "Prune" && APP::criteria == "L2-norm") { 
                /// UpdateMasks(); 
            } else if (APP::prune_method == "FP") {
                CHECK_GE(APP::prune_interval, 1)
                        << "Error: if 'FP' is used, 'prune_interval' must be set.";
                if ((APP::step_ - 1) % APP::prune_interval == 0) { FilterPrune(); }    
            } else if (APP::prune_method == "PP") {
                bool IF_hppf = true; /// IF_higher_priority_prune_finished 
                for (int i = 0; i <= APP::layer_cnt[0]; ++i) {
                    if (APP::priority[i] < APP::priority[this->layer_index] && !APP::IF_prune_finished[i]) {
                        IF_hppf = false;
                        break;
                    }
                }
                if (IF_hppf) { ProbPrune(); }
            }  else if (APP::prune_method == "TP") {
                for (int i = 0; i < count; ++i) {
                    muweight[i] *= this->masks_[i]; 
                }  // explictly prune, because seems TP is wrong somewhere.
            }
            
        }  
        bool IF_log = true;
        if (IF_log) {
            const int num_log = APP::log_index[this->layer_index].size();
            for (int i = 0; i < num_log; ++i) {
                const int index = APP::log_index[this->layer_index][i];
                Dtype sum = 0;
                for (int i = 0; i < num_row; ++i) {
                    sum += fabs(muweight[i * num_col + index]);
                }
                sum /= num_row;
                APP::log_weight[this->layer_index][i].push_back(sum);
            }
        }
    } else {
        if (APP::prune_method == "PP") {
            Dtype rands[num_col];
            caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
            for (int i = 0; i < count; ++i) {
                this->masks_[i] = rands[i % num_col] < APP::history_prob[this->layer_index][i % num_col] ? 1 : 0; /// geerate masks
            }              
            for (int i = 0; i < count; ++i) { this->weight_backup[i] = muweight[i]; } /// backup weights
            this->IF_restore = true;
            for (int i = 0; i < count; ++i) { muweight[i] *= this->masks_[i]; } /// do pruning
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
    if (!APP::IN_TEST && this->layer_index == 0) {
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
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    
    /// UpdateDiffs(); /// update second diff and so on

    /// Print and check
    if (this->layer_index == 1 && APP::step_ % SHOW_INTERVAL == 0 && APP::inner_iter == 0) {
        cout.width(5);  cout << "Index" << "   ";
        cout.width(16); cout << "DiffBeforeMasked" << "   ";
        cout.width(4);  cout << "Mask" << "   ";
        cout.width(4);  cout << "Prob" << endl;
        for (int i = 0; i < 20; ++i) {
            cout.width(3);  cout << "#";
            cout.width(2);  cout << i+1 << "   ";
            cout.width(16); cout << muweight_diff[i] << "   ";
            cout.width(4);  cout << this->masks_[i] << "   ";
            cout.width(4);  cout << APP::history_prob[this->layer_index][i] << endl;
        }
    }
    
    /// Diff log
    bool IF_log = true;
    if (IF_log) {
        const int num_log = APP::log_index[this->layer_index].size();
        for (int i = 0; i < num_log; ++i) {
            const int index = APP::log_index[this->layer_index][i];
            Dtype sum = 0;
            for (int r = 0; r < num_row; ++r) {
                sum += fabs(muweight_diff[r * num_col + index]);
            }
            sum /= num_row;
            APP::log_diff[this->layer_index][i].push_back(sum);
        }
    }
    

    /// IF_mask
    const bool IF_prune       = APP::prune_method != "None";
    const bool IF_enough_iter = (APP::step_ - 1) >= APP::prune_begin_iter;
    const bool IF_pruned      = this->pruned_ratio > 0;
    const bool IF_mask        = IF_prune && (IF_enough_iter || IF_pruned) ;
    if (IF_mask) {
        for (int j = 0; j < count; ++j) { muweight_diff[j] *= this->masks_[j]; }
        if (this->pruned_ratio < this->prune_ratio) {
            if (APP::prune_method == "Prune" && APP::criteria == "diff") {
                /// UpdateMasks(); 
            } else if (APP::prune_method == "TP") {
                CHECK_GE(APP::prune_interval, 1)
                    << "Error: if 'TP' is used, 'prune_interval' must be set.";
                if ((APP::step_ - 1) % APP::prune_interval == 0) { TaylorPrune(top); }
            }
        }
    }
    


/// ------------------------------------------------------------- 
  
  
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
