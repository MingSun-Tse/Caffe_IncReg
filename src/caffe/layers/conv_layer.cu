#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/deep_compression.hpp"
#define SHOW_INTERVAL 20

using namespace std;

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // ADDED BY WANGHUAN -----------------------------------
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    if (this->layer_index > DeepCompression::max_layer_index) { DeepCompression::max_layer_index = this->layer_index; }
    const int num_col_to_prune  = ceil( this->prune_ratio                * num_col); // true pruning goal
    const int num_col_to_prune_ = ceil((this->prune_ratio + this->delta) * num_col); // a little bit higher goal
    const bool IF_mask =  DeepCompression::IN_RETRAIN 
                                 || (DeepCompression::step_ - 1) >= DeepCompression::prune_begin_iter;
    vector<Dtype> weight_backup(count, 0);
    bool IF_RESTORE = false;
    
    // Check -------------------------------------------
    /*
    if (!DeepCompression::IN_TEST && this->layer_index == 0) {
        for (int j = 0; j < num_col; ++j) { muweight[1  * num_col + j] = 0; }
        for (int j = 0; j < num_col; ++j) { muweight[3  * num_col + j] = 0; }
        for (int j = 0; j < num_col; ++j) { muweight[12 * num_col + j] = 0; }
        for (int j = 0; j < num_col; ++j) { muweight[16 * num_col + j] = 0; }
        for (int j = 0; j < num_col; ++j) { muweight[24 * num_col + j] = 0; }
        for (int j = 0; j < num_col; ++j) { muweight[27 * num_col + j] = 0; }
        // for (int j = 0; j < num_col; ++j) { muweight[31 * num_col + j] = 0; }
        // for (int j = 0; j < num_col; ++j) { muweight[37 * num_col + j] = 0; }
        // for (int j = 0; j < num_col; ++j) { muweight[42 * num_col + j] = 0; }
    }
    */
    // -------------------------------------------------
    
    if (!DeepCompression::IN_TEST) {
        // UpdateNumPrunedRow();
        UpdateNumPrunedCol();
        const Dtype pruned_ratio = 1 - (1 - this->num_pruned_column * 1.0 / num_col) * (1 - this->num_pruned_row * 1.0 / num_row);
        // Print and check
        cout << layer_name 
             << "  IF_mask: " << IF_mask 
             << "  pruned_ratio: ";
        cout.width(3); cout << pruned_ratio
                            << "  prune_ratio: " << this->prune_ratio
                            << endl;
        // Update masks and apply masks
        if (IF_mask && pruned_ratio < this->prune_ratio) {
            // Print and check (before pruning)
            if (this->layer_index == 1 && DeepCompression::step_ % SHOW_INTERVAL == 0) {
                // cout.setf(std::ios::left);
                cout.width(5);  cout << "Index" << "   ";
                cout.width(18); cout << "WeightBeforeMasked" << "   ";
                cout.width(4);  cout << "Mask" << "   ";
                cout.width(4);  cout << "Prob" << endl;
                for (int i = 0; i < 20; ++i) {
                    cout.width(3);  cout << "#";
                    cout.width(2);  cout << i+1 << "   ";
                    cout.width(18); cout << muweight[i] << "   ";
                    cout.width(4);  cout << this->masks_[i] << "   ";
                    cout.width(4);  cout << DeepCompression::history_prob[this->layer_index][i] << endl;
                }
            }             
            if (DeepCompression::prune_method == "PPruning" && DeepCompression::criteria == "energy") { 
                //UpdateMasks(); 
            } else if (DeepCompression::prune_method == "PFilter") {
                CHECK_GE(DeepCompression::prune_interval, 1)
                        << "Error: if 'Pfilter' is used, 'prune_interval' must be set.";
                if ((DeepCompression::step_ - 1) % DeepCompression::prune_interval == 0) {
                     UpdateMasks_pfilter();
                }
            } else if (DeepCompression::prune_method == "SPP") {
                // Recover the best columns, according to some probabilities
                Dtype p_recover;
                caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_recover);
                if (pow(15 + 0.00027 * DeepCompression::step_, -1.3) > p_recover) { // 40
                    // Sort
                    typedef std::pair<Dtype, int> mypair;
                    vector<mypair> col_score(num_col);
                    for (int j = 0; j < num_col; ++j) {
                        col_score[j].second = j; // index
                        col_score[j].first = 0;  // score
                        for (int i = 0; i < num_row; ++i) {
                            col_score[j].first += fabs(muweight[i * num_col +j]);
                        }
                        if (this->is_pruned[j]) { col_score[j].first = INT_MAX; } // make the pruned columns "float" up
                    } 
                    sort(col_score.begin(), col_score.end());
                    // Print and check
                    cout << "recover prob: " << layer_name << " step: " << DeepCompression::step_ << endl;
                    cout << " score: ";   for (int j = 0; j < num_col; ++j) { cout << col_score[j].first  << " "; }
                    cout << "\ncolumn: "; for (int j = 0; j < num_col; ++j) { cout << col_score[j].second << " "; }
                    cout << "\n  prob: "; for (int j = 0; j < num_col; ++j) { cout << DeepCompression::history_prob[this->layer_index][col_score[j].second] << " "; }
                    cout << "\n";                    
                    
                    for (int j = num_col_to_prune_ - DeepCompression::num_pruned_column[this->layer_index] - 1; j < num_col - DeepCompression::num_pruned_column[this->layer_index]; ++j) {
                        const int col_of_rank_j = col_score[j].second;
                        DeepCompression::history_prob[this->layer_index][col_of_rank_j] = 1;
                    }
                }

                // Update history_prob, according to some probabilities
                Dtype p_prune;
                caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_prune);
                // if ((DeepCompression::step_ - 1) % DeepCompression::prune_interval[this->layer_index] == 0) {  // when to update probs
                if (pow(70 + 0.0008 * DeepCompression::step_, -1.2) > p_prune) { // 230
                // if (std::min(Dtype(DeepCompression::learning_speed), (Dtype)0.004) * 4 > p_prune) {  
                    typedef std::pair<Dtype, int> mypair;
                    vector<mypair> col_score(num_col);
                    for (int j = 0; j < num_col; ++j) {
                        col_score[j].second = j; // index
                        col_score[j].first = 0;  // score
                        for (int i = 0; i < num_row; ++i) {
                            col_score[j].first += fabs(muweight[i * num_col +j]);
                        }
                        if (this->is_pruned[j]) { col_score[j].first = INT_MAX; } // make the pruned columns "float" up
                    }
                    sort(col_score.begin(), col_score.end());
                
                    // Print and check
                    cout << "update prob: " << layer_name << " step: " << DeepCompression::step_ << endl;
                    cout << " score: ";   for (int j = 0; j < num_col; ++j) { cout << col_score[j].first  << " "; }
                    cout << "\ncolumn: "; for (int j = 0; j < num_col; ++j) { cout << col_score[j].second << " "; }
                    cout << "\n  prob: "; for (int j = 0; j < num_col; ++j) { cout << DeepCompression::history_prob[this->layer_index][col_score[j].second] << " "; }
                    cout << "\n";
                
                    // Calculate functioning probability of each weight
                    const Dtype AA = 0.05; //0.05;
                    const Dtype aa = 0.0041; //0.0041;
                    const Dtype alpha = -log(aa/AA) / (num_col_to_prune - DeepCompression::num_pruned_column[this->layer_index] - 1);  // adjust alpha according to the remainder of cloumns
                    for (int j = 0; j < num_col_to_prune - DeepCompression::num_pruned_column[this->layer_index]; ++j) {               // note the range of j: only undermine those not-good-enough columns
                        const int col_of_rank_j = col_score[j].second;
                        DeepCompression::history_prob[this->layer_index][col_of_rank_j] = std::max(DeepCompression::history_prob[this->layer_index][col_of_rank_j] - AA * exp(-j * alpha), (Dtype)0);
                        if (DeepCompression::history_prob[this->layer_index][col_of_rank_j] == 0) {
                            ++ DeepCompression::num_pruned_column[this->layer_index]; 
                            this->is_pruned[col_of_rank_j] = true; 
                            for (int i = 0; i < num_row; ++i) { muweight[i * num_col + col_of_rank_j] = 0; } // if pruned, zero out weights
                        }
                    }
                } // update probability
              
              
                // Once the pruning ratio reached, set all the masks of non-zero prob to 1 and adjust their weights.
                // Get into here ONLY ONCE.
                if (DeepCompression::num_pruned_column[this->layer_index] >= num_col_to_prune) {
                    // Print and check
                    typedef std::pair<Dtype, int> mypair;
                    vector<mypair> col_score(num_col);
                    for (int j = 0; j < num_col; ++j) {
                        col_score[j].second = j; // index
                        col_score[j].first = 0;  // score
                        for (int i = 0; i < num_row; ++i) {
                            col_score[j].first += fabs(muweight[i * num_col +j]);
                        }
                    }
                    sort(col_score.begin(), col_score.end());
                    cout << "last time score: ";         for (int j = 0; j < num_col; ++j) { cout << col_score[j].first << " "; }
                    cout << "\nlast time column: ";      for (int j = 0; j < num_col; ++j) { cout << col_score[j].second << " "; }
                    cout << "\nlast time probability: "; for (int j = 0; j < num_col; ++j) { cout << DeepCompression::history_prob[this->layer_index][col_score[j].second] << " "; }
                    cout << "\n";
                    
                    for (int i = 0; i < count; ++i) {
                        if (DeepCompression::history_prob[this->layer_index][i % num_col] > 0) {
                            muweight[i] *= DeepCompression::history_prob[this->layer_index][i % num_col];
                            DeepCompression::history_prob[this->layer_index][i % num_col] = 1;
                        }
                    }
                }
                
                // With probability updated, generate masks
                Dtype rands[num_col];
                caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
                for (int i = 0; i < count; ++i) {
                    this->masks_[i] = rands[i % num_col] < DeepCompression::history_prob[this->layer_index][i % num_col] ? 1 : 0; // generate masks
                }              
                for (int i = 0; i < count; ++i) { weight_backup[i] = muweight[i]; }
                IF_RESTORE = true;
                
                // Print and check (before pruning)
                if (layer_name == "conv2" && DeepCompression::step_ % SHOW_INTERVAL == 0) {
                    // cout.setf(std::ios::left);
                    cout.width(5);  cout << "Index" << "   ";
                    cout.width(18); cout << "WeightBeforeMasked" << "   ";
                    cout.width(4);  cout << "Mask" << "   ";
                    cout.width(4);  cout << "Prob" << endl;
                    for (int i = 0; i < 20; ++i) {
                        cout.width(3);  cout << "#";
                        cout.width(2);  cout << i+1 << "   ";
                        cout.width(18); cout << muweight[i] << "   ";
                        cout.width(4);  cout << this->masks_[i] << "   ";
                        cout.width(4);  cout << DeepCompression::history_prob[this->layer_index][i] << endl;
                    }
                }
                
                for (int i = 0; i < count; ++i) { muweight[i] *= this->masks_[i]; } // do pruning
            } // use what criteria
        } // if in mask time && not reach target pruning ratio      
    } else {
        if (DeepCompression::prune_method == "SPP") {
            Dtype rands[num_col];
            caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
            for (int i = 0; i < count; ++i) {
                this->masks_[i] = rands[i % num_col] < DeepCompression::history_prob[this->layer_index][i % num_col] ? 1 : 0; // geerate masks
            }              
            for (int i = 0; i < count; ++i) { weight_backup[i] = muweight[i]; } // backup weights
            IF_RESTORE = true;
            for (int i = 0; i < count; ++i) { muweight[i] *= this->masks_[i]; } // do pruning
        }
    }
  // ------------------------------------------------------
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
    
    // Print feature map to check --------
    // If row 3 and 8 are pruned in previous layer, then channel 3 and 8 will be only biases in this layer's feature map.
    /*
    if (!DeepCompression::IN_TEST && this->layer_index == 0) {
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
    
    // -----------------------------------
    
    
    
    // Restore weights ----------------
    if (IF_RESTORE) {
        // cout << layer_name << ": restore weights! " << endl;
        this->blobs_[0]->mutable_cpu_data();
        // this->blobs_[0]->gpu_data(); 
        // Interesting! If the above line is added, something like "control" seems to transfer from cpu to gpu. 
        // Then modifying cpu weights won't affect their gpu counterparts.
        for (int i = 0; i < count; ++i) {
            muweight[i] = weight_backup[i];
        }
        
        /*
        // ========================
        // Chech restore
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
        // ========================
        */
    }
    // --------------------------------
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
  
  // ADDED BY WANGHUAN ------------------------------------------
  Dtype* muweight_diff = this->blobs_[0]->mutable_cpu_diff();      
  const int count = this->blobs_[0]->count();
  
  // UpdateDiffs(); // update second diff and so on
  
  // Print and check
  const string layer_name = this->layer_param_.name();
  if (layer_name == "conv2" && DeepCompression::step_ % SHOW_INTERVAL == 0) {
      cout.width(5);  cout << "Index" << "   ";
      cout.width(16); cout << "DiffBeforeMasked" << "   ";
      cout.width(4);  cout << "Mask" << "   ";
      cout.width(4);  cout << "Prob" << endl;
      for (int i = 0; i < 20; ++i) {
          cout.width(3);  cout << "#";
          cout.width(2);  cout << i+1 << "   ";
          cout.width(16); cout << muweight_diff[i] << "   ";
          cout.width(4);  cout << this->masks_[i] << "   ";
          cout.width(4);  cout << DeepCompression::history_prob[this->layer_index][i] << endl;
      }
  }
  
  // Apply masks to diff
  const bool IF_mask = DeepCompression::IN_RETRAIN || (DeepCompression::step_ - 1) >= DeepCompression::prune_begin_iter;
  if (IF_mask) {
      if (DeepCompression::prune_method == "ProgressivePruning" && DeepCompression::criteria == "diff") { }//UpdateMasks(); }
      for (int j = 0; j < count; ++j) { muweight_diff[j] *= this->masks_[j]; }
  }
  // ------------------------------------------------------------- 
  
  
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
