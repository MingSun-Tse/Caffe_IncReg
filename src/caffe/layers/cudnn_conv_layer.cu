#ifdef USE_CUDNN
#include <vector>
#include "caffe/layers/cudnn_conv_layer.hpp"
#include <climits>

using namespace std;
#define SHOW_INTERVAL 20

namespace caffe {

__global__ void sync_conv_groups() { }

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    // ADDED BY WANGHUAN -----------------------------------
    /* DEPRECATED!!
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_filter = this->blobs_[0]->shape()[0];
    const int num_col = count / num_filter;
    const string layer_name = this->layer_param_.name();
    const int conv_index = layer_name[4] - '1';
    const int num_col_to_prune = (int) (DeepCompression::PruneRate[conv_index] * num_col);
    const bool if_apply_masks =  DeepCompression::IN_RETRAIN 
                                 || (DeepCompression::step_ - 1) % DeepCompression::period >= DeepCompression::when_to_apply_masks;
    const bool if_reboot = false; // not used now
    vector<Dtype> weight_backup(count, 0);
    bool IF_RESTORE = false;
  
    cout << " !!! This is in cudnn_conv_layer.cu !!!" << endl;
    if (1) {
        // Print and check
        cout << layer_name 
             << "  if_apply_masks: " << if_apply_masks 
             << "  num_pruned_column: ";
        cout.width(3); cout << this->num_pruned_column 
                            << "  num_col_to_prune: " << num_col_to_prune
                            << endl;
              
        // Update masks and apply masks
        if (if_apply_masks) {
            // Used in IHT, not well experimented yet.
            if (if_reboot) {
                for (int i = 0; i < count; ++i) { this->masks_[i] = 1; }
                for (int j = 0; j < num_col; ++j) { this->is_pruned[j] = 0; this->history_score[j] = 0; }
                this->num_pruned_column = 0;
            }
          
            // Step1 - Update masks
            if (this->num_pruned_column < num_col_to_prune) { // if pruning ratio reached, no need to update masks
                if (DeepCompression::Method == "ProgressivePruning" && DeepCompression::criteria == "energy") { 
                    UpdateMasks(); 
                } else if (DeepCompression::Method == "SPP" && !DeepCompression::IN_TEST) {
                    // Recover the best columns, according to some probability.
                    Dtype p_recover;
                    caffe_rng_uniform(1, Dtype(0), Dtype(1), &p_recover);
                    if (pow(40 + 0.00027 * DeepCompression::step_, -1.3) > p_recover) {
                        // Sort
                        typedef std::pair<Dtype, int> mypair;
                        vector<mypair> col_score(num_col);
                        for (int j = 0; j < num_col; ++j) {
                            col_score[j].second = j; // index
                            col_score[j].first = 0;  // score
                            for (int i = 0; i < num_filter; ++i) {
                                col_score[j].first += fabs(muweight[i * num_col +j]);
                            }
                            if (this->is_pruned[j]) { col_score[j].first = INT_MAX; } // make the pruned columns "float" up
                        } 
                        sort(col_score.begin(), col_score.end());
                        for (int j = 0; j < num_col - this->num_pruned_column; ++j) {
                            if (j >= (num_col_to_prune - this->num_pruned_column)) {
                                const int col_of_rank_j = col_score[j].second;
                                this->history_prob[col_of_rank_j] = 1;
                            }
                        }
                    }
                  

                    // Update history_prob, according to some probability.
                    Dtype p_prune;
                    caffe_rng_uniform(1, Dtype(0), Dtype(1), &p_prune);
                    // if ((DeepCompression::step_ - 1) % DeepCompression::prune_interval[conv_index] == 0) {  // when to update probs
                    if (pow(230 + 0.0008 * DeepCompression::step_, -1.2) * 5 > p_prune) {
                    // if (std::min(Dtype(DeepCompression::learning_speed) * 2, (Dtype)0.004)  > p_prune) {
                        cout << layer_name << " update drop probability!" << endl;
                        // Sort
                        typedef std::pair<Dtype, int> mypair;
                        vector<mypair> col_score(num_col);
                        for (int j = 0; j < num_col; ++j) {
                            col_score[j].second = j; // index
                            col_score[j].first = 0;  // score
                            for (int i = 0; i < num_filter; ++i) {
                                col_score[j].first += fabs(muweight[i * num_col +j]);
                            }
                            if (this->is_pruned[j]) { col_score[j].first = INT_MAX; } // make the pruned columns "float" up
                        }
                        sort(col_score.begin(), col_score.end());
                    
                        // Check order
                        cout << "score: ";
                        for (int j = 0; j < num_col; ++j) {
                            cout << col_score[j].first << " ";
                        }
                        cout << endl; 
                        cout << "column: ";
                        for (int j = 0; j < num_col; ++j) {
                            cout << col_score[j].second << " ";
                        }
                        cout << endl; 
                        cout << "probability: ";
                        for (int j = 0; j < num_col; ++j) {
                            cout << this->history_prob[col_score[j].second] << " ";
                        }
                        cout << endl;
                    
                        // Calculate functioning probability of each weight
                        const Dtype AA = 0.05;
                        const Dtype aa = 0.0041;
                        const Dtype alpha = -log(aa/AA) / (num_col_to_prune - this->num_pruned_column - 1);  // adjust alpha according to the remainder of cloumns
                        for (int j = 0; j < num_col_to_prune - this->num_pruned_column; ++j) {               // Note the range of j: only undermine those not-good-enough columns
                            const int col_of_rank_j = col_score[j].second;
                            this->history_prob[col_of_rank_j] = std::max(this->history_prob[col_of_rank_j] - AA * exp(-j * alpha), (Dtype)0);
                            if (this->history_prob[col_of_rank_j] == 0) {
                                ++ this->num_pruned_column;
                                ++ DeepCompression::num_pruned_column[conv_index];
                                this->is_pruned[col_of_rank_j] = true; // TODO repace this
                            }
                        }
                    }
                  
                    // Apply SPP, generate masks
                    Dtype rands[num_col];
                    caffe_rng_uniform(num_col, Dtype(0), Dtype(1), rands);
                    for (int i = 0; i < count; ++i) {
                        this->masks_[i] = rands[i % num_col] < this->history_prob[i % num_col] ? 1 : 0;
                    }
                  
                    // Backup weights to restore later
                    for (int i = 0; i < count; ++i) {
                        weight_backup[i] = muweight[i]; 
                    }
                    IF_RESTORE = true;
                  
                    // Once the pruning ratio reached, set all the masks of non-zero prob to 1 and adjust their weights.
                    // Get into here ONLY ONCE.
                    if (this->num_pruned_column >= num_col_to_prune) {
                        for (int i = 0; i < count; ++i) {
                            if (this->history_prob[i % num_col] > 0) {
                                this->masks_[i] = 1;
                                muweight[i] *= this->history_prob[i % num_col];
                            }
                        }
                        
                        // Sort
                        typedef std::pair<Dtype, int> mypair;
                        vector<mypair> col_score(num_col);
                        for (int j = 0; j < num_col; ++j) {
                            col_score[j].second = j; // index
                            col_score[j].first = 0;  // score
                            for (int i = 0; i < num_filter; ++i) {
                                col_score[j].first += fabs(muweight[i * num_col +j]);
                            }
                        }
                        sort(col_score.begin(), col_score.end());

                        // Check order
                        cout << "last time score: ";
                        for (int j = 0; j < num_col; ++j) {
                            cout << col_score[j].first << " ";
                        }
                        cout << endl;                        
                        cout << "last time column: ";
                        for (int j = 0; j < num_col; ++j) {
                            cout << col_score[j].second << " ";
                        }
                        cout << endl;                                               
                        cout << "last time probability: ";
                        for (int j = 0; j < num_col; ++j) {
                            cout << this->history_prob[col_score[j].second] << " ";
                        }
                        cout << endl;
                    }
                } // USE WHAT CRITERIA
                
                // Print and check
                if (layer_name == "conv2" && DeepCompression::step_ % SHOW_INTERVAL == 0) {
                    // cout.setf(std::ios::left);
                    cout.width(5); cout << "Index" << "   ";
                    cout.width(18); cout << "WeightBeforeMasked" << "   ";
                    cout.width(4); cout << "Mask" << "   ";
                    cout.width(4); cout << "Prob" << endl;
                    for (int i = 0; i < 20; ++i) {
                        cout.width(3); cout << "#";
                        cout.width(2); cout << i+1 << "   ";
                        cout.width(18); cout << muweight[i] << "   ";
                        cout.width(4); cout << this->masks_[i] << "   ";
                        cout.width(4); cout << this->history_prob[i] << endl;
                    }
                }
            } // IF REACH PRUNINGRATIO
          
            // Step2 - Do pruning
            // No matter whether SSP used or hard pruning, they all need pruning, as long as if_apply_masks = 1.
            for (int i = 0; i < count; ++i) { 
                muweight[i] *= this->masks_[i];
            }        
        } // IF APPLY MASKS     
    } else {
        LOG(INFO) << "Local Test Forward GPU in Layer " << layer_name;
        // Generate masks. Since the weights function in model "probably" during training, they should also function the same way during testing.
        Dtype rands[num_col];
        caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
        for (int i = 0; i < count; ++i) {
            this->masks_[i] = rands[i % num_col] < this->history_prob[i % num_col] ? 1 : 0;
        }
        
        // Backup weights to restore later
        for (int i = 0; i < count; ++i) {
            weight_backup[i] = muweight[i]; 
        }
        IF_RESTORE = true;
        
        // Apply masks
        for (int i = 0; i < count; ++i) {
            muweight[i] *= this->masks_[i];
        }
        cout << "Local Test Forward GPU, masks and probs: " << layer_name << endl;
        // for (int i = 0; i < 40; ++i) {
            // cout << this->masks_[i] << " "; 
        // }
        // cout << endl;
        for (int i = 0; i < num_col; ++i) {
            cout << this->history_prob[i] << " "; 
        }
        cout << endl;
    }
  */
  // ------------------------------------------------------
  
  
  const Dtype* weight = this->blobs_[0]->gpu_data();
  // cout << "gpu weight " << weight[1] << endl; // why this is wrong? WANGHUAN
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // Forward through cuDNN in parallel over groups.
    for (int g = 0; g < this->group_; g++) {
      // Filters.
      CUDNN_CHECK(cudnnConvolutionForward(handle_[g],
            cudnn::dataType<Dtype>::one,
            bottom_descs_[i], bottom_data + bottom_offset_ * g,
            filter_desc_, weight + this->weight_offset_ * g,
            conv_descs_[i],
            fwd_algo_[i], workspace[g], workspace_fwd_sizes_[i],
            cudnn::dataType<Dtype>::zero,
            top_descs_[i], top_data + top_offset_ * g));
            
      // Bias.
      if (this->bias_term_) {
        const Dtype* bias_data = this->blobs_[1]->gpu_data();
        CUDNN_CHECK(cudnnAddTensor(handle_[g],
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_data + bias_offset_ * g,
              cudnn::dataType<Dtype>::one,
              top_descs_[i], top_data + top_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
    
    // Restore weights ----------------
    /*
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
    */
    // --------------------------------
   
  }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* weight = NULL;
  Dtype* weight_diff = NULL;
  if (this->param_propagate_down_[0]) {
    weight = this->blobs_[0]->gpu_data();
    weight_diff = this->blobs_[0]->mutable_gpu_diff();

  }

  Dtype* bias_diff = NULL;
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    bias_diff = this->blobs_[1]->mutable_gpu_diff();
  }

  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Backward through cuDNN in parallel over groups and gradients.
    for (int g = 0; g < this->group_; g++) {
      // Gradient w.r.t. bias.
      if (this->bias_term_ && this->param_propagate_down_[1]) {
        CUDNN_CHECK(cudnnConvolutionBackwardBias(handle_[0*this->group_ + g],   /*what is CUDNN_CHECK() ?? */
              cudnn::dataType<Dtype>::one,
              top_descs_[i],  top_diff + top_offset_ * g,
              cudnn::dataType<Dtype>::one,
              bias_desc_, bias_diff + bias_offset_ * g));
      }



      // Gradient w.r.t. weights.
      if (this->param_propagate_down_[0]) {
        const Dtype* bottom_data = bottom[i]->gpu_data();
        CUDNN_CHECK(cudnnConvolutionBackwardFilter(
              handle_[1*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              bottom_descs_[i], bottom_data + bottom_offset_ * g,
              top_descs_[i],    top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_filter_algo_[i], workspace[1*this->group_ + g],
              workspace_bwd_filter_sizes_[i],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight_diff + this->weight_offset_ * g));
      }


      // ADDED BY WANGHUAN ------------------------------------------
      /*
      Dtype* muweight_diff = this->blobs_[0]->mutable_cpu_diff();      
      const int count = this->blobs_[0]->count();
      
      // UpdateDiffs(); // update second diff and so on
      
      // Print and check
      const string layer_name = this->layer_param_.name();
      if (layer_name == "conv2" && DeepCompression::step_ % SHOW_INTERVAL == 0) {
          cout.width(5); cout << "Index" << "   ";
          cout.width(16); cout << "DiffBeforeMasked" << "   ";
          cout.width(4); cout << "Mask" << "   ";
          cout.width(4); cout << "Prob" << endl;
          for (int i = 0; i < 20; ++i) {
              cout.width(3); cout << "#";
              cout.width(2); cout << i+1 << "   ";
              cout.width(16); cout << muweight_diff[i] << "   ";
              cout.width(4); cout << this->masks_[i] << "   ";
              cout.width(4); cout << this->history_prob[i] << endl;
          }
      }
      
      // Apply masks to diff
      const bool if_apply_masks = DeepCompression::IN_RETRAIN || (DeepCompression::step_ - 1) % DeepCompression::period >= DeepCompression::when_to_apply_masks;
      if (if_apply_masks) {
          if (DeepCompression::Method == "ProgressivePruning" && DeepCompression::criteria == "diff") { UpdateMasks(); }
          for (int j = 0; j < count; ++j) {
              muweight_diff[j] *= this->masks_[j];
          }
      }
      */
      // -------------------------------------------------------------


      // Gradient w.r.t. bottom data.
      if (propagate_down[i]) {
        if (weight == NULL) {
          weight = this->blobs_[0]->gpu_data();
        }
        Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
        CUDNN_CHECK(cudnnConvolutionBackwardData(
              handle_[2*this->group_ + g],
              cudnn::dataType<Dtype>::one,
              filter_desc_, weight + this->weight_offset_ * g,
              top_descs_[i], top_diff + top_offset_ * g,
              conv_descs_[i],
              bwd_data_algo_[i], workspace[2*this->group_ + g],
              workspace_bwd_data_sizes_[i],
              cudnn::dataType<Dtype>::zero,
              bottom_descs_[i], bottom_diff + bottom_offset_ * g));
      }
    }

    // Synchronize the work across groups, each of which went into its own
    // stream, by launching an empty kernel into the default (null) stream.
    // NOLINT_NEXT_LINE(whitespace/operators)
    sync_conv_groups<<<1, 1>>>();
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(CuDNNConvolutionLayer);

}  // namespace caffe
#endif
