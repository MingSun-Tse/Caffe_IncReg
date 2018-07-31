#include <boost/thread.hpp>
#include <boost/thread.hpp>
#include "caffe/layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"

namespace caffe {
template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

template<typename Dtype>
void Layer<Dtype>::IF_prune_finished() {
    const string layer_name = this->layer_param_.name();
    if (APP<Dtype>::layer_index.count(layer_name) != 0) {
        const int L = APP<Dtype>::layer_index[layer_name];
        if (APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
            Dtype pruned_ratio = 0;
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
                std::cout << layer_name << " prune finished!" 
                     << "  step: " << APP<Dtype>::step_
                     << "  net speedup: " << APP<Dtype>::speedup
                     << "  net compRatio: " << APP<Dtype>::compRatio
                     << "  pruned_ratio: " << rlayer
                     << "  pruned_ratio_row: " << rrow
                     << "  pruned_ratio_col: " << rcol 
                     << "  prune_ratio: " << APP<Dtype>::prune_ratio[L] << std::endl;
                
                // IF_all_layer_prune_finished
                APP<Dtype>::IF_alpf = true;
                for (int i = 0; i < APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt; ++i) {
                    if (APP<Dtype>::iter_prune_finished[i] == INT_MAX) {
                        APP<Dtype>::IF_alpf = false;
                        break;
                    }
                }
            }
        }
    }
}

template <typename Dtype>
bool Layer<Dtype>::IF_hppf() {
    // IF_higher_priority_prune_finished 
    bool IF_hppf = true;
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    for (int i = 0; i < APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt; ++i) {
        if (APP<Dtype>::priority[i] < APP<Dtype>::priority[L] && APP<Dtype>::iter_prune_finished[i] == INT_MAX) {
            IF_hppf = false;
            break;
        }
    }
    return IF_hppf;
}

template <typename Dtype> 
void Layer<Dtype>::UpdateNumPrunedRow() {
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    const int num_col = this->blobs_[0]->count(1);
    
    cout << "        " << this->layer_param_.name() << " in UpdateNumPrunedRow" << endl;
    vector<int>::iterator it;
    for (it = APP<Dtype>::rows_to_prune[L].begin(); it != APP<Dtype>::rows_to_prune[L].end(); ++it) {
      caffe_gpu_set(num_col, (Dtype)0, this->blobs_[0]->mutable_gpu_data() + *it * num_col);
      caffe_gpu_set(num_col, (Dtype)0, this->masks_[0]->mutable_gpu_data() + *it * num_col);
      APP<Dtype>::IF_row_pruned[L][*it] = true;
      cout << " " << this->layer_param_.name() << " prune a row successfully: " << (*it) << endl;
    }
    APP<Dtype>::num_pruned_row[L] += APP<Dtype>::rows_to_prune[L].size();
    APP<Dtype>::rows_to_prune[L].clear();
}

template <typename Dtype> 
void Layer<Dtype>::UpdateNumPrunedCol() {
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int num_chl = this->blobs_[0]->shape()[1];
    const int num_row_per_g = num_row / APP<Dtype>::group[L];
    const int filter_spatial_size = this->blobs_[0]->count(2);
    
    cout << "        " << this->layer_param_.name() << " in UpdateNumPrunedCol" << endl;
    vector<int>::iterator it;
    for (it = APP<Dtype>::pruned_rows[L-1].begin(); it != APP<Dtype>::pruned_rows[L-1].end(); ++it) {
        const int chl = *it % num_chl;
        const int g   = *it / num_chl;
        for (int i = g * num_row_per_g; i < (g + 1) * num_row_per_g; ++i) {
            for (int j = chl * filter_spatial_size; j < (chl + 1) * filter_spatial_size; ++j) {
                this->masks_[0]->mutable_cpu_data()[i * num_col + j] = 0;
                APP<Dtype>::IF_col_pruned[L][j][g] = true;
            }
        }
        APP<Dtype>::num_pruned_col[L] += filter_spatial_size * 1.0 / APP<Dtype>::group[L];
        cout << "  " << this->layer_param_.name() << " prune a channel successfully: " << chl << endl;
    }
    APP<Dtype>::pruned_rows[L-1].clear();
}


template <typename Dtype> 
void Layer<Dtype>::UpdatePrunedRatio() {
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int group = APP<Dtype>::group[L];
    
    if (APP<Dtype>::prune_unit == "Weight") {
        // Row
        for (int i = 0; i < num_row; ++i) {
            if (APP<Dtype>::IF_row_pruned[L][i]) { continue; }
            bool IF_whole_row_pruned = true;
            for (int j = 0; j < num_col; ++j) {
                if (!APP<Dtype>::IF_weight_pruned[L][i * num_col + j]) {
                    IF_whole_row_pruned = false;
                    break;
                }
            }
            if (IF_whole_row_pruned) {
                APP<Dtype>::IF_row_pruned[L][i] = true;
                APP<Dtype>::num_pruned_row[L] += 1;
            }
        }
        // Column
        for (int j = 0; j < num_col; ++j) {
            if (APP<Dtype>::IF_col_pruned[L][j][0]) { continue; }
            bool IF_whole_col_pruned = true;
            for (int i = 0; i < num_row; ++i) {
                if (!APP<Dtype>::IF_weight_pruned[L][i * num_col + j]) {
                    IF_whole_col_pruned = false;
                    break;
                }
            }
            if (IF_whole_col_pruned) {
                for (int g = 0; g < group; ++g) {
                    APP<Dtype>::IF_col_pruned[L][j][g] = true;
                }
                APP<Dtype>::num_pruned_col[L] += 1;
            }
        }
    }
    APP<Dtype>::pruned_ratio_col[L] = APP<Dtype>::num_pruned_col[L] / num_col;
    APP<Dtype>::pruned_ratio_row[L] = APP<Dtype>::num_pruned_row[L] * 1.0 / num_row;
    
    if (APP<Dtype>::prune_unit == "Weight") {
        const Dtype new_pruned_ratio = APP<Dtype>::num_pruned_weight[L] * 1.0 / count;
        if (new_pruned_ratio > APP<Dtype>::pruned_ratio[L]) {
            this->IF_masks_updated = true;
            APP<Dtype>::pruned_ratio[L] = new_pruned_ratio;
        }
    } else {
        const Dtype new_pruned_ratio = (APP<Dtype>::pruned_ratio_col[L] + APP<Dtype>::pruned_ratio_row[L]) 
                                      - APP<Dtype>::pruned_ratio_col[L] * APP<Dtype>::pruned_ratio_row[L];
        if (new_pruned_ratio > APP<Dtype>::pruned_ratio[L]) {
            this->IF_masks_updated = true;
            APP<Dtype>::pruned_ratio[L] = new_pruned_ratio;
        }
    }
}

template <typename Dtype> 
void Layer<Dtype>::Print(char mode) {
    assert(mode == 'f' || mode = 'b'); /// forward, backward
    const string layer_name = this->layer_param_.name();
    const int num_col = this->blobs_[0]->count() / this->blobs_[0]->shape()[0];
    const int num_row = this->blobs_[0]->shape()[0];
    const Dtype* w = this->blobs_[0]->cpu_data();
    const Dtype* d = this->blobs_[0]->cpu_diff();
    const Dtype* m = this->masks_[0]->cpu_data();
    // print Index, blob, Mask
    cout.width(5);  cout << "Index" << "   ";
    const string blob = (mode == 'f') ? "WeightBeforeMasked" : "DiffBeforeMasked";
    cout.width(blob.size()); cout << blob << "   ";
    cout.width(4);  cout << "Mask" << "   ";
    // print additional info
    string info = "";
    if (APP<Dtype>::prune_coremthd.substr(0, 2) == "PP") {
      info = "HistoryProb";
    } else if (APP<Dtype>::prune_coremthd.substr(0, 2) == "Reg") {
      info = "HistoryReg";
    } else {
      info = "WeightBeforeMasked";
    }
    Dtype* info_data = NULL;
    if (APP<Dtype>::prune_method.substr(0, 2) == "PP" || APP<Dtype>::prune_method.substr(0, 3) == "Reg") {
      info_data = this->history_punish_[0]->mutable_cpu_data();
    } else {
      info_data = this->blobs_[0]->mutable_cpu_data();
    }
    cout.width(info.size()); cout << info << " - " << this->layer_param_.name() << endl;

    if (APP<Dtype>::prune_unit == "Row") {
        const int show_num = APP<Dtype>::show_num_weight > num_row ? num_row : APP<Dtype>::show_num_weight;
        for (int i = 0; i < show_num; ++i) {
            // print Index
            cout.width(3); cout << "r"; 
            cout.width(2); cout << i+1 << "   ";
            // print blob
            Dtype sum_w = 0, sum_d = 0;
            for (int j = 0; j < num_col; ++j) {
                sum_w += fabs(w[i * num_col + j]);
                sum_d += fabs(d[i * num_col + j]);
            }
            sum_w /= num_col; /// average abs weight
            sum_d /= num_col; /// average abs diff
            const Dtype s = mode == 'f' ? sum_w : sum_d;
            cout.width(blob.size()); cout << s << "   ";
            // print Mask
            cout.width(4);  cout << m[i * num_col] << "   ";
            // print info
            cout.width(info.size());  cout << info_data[i * num_col] << endl;
        }
        
    } else if (APP<Dtype>::prune_unit == "Col") {
        const int show_num = APP<Dtype>::show_num_weight > num_col ? num_col : APP<Dtype>::show_num_weight;
        for (int j = 0; j < show_num; ++j) {
            // print Index
            cout.width(3); cout << "c"; 
            cout.width(2); cout << j+1 << "   ";
            // print blob
            Dtype sum_w = 0, sum_d = 0;
            for (int i = 0; i < num_row; ++i) {
                sum_w += fabs(w[i * num_col + j]);
                sum_d += fabs(d[i * num_col + j]);
            }
            sum_w /= num_row; /// average abs weight
            sum_d /= num_row; /// average abs diff
            const Dtype s = mode == 'f' ? sum_w : sum_d;
            cout.width(blob.size()); cout << s << "   ";
            // print Mask
            cout.width(4);  cout << m[j] << "   ";
            // print info
            cout.width(info.size());  cout << info_data[j] << endl;
        }
    } else if (APP<Dtype>::prune_unit == "Weight") {
        for (int i = 0; i < APP<Dtype>::show_num_weight; ++i) {
            // print Index
            cout.width(3); cout << "w";
            cout.width(2); cout << i+1 << "   ";
            const Dtype s = mode == 'f' ? fabs(w[i]) : fabs(d[i]);
            cout.width(blob.size()); cout << s << "   ";
            // print Mask
            cout.width(4);  cout << m[i] << "   ";
            // print info
            cout.width(info.size());  cout << info_data[i] << endl;
        }
    }
}

template <typename Dtype>
void Layer<Dtype>::RestoreMasks() {
    /* Restore pruning state when retrain */
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const Dtype *weight = this->blobs_[0]->cpu_data();
    const string layer_name = this->layer_param_.name();
    const int L = APP<Dtype>::layer_index[layer_name];
    const int group = APP<Dtype>::group[L];
    const int num_row_per_g = num_row / group;
    const string mthd = APP<Dtype>::prune_method;
    Dtype num_pruned_col = 0;
    int num_pruned_row = 0;
    if (APP<Dtype>::prune_unit == "Weight") {
        for (int i = 0; i < count; ++i) {
            if (!weight[i]) {
                this->masks_[0]->mutable_cpu_data()[i] = 0;
                ++ APP<Dtype>::num_pruned_weight[L];
            }
        }
    } else {
        // Column
        for (int j = 0; j < num_col; ++j) {
            for (int g = 0; g < group; ++g) {
                Dtype sum = 0;
                for (int i = g * num_row_per_g; i < (g+1) * num_row_per_g; ++i) { 
                    sum += fabs(weight[i * num_col + j]);
                }
                if (sum == 0) { 
                    num_pruned_col += 1.0 / group; /// note that num_pruned_row is always integer while num_pruned_col can be non-integer.
                    APP<Dtype>::IF_col_pruned[L][j][g] = true;
                    for (int i = g * num_row_per_g; i < (g+1) * num_row_per_g; ++i) { 
                        this->masks_[0]->mutable_cpu_data()[i * num_col + j] = 0;
                    }
                }
            }
        }
        // Row
        for (int i = 0; i < num_row; ++i) { 
            Dtype sum = 0;
            for (int j = 0; j < num_col; ++j) { 
                sum += fabs(weight[i * num_col + j]); 
            }
            if (sum == 0) {
                ++ num_pruned_row;
                APP<Dtype>::IF_row_pruned[L][i] = true;
                for (int j = 0; j < num_col; ++j) { 
                    this->masks_[0]->mutable_cpu_data()[i * num_col + j] = 0;
                }
            }
        }
        APP<Dtype>::num_pruned_col[L] = num_pruned_col;
        APP<Dtype>::num_pruned_row[L] = num_pruned_row;
    }
    this->UpdatePrunedRatio();
    Dtype pruned_ratio = 0;
    if      (APP<Dtype>::prune_unit == "Weight") { pruned_ratio = APP<Dtype>::pruned_ratio[L];     }
    else if (APP<Dtype>::prune_unit == "Row"   ) { pruned_ratio = APP<Dtype>::pruned_ratio_row[L]; }
    else if (APP<Dtype>::prune_unit == "Col"   ) { pruned_ratio = APP<Dtype>::pruned_ratio_col[L]; }
    if (pruned_ratio >= APP<Dtype>::prune_ratio[L]) {
        APP<Dtype>::iter_prune_finished[L] = -1; /// To check multi-GPU
        cout << L << ": " << layer_name << " prune finished." << endl;
    }
    LOG(INFO) << "  Masks restored,"
              << "  num_pruned_col=" << APP<Dtype>::num_pruned_col[L] << "(" << APP<Dtype>::num_pruned_col[L] * 1.0 / num_col << ")"
              << "  num_pruned_row=" << APP<Dtype>::num_pruned_row[L] << "(" << APP<Dtype>::num_pruned_row[L] * 1.0 / num_row << ")"
              << "  pruned_ratio="   << APP<Dtype>::pruned_ratio[L]
              << "  prune_ratio="    << APP<Dtype>::prune_ratio[L];
}

template <typename Dtype>
void Layer<Dtype>::PruneSetUp(const PruneParameter& prune_param) {
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    APP<Dtype>::prune_ratio.push_back(prune_param.prune_ratio());
    APP<Dtype>::pruned_ratio.push_back(0); // used in TEST
    this->IF_masks_updated = true;
    if (this->phase_ == TEST) { return; }
    
    // Get layer_index
    const string layer_name = this->layer_param_.name();
    if (APP<Dtype>::layer_index.count(layer_name) == 0) {
        APP<Dtype>::layer_index[layer_name] = APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt;
        if (!strcmp(this->type(), "Convolution") or !strcmp(this->type(), "NdConvolution")) {
            ++ APP<Dtype>::conv_layer_cnt;
        } else if (!strcmp(this->type(), "InnerProduct")) {
            ++ APP<Dtype>::fc_layer_cnt;
        } else {
            LOG(FATAL) << "Seems wrong, pruning setup can ONLY be put in the layers with learnable parameters (Conv and FC), please check.";
        }
        LOG(INFO) << "New learnable layer registered: " << layer_name
            << ". Its layer index: " << APP<Dtype>::layer_index[layer_name] << endl;
    }
    const int L = APP<Dtype>::layer_index[layer_name];
    
    // Note: the varibales below can ONLY be used in training.
    // Note: These varibales will be called for every GPU, whereas since we use `layer_index` to index, so it doesn't matter.
    // Set up prune parameters of layer
    APP<Dtype>::IF_update_row_col_layer.push_back(prune_param.if_update_row_col());
    APP<Dtype>::rows_to_prune.push_back(vector<int>());
    APP<Dtype>::pruned_rows.push_back(vector<int>());
    APP<Dtype>::pruned_ratio_col.push_back(0);
    APP<Dtype>::pruned_ratio_row.push_back(0);
    APP<Dtype>::GFLOPs.push_back(this->blobs_[0]->count()); // further calculated in `net.cpp`, after layer SetUp
    APP<Dtype>::num_param.push_back(count);
    // Pruning state
    APP<Dtype>::num_pruned_col.push_back(0);
    APP<Dtype>::num_pruned_row.push_back(0);
    APP<Dtype>::num_pruned_weight.push_back(0);
    APP<Dtype>::IF_row_pruned.push_back(vector<bool>(num_row, false));
    vector<bool> vec_tmp(APP<Dtype>::group[L], false); // initialization
    APP<Dtype>::IF_col_pruned.push_back(vector<vector<bool> >(num_col, vec_tmp));
    APP<Dtype>::IF_weight_pruned.push_back(vector<bool>(count, false));
    // Info shared among layers
    APP<Dtype>::filter_spatial_size.push_back(this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3]); // TODO(mingsuntse): check 3D CNN, this is okay?
    APP<Dtype>::priority.push_back(prune_param.priority());
    APP<Dtype>::iter_prune_finished.push_back(INT_MAX);
    LOG(INFO) << "Pruning setup done: " << layer_name;
}

template <typename Dtype> 
void Layer<Dtype>::PruneForward() {
    #ifdef ShowTimingLog
    clock_t t1 = clock();
    cout << this->layer_param_.name() << ": forward GPU begins timing" << endl;
    #endif
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
                if (APP<Dtype>::prune_unit == "Col" 
                      && L != APP<Dtype>::conv_layer_cnt - 1 
                      && L != APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt - 1 // The last conv layer and last fc layer need not update row.
                      && APP<Dtype>::step_-1 - APP<Dtype>::iter_prune_finished[L+1] <= 1) {
                  this->UpdateNumPrunedRow();
                } else if (APP<Dtype>::prune_unit == "Row"
                      && L != 0
                      && L != APP<Dtype>::conv_layer_cnt // The first conv layer and first fc layer need not update column.
                      && APP<Dtype>::pruned_rows[L-1].size()) {
                    this->UpdateNumPrunedCol();
                }
                this->UpdatePrunedRatio();
                this->IF_prune_finished();
            }
        }
        
        // Print and check
        // put this outside, to print even when we do not prune
        if (APP<Dtype>::show_layer.size() >= L+1 && APP<Dtype>::show_layer[L] == '1'
                    && APP<Dtype>::step_ % APP<Dtype>::show_interval == 0) {
            this->Print('f');
        }
        
        #ifdef ShowTimingLog
        cout << "  after updating masks: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        #endif
        

        // Summary print 
        if (mthd != "None" && L < APP<Dtype>::show_num_layer) {
            cout << layer_name << "  IF_prune: " << IF_prune 
                 << "  pruned_ratio: " << APP<Dtype>::pruned_ratio[L];
            cout << "  pruned_ratio_row: " << APP<Dtype>::num_pruned_row[L] * 1.0 / num_row << "(" << APP<Dtype>::num_pruned_row[L] << ")"
                 << "  pruned_ratio_col: " << APP<Dtype>::num_pruned_col[L] * 1.0 / num_col << "(" << APP<Dtype>::num_pruned_col[L] << ")";
            cout << "  prune_ratio: "  << APP<Dtype>::prune_ratio[L] << endl;
        }
        
        // Apply masks
        if (this->IF_masks_updated) {
            caffe_gpu_mul(this->blobs_[0]->count(), 
                          this->blobs_[0]->gpu_data(),
                          this->masks_[0]->gpu_data(),
                          this->blobs_[0]->mutable_gpu_data());
            this->IF_masks_updated = false;
        }
        #ifdef ShowTimingLog
        cout << "  after updating masks: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
        #endif
        
    }
}

template <typename Dtype> 
void Layer<Dtype>::PruneBackward(const vector<Blob<Dtype>*>& top) {
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    
    // Print and check
    if (APP<Dtype>::show_layer.size() >= L+1 && APP<Dtype>::show_layer[L] == '1' 
                && APP<Dtype>::step_ % APP<Dtype>::show_interval == 0 && APP<Dtype>::inner_iter == 0) { 
       this->Print('b');
    }
    
    if (APP<Dtype>::pruned_ratio[L] > 0) { 
        caffe_gpu_mul(this->blobs_[0]->count(), 
                      this->blobs_[0]->gpu_diff(), 
                      this->masks_[0]->gpu_data(), 
                      this->blobs_[0]->mutable_gpu_diff());
    }
}

INSTANTIATE_CLASS(Layer);

}  // namespace caffe
