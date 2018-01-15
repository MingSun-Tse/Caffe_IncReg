#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#include <cstdlib>
#include <cmath>
#define NSUM 50
#define SHOW_INTERVAL 20
#define SHOW_NUM 20

namespace caffe {
using namespace std;

template <typename Dtype>
void ConvolutionLayer<Dtype>::PruneSetUp(const PruneParameter& prune_param) {
    // Basic setting
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    this->weight_backup.resize(count); /// still used in TEST phase when using PP
    
    // Get layer_index
    const string layer_name = this->layer_param_.name();
    if (this->phase_ == TRAIN) {
        if (APP<Dtype>::layer_index.count(layer_name) == 0) {
            APP<Dtype>::layer_index[layer_name] = APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt;
            ++ APP<Dtype>::conv_layer_cnt;
            cout << "a new layer registered: " << layer_name 
                 << "  total layers: " << APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt << endl;
        }
    } else { return; }
    const int L = APP<Dtype>::layer_index[layer_name];
    cout << "prune setup: " << layer_name  
         << "  its layer_index: " << L
         << "  total layers: " << APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt << endl;
    
    
    // Note: the varibales below can ONLY be used in training.
    // Note: These varibales will be called for every GPU, whereas since we use `layer_index` to index, so it doesn't matter.
    // Set up prune parameters of layer
    APP<Dtype>::prune_ratio.push_back(prune_param.prune_ratio());
    APP<Dtype>::delta.push_back(prune_param.delta()); // TODO: abolish delta
    APP<Dtype>::IF_update_row_col_layer.push_back(prune_param.if_update_row_col());
    APP<Dtype>::pruned_ratio.push_back(0);
    APP<Dtype>::pruned_ratio_col.push_back(0);
    APP<Dtype>::pruned_ratio_row.push_back(0);
    APP<Dtype>::GFLOPs.push_back(this->blobs_[0]->shape()[0] * this->blobs_[0]->shape()[1] 
                        * this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3]); /// further calculated in `net.cpp`, after layer SetUp
    APP<Dtype>::num_param.push_back(count);
    
    // Pruning state {masks, IF_col/row/weight_pruned, num_pruned_col/row/weight, history_prob/score/reg/rank}
    APP<Dtype>::masks.push_back( vector<Dtype>(count, 1) );
    //this->blobmasks.resize(count, 1);
    APP<Dtype>::num_pruned_col.push_back(0);
    APP<Dtype>::num_pruned_row.push_back(0);
    APP<Dtype>::num_pruned_weight.push_back(0);
    APP<Dtype>::IF_row_pruned.push_back( vector<bool>(num_row, false) );
    vector<bool> vec_tmp(this->group_, false); // initialization
    APP<Dtype>::IF_col_pruned.push_back( vector<vector<bool> >(num_col, vec_tmp) );
    APP<Dtype>::IF_weight_pruned.push_back( vector<bool>(count, false) );
    APP<Dtype>::reg_to_distribute.push_back(ceil(prune_param.prune_ratio() * num_col) * APP<Dtype>::target_reg);

    int num_ = num_col; // default, even when not pruning
    if (APP<Dtype>::prune_unit == "Weight") {
        num_ = count;
    } else if (APP<Dtype>::prune_unit == "Col") {
        num_ = num_col;
    } else if (APP<Dtype>::prune_unit == "Row") {
        num_ = num_row;
    }
    APP<Dtype>::hscore.push_back( vector<Dtype>(num_, 0) );
    APP<Dtype>::hrank.push_back( vector<Dtype>(num_, 0) );
    APP<Dtype>::hhrank.push_back( vector<Dtype>(num_, 0) );
    APP<Dtype>::history_prob.push_back( vector<Dtype>(num_, 1) );
    APP<Dtype>::history_reg.push_back( vector<Dtype>(num_, 0) );
    
    
    // Info shared among layers
    APP<Dtype>::filter_area.push_back(this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3]);
    APP<Dtype>::group.push_back(this->group_);
    APP<Dtype>::priority.push_back(prune_param.priority());
    APP<Dtype>::iter_prune_finished.push_back(INT_MAX);

    // Logging
    if (APP<Dtype>::num_log) {
        const int num_log = APP<Dtype>::num_log;
        Dtype rands[num_log];
        caffe_rng_uniform(num_log, (Dtype)0, (Dtype)(num_col - 1), rands);
        APP<Dtype>::log_index.push_back( vector<int>(num_log) );
        for (int i = 0; i < num_log; ++i) {
            APP<Dtype>::log_index[L][i] = int(rands[i]);
        }
        APP<Dtype>::log_weight.push_back( vector<vector<Dtype> >(num_log) );
        APP<Dtype>::log_diff.push_back( vector<vector<Dtype> >(num_log) );
    }

    cout << "=== Masks etc. Initialized" << endl;
}


template <typename Dtype>
bool ConvolutionLayer<Dtype>::IF_hppf() {
    /** IF_higher_priority_prune_finished 
    */
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
void ConvolutionLayer<Dtype>::IF_alpf() {
    /** IF_all_layer_prune_finished
    */
    APP<Dtype>::IF_alpf = true;
    for (int i = 0; i < APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt; ++i) {
        if (APP<Dtype>::iter_prune_finished[i] == INT_MAX) {
            APP<Dtype>::IF_alpf = false;
            break;
        }
    }
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::Print(const int& L, char mode) {
/** print example:
forward:
Index   WeightBeforeMasked   Mask   Prob - conv1
  c 1              0.04044      1      1
  c 2              0.05401      1      1
  c 3              0.06174      1      1

backward:
Index   DiffBeforeMasked   Mask   Prob - conv1
  c 1   0.08216(0.00003)      1      1
  c 2   0.08249(0.00004)      1      1
  c 3   0.08178(0.00007)      1      1
*/
    assert(mode == 'f' || mode = 'b'); /// forward, backward
    const int num_col = this->blobs_[0]->count() / this->blobs_[0]->shape()[0];
    const int num_row = this->blobs_[0]->shape()[0];
    const Dtype* w = this->blobs_[0]->cpu_data();
    const Dtype* d = this->blobs_[0]->cpu_diff();

    // print Index, blob, Mask
    cout.width(5);  cout << "Index" << "   ";
    const string blob = (mode == 'f') ? "WeightBeforeMasked" : "DiffBeforeMasked";
    cout.width(blob.size()); cout << blob << "   ";
    cout.width(4);  cout << "Mask" << "   ";
    
    // print additional info
    char* mthd = new char[strlen(APP<Dtype>::prune_method.c_str()) + 1];
    strcpy(mthd, APP<Dtype>::prune_method.c_str());
    const string mthd_ = strtok(mthd, "_"); // mthd is like "Reg_Col", the first split is `Reg`
    string info = "Unknown";
    vector<Dtype> info_data; 
    if (mthd_ == "Reg") {
        info = "HistoryReg";
        info_data = APP<Dtype>::history_reg[L];
    } else if (mthd_ == "SPP") {
        info = "HistoryProb";
        info_data = APP<Dtype>::history_prob[L];
    }
    cout.width(info.size()); cout << info << " - " << this->layer_param_.name() << endl;
    
    if (APP<Dtype>::prune_unit == "Row") {
        for (int i = 0; i < SHOW_NUM; ++i) {
            // print Index
            cout.width(3); cout << "r"; 
            cout.width(2); cout << i+1 << "   ";
            
            // print blob
            char s[10]; sprintf(s, "%7.5f", d[i * num_col]); // TODO: improve the print for row prune
            if (mode == 'f') { sprintf(s, "%f", fabs(w[i * num_col])); }
            cout.width(blob.size()); cout << s << "   ";
            
            // print Mask
            cout.width(4);  cout << APP<Dtype>::masks[L][i * num_col] << "   ";
            
            // print info
            cout.width(info.size());  cout << info_data[i] << endl;
        }
        
    } else if (APP<Dtype>::prune_unit == "Col") {
        for (int j = 0; j < SHOW_NUM; ++j) {
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
            const Dtype reg_force = APP<Dtype>::history_reg[L][j] * sum_w;
            char s[20]; sprintf(s, "%7.5f(%7.5f)", sum_d, reg_force);
            if (mode == 'f') { sprintf(s, "%f", sum_w); }
            cout.width(blob.size()); cout << s << "   ";
            
            // print Mask
            cout.width(4);  cout << APP<Dtype>::masks[L][j] << "   ";
            
            // print info
            cout.width(info.size());  cout << info_data[j] << endl;
        }
    } else if (APP<Dtype>::prune_unit == "Weight") {
        for (int i = 0; i < SHOW_NUM; ++i) {
            // print Index
            cout.width(3); cout << "w";
            cout.width(2); cout << i+1 << "   ";
            
            // print blob
            const Dtype reg_force = APP<Dtype>::history_reg[L][i] * fabs(w[i]);
            char s[20]; sprintf(s, "%7.5f(%7.5f)", fabs(d[i]), reg_force);
            if (mode == 'f') { sprintf(s, "%f", fabs(w[i])); }
            cout.width(blob.size()); cout << s << "   ";
            
            // print Mask
            cout.width(4);  cout << APP<Dtype>::masks[L][i] << "   ";
            
            // print info
            cout.width(info.size());  cout << info_data[i] << endl;
        }
    }
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdatePrunedRatio() {
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
        APP<Dtype>::pruned_ratio[L] = APP<Dtype>::num_pruned_weight[L] * 1.0 / count;
    } else {
        APP<Dtype>::pruned_ratio[L] =  (APP<Dtype>::pruned_ratio_col[L] + APP<Dtype>::pruned_ratio_row[L]) 
                               - APP<Dtype>::pruned_ratio_col[L] * APP<Dtype>::pruned_ratio_row[L];
    }
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::TaylorPrune(const vector<Blob<Dtype>*>& top) {
    for (int i = 0; i < top.size(); ++i) {
        const Dtype* top_data = top[i]->cpu_data();
        const Dtype* top_diff = top[i]->cpu_diff();
        Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
        const int num_c = top[i]->shape()[1]; /// channel
        const int num_h = top[i]->shape()[2]; /// height
        const int num_w = top[i]->shape()[3]; /// width
        const int count = this->blobs_[0]->count();
        const int num_row = this->blobs_[0]->shape()[0];
        const int num_col = count / num_row;
        const int L = APP<Dtype>::layer_index[this->layer_param_.name()];

        typedef std::pair<Dtype, int> mypair;
        vector<mypair> fm_score(num_c); /// feature map score
        for (int c = 0; c < num_c; ++c) {
            fm_score[c].second = c;
            fm_score[c].first  = 0;
        }
        for (int n = 0; n < this->num_; ++n) {
            for (int c = 0; c < num_c; ++c) {
                for (int i = 0; i < num_h * num_w; ++i) {
                    fm_score[c].first += fabs(top_diff[n * num_c * num_w * num_h + c * num_w * num_h + i] 
                                            * top_data[n * num_c * num_w * num_h + c * num_w * num_h + i]);                          
                }
            }
        }
        for (int c = 0; c < num_c; ++c) {
            if (APP<Dtype>::IF_row_pruned[L][c]) {
                fm_score[c].first = INT_MAX;
            }
        }
        sort(fm_score.begin(), fm_score.end());
        int num_once_prune = 1;
        if (APP<Dtype>::num_once_prune > 1) { num_once_prune = APP<Dtype>::num_once_prune; }
        for (int i = 0; i < num_once_prune; ++i) {
            const int c = fm_score[i].second;
            for (int j = 0; j < num_col; ++j) {
                muweight[c * num_col + j] = 0; /// Seems don't work
                APP<Dtype>::masks[L][c * num_col + j] = 0;
            }
            APP<Dtype>::IF_row_pruned[L][c] = true;
            ++ APP<Dtype>::num_pruned_row[L];
            // if (L != APP<Dtype>::layer_cnt - 1) {
                // APP<Dtype>::pruned_rows.push_back(c); // in TP paper, they didn't mention update corresponding column, so do not do this.
            // }
        }
        if (L == 1) {
            for (int i = 0; i < num_row; ++i) {
                cout << muweight[i*num_col] << " " << endl;
            
            }
        }

    }
    
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::FilterPrune() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];

    typedef pair<Dtype, int> mypair;
    vector<mypair> row_score(num_row);
    for (int i = 0; i < num_row; ++i) {
        row_score[i].second = i; /// index 
        if (APP<Dtype>::IF_row_pruned[L][i]) { 
            row_score[i].first = INT_MAX; /// make those pruned row "float" up
            continue;
        } 
        row_score[i].first  = 0; /// score
        for (int j = 0; j < num_col; ++j) {
            row_score[i].first += fabs(muweight[i * num_col +j]);
        }
    }
    sort(row_score.begin(), row_score.end()); /// in ascending order
    for (int i = 0; i < APP<Dtype>::num_once_prune; ++i) {
        const int r = row_score[i].second;
        for (int j = 0; j < num_col; ++j) {
            muweight[r * num_col + j] = 0;
            APP<Dtype>::masks[L][r * num_col + j] = 0;
        }
        APP<Dtype>::IF_row_pruned[L][r] = true;
        ++ APP<Dtype>::num_pruned_row[L];
        APP<Dtype>::pruned_rows.push_back(r);
    }
} 

template <typename Dtype> 
void ConvolutionLayer<Dtype>::ProbPruneCol() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    const int L = APP<Dtype>::layer_index[layer_name];
    const int num_col_to_prune_ = ceil((APP<Dtype>::prune_ratio[L] + APP<Dtype>::delta[L]) * num_col); /// a little bit higher goal
    const int iter_size = APP<Dtype>::iter_size;
    const Dtype rgamma = APP<Dtype>::rgamma;
    const Dtype rpower = APP<Dtype>::rpower;
    const Dtype cgamma = APP<Dtype>::cgamma;
    const Dtype cpower = APP<Dtype>::cpower;
    const int group = APP<Dtype>::group[L];

    /// Calculate history score
    typedef std::pair<Dtype, int> mypair;
    vector<mypair> col_score(num_col);
    for (int j = 0; j < num_col; ++j) {
        col_score[j].second = j; /// index
        Dtype score = 0;
        for (int i = 0; i < num_row; ++i) {
            score += fabs(muweight[i * num_col +j]);
        }
        APP<Dtype>::hscore[L][j] = APP<Dtype>::score_decay * APP<Dtype>::hscore[L][j] + score;
        col_score[j].first = APP<Dtype>::hscore[L][j];
        if (APP<Dtype>::IF_col_pruned[L][j][0]) { col_score[j].first = INT_MAX; } /// make the pruned columns "float" up
    }
    sort(col_score.begin(), col_score.end());
    
    /// Recover the best columns, according to some probabilities
    Dtype p_recover;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_recover);
    if (rgamma > 0 && pow(rgamma + 0.00027 * APP<Dtype>::step_, rpower) > p_recover * iter_size) {

        /// Print and check
        cout << "recover prob: " << layer_name << "  step: " << APP<Dtype>::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << APP<Dtype>::history_prob[L][col_score[j].second] << " "; }
        cout << "\n";                    
        
        for (int j = num_col_to_prune_ - APP<Dtype>::num_pruned_col[L] - 1; 
                 j < num_col - APP<Dtype>::num_pruned_col[L]; ++j) {
            const int col_of_rank_j = col_score[j].second;
            //cout << "recover col: " << col_of_rank_j 
                 //<< "  its prob: " << APP<Dtype>::history_prob[L][col_of_rank_j] 
                 //<< "  step: " << APP<Dtype>::step_ << endl;
            APP<Dtype>::history_prob[L][col_of_rank_j] = 1;
        }
    }

    /// Update history_prob, according to some probabilities
    Dtype p_prune;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_prune);
    /// if ((APP<Dtype>::step_ - 1) % APP<Dtype>::prune_interval == 0) {  
    if (pow(cgamma + 0.0008 * APP<Dtype>::step_, cpower) > p_prune * iter_size) { 
    /// if (std::min(Dtype(APP<Dtype>::learning_speed), (Dtype)0.004) * 4 > p_prune) {  
    
        /// Print and check
        cout << "update prob: " << layer_name << " step: " << APP<Dtype>::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << APP<Dtype>::history_prob[L][col_score[j].second] << " "; }
        cout << "\n";
    
        /// Calculate functioning probability of each weight
        const Dtype AA = APP<Dtype>::AA; 
        const Dtype aa = AA / 10.0;
        const Dtype alpha = -log(aa/AA) / (num_col_to_prune_ - APP<Dtype>::num_pruned_col[L] - 1);  /// adjust alpha according to the remainder of cloumns
        for (int j = 0; j < num_col_to_prune_ - APP<Dtype>::num_pruned_col[L]; ++j) {               /// note the range of j: only undermine those not-good-enough columns
            const int col_of_rank_j = col_score[j].second;
            APP<Dtype>::history_prob[L][col_of_rank_j] = std::max(APP<Dtype>::history_prob[L][col_of_rank_j] - AA * exp(-j * alpha), (Dtype)0);  
            if (APP<Dtype>::history_prob[L][col_of_rank_j] == 0) {
                APP<Dtype>::num_pruned_col[L] += 1;
                for (int g = 0; g < group; ++g) {
                    APP<Dtype>::IF_col_pruned[L][col_of_rank_j][g] = true;
                }
                for (int i = 0; i < num_row; ++i) { 
                    muweight[i * num_col + col_of_rank_j] = 0;
                } /// once pruned, zero out weights
            }
        } 
    }

    // With probability updated, generate masks and do pruning
    Dtype rands[num_col];
    caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
    for (int i = 0; i < count; ++i) {
        const bool cond1 = rands[i % num_col] < APP<Dtype>::history_prob[L][i % num_col];
        const bool cond2 = !APP<Dtype>::IF_row_pruned[L][i / num_col];
        APP<Dtype>::masks[L][i] = (cond1 && cond2) ? 1 : 0;
        this->weight_backup[i] = muweight[i];
        muweight[i] *= APP<Dtype>::masks[L][i];
    }
    this->IF_restore = true;
    for (int i = 0; i < count; ++i) { 
        muweight[i] *= APP<Dtype>::masks[L][i]; 
    }
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::ProbPruneCol(const int& prune_interval) {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    const int L = APP<Dtype>::layer_index[layer_name];
    const int num_col_to_prune_ = ceil((APP<Dtype>::prune_ratio[L] + APP<Dtype>::delta[L]) * num_col); /// a little bit higher goal
    const int group = APP<Dtype>::group[L];

    /// Calculate history score
    typedef std::pair<Dtype, int> mypair;
    vector<mypair> col_score(num_col);
    for (int j = 0; j < num_col; ++j) {
        col_score[j].second = j; /// index
        Dtype score = 0;
        for (int i = 0; i < num_row; ++i) {
            score += fabs(muweight[i * num_col +j]);
        }
        APP<Dtype>::hscore[L][j] = APP<Dtype>::score_decay * APP<Dtype>::hscore[L][j] + score;
        col_score[j].first = APP<Dtype>::hscore[L][j];
        if (APP<Dtype>::IF_col_pruned[L][j][0]) { col_score[j].first = INT_MAX; } /// make the pruned columns "float" up
    }
    sort(col_score.begin(), col_score.end());
    

    /// Update history_prob
    if ((APP<Dtype>::step_ - 1) % prune_interval == 0 && APP<Dtype>::inner_iter == 0) {
        /// Print and check
        cout << "update prob: " << layer_name << " step: " << APP<Dtype>::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << APP<Dtype>::history_prob[L][col_score[j].second] << " "; }
        cout << "\n";
    
        /// Calculate functioning probability of each weight
        const Dtype AA = APP<Dtype>::AA;
        const Dtype kk = APP<Dtype>::kk;
        const Dtype alpha = log(2/kk) / (num_col_to_prune_ - APP<Dtype>::num_pruned_col[L]);
        const Dtype N1 = -log(kk)/alpha;
        const Dtype k_ = AA / (num_col_to_prune_ - APP<Dtype>::num_pruned_col[L]); /// linear punishment
        
        for (int j = 0; j < num_col - APP<Dtype>::num_pruned_col[L]; ++j) {
            
            const int col_of_rank_j = col_score[j].second;
            Dtype delta = j < N1 ? AA * exp(-alpha * j) : -AA * exp(-alpha * (2*N1-j)) + 2*kk*AA;
            if (APP<Dtype>::prune_method == "PPc_l") {
                delta = AA - k_ * j; /// linear punishment
            }            
            const Dtype old_prob = APP<Dtype>::history_prob[L][col_of_rank_j];
            const Dtype new_prob = std::min(std::max(old_prob - delta, Dtype(0)), Dtype(1));
            APP<Dtype>::history_prob[L][col_of_rank_j] = new_prob;
            
            if (new_prob == 0) {
                APP<Dtype>::num_pruned_col[L] += 1;
                for (int g = 0; g < group; ++g) {
                    APP<Dtype>::IF_col_pruned[L][col_of_rank_j][g] = true;
                }
                for (int i = 0; i < num_row; ++i) { 
                    muweight[i * num_col + col_of_rank_j] = 0;
                } /// once pruned, zero out weights
            }
            
            // Print
            if (new_prob > old_prob) {
                cout << "recover prob: " << layer_name << "-" << col_of_rank_j 
                     << "  old prob: " << old_prob
                     << "  new prob: " << new_prob << endl;
            }
        }
    }

    // With probability updated, generate masks and do pruning
    Dtype rands[num_col];
    caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
    for (int i = 0; i < count; ++i) {
        const bool cond1 = rands[i % num_col] < APP<Dtype>::history_prob[L][i % num_col];
        const bool cond2 = !APP<Dtype>::IF_row_pruned[L][i / num_col];
        APP<Dtype>::masks[L][i] = (cond1 && cond2) ? 1 : 0;
        this->weight_backup[i] = muweight[i];
        muweight[i] *= APP<Dtype>::masks[L][i];
    }
    this->IF_restore = true;
    for (int i = 0; i < count; ++i) { 
        muweight[i] *= APP<Dtype>::masks[L][i]; 
    }
}



template <typename Dtype> 
void ConvolutionLayer<Dtype>::ProbPruneRow() {}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::CleanWorkForPP() {
    /// Once the pruning ratio reached, set all the masks of non-zero prob to 1 and adjust their weights.
    /// Get into here ONLY ONCE.
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;  
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int num_row_per_g = num_row / APP<Dtype>::group[L];
    
    for (int i = 0; i < count; ++i) {
        const int k = (APP<Dtype>::prune_method == "PPr") ? i / num_col : i % num_col;
        const bool cond = (APP<Dtype>::prune_method == "PPr") ? APP<Dtype>::IF_col_pruned[L][i % num_col][i / num_col / num_row_per_g]
                                                       : APP<Dtype>::IF_row_pruned[L][i / num_col]; /// if i is pruned
        if (APP<Dtype>::history_prob[L][k] > 0) {
            muweight[i] *= APP<Dtype>::history_prob[L][k];
            APP<Dtype>::history_prob[L][k] = 1; 
            APP<Dtype>::masks[L][i] = cond ? 0 : 1;
        }
    }
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdateNumPrunedRow() {
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int fanl = APP<Dtype>::filter_area[L+1]; /// filter_area_next_layer
    const int num_col = count / num_row;
    const int num_row_per_g = num_row / APP<Dtype>::group[L+1];
    
    cout << "        " << this->layer_param_.name() << " in UpdateNumPrunedRow" << endl;
    for (int i = 0; i < num_row; ++i) {
        if (!APP<Dtype>::IF_row_pruned[L][i]) {
            const int chl = i % num_row_per_g;
            const int g   = i / num_row_per_g;
            bool IF_consecutive_pruned = true; /// If the corresponding columns in next layer are pruned consecutively, 
                                               /// then this row can be removed.
            for (int j = chl * fanl; j < (chl + 1) * fanl; ++j) {
                if (!APP<Dtype>::IF_col_pruned[L + 1][j][g]) { 
                    IF_consecutive_pruned = false; 
                    break;
                }
            }
            if (IF_consecutive_pruned) {
                for (int j = 0; j < num_col; ++j) {
                    muweight[i * num_col + j] = 0;
                    APP<Dtype>::masks[L][i * num_col + j] = 0;
                }
                APP<Dtype>::IF_row_pruned[L][i] = true;
                ++ APP<Dtype>::num_pruned_row[L];
                cout << " " << this->layer_param_.name() << " prune a row successfully: " << i << endl;
            }
        }
    }
    
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdateNumPrunedCol() {
    
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int num_chl = this->blobs_[0]->shape()[1];
    const int num_row_per_g = num_row / APP<Dtype>::group[L];
    const int filter_area = this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3];
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    
    cout << "        " << this->layer_param_.name() << " in UpdateNumPrunedCol" << endl;
    vector<int>::iterator it;
    for (it = APP<Dtype>::pruned_rows.begin(); it != APP<Dtype>::pruned_rows.end(); ++it) {
        const int chl = *it % num_chl;
        const int g   = *it / num_chl;
        for (int i = g * num_row_per_g; i < (g + 1) * num_row_per_g; ++i) {
            for (int j = chl * filter_area; j < (chl + 1) * filter_area; ++j) {
                APP<Dtype>::masks[L][i * num_col + j] = 0;
                muweight[i * num_col + j] = 0;
                APP<Dtype>::IF_col_pruned[L][j][g] = true;
            }
        }
        APP<Dtype>::num_pruned_col[L] += filter_area * 1.0 / APP<Dtype>::group[L];
        cout << "  " << this->layer_param_.name() << " prune a channel successfully: " << chl << endl;
    }
    APP<Dtype>::pruned_rows.clear();
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::ComputeBlobMask() {
    /** Restore pruning state when retrain
    */
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
    int   num_pruned_row = 0;
    
    if (APP<Dtype>::prune_unit == "Weight") {
        for (int i = 0; i < count; ++i) {
            if (!weight[i]) {
                APP<Dtype>::masks[L][i] = 0;
                ++ APP<Dtype>::num_pruned_weight[L];
                APP<Dtype>::IF_weight_pruned[L][i] = true;
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
                        APP<Dtype>::masks[L][i * num_col + j] = 0;
                    }
                    if (mthd == "PPc") {
                        APP<Dtype>::history_prob[L][j] = 0; /// TODO: count group;
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
                    APP<Dtype>::masks[L][i * num_col + j] = 0; 
                }
                if (mthd == "PPr") {
                    APP<Dtype>::history_prob[L][i] = 0; /// TODO: count group;
                }
            }
            
        }
        APP<Dtype>::num_pruned_col[L] = num_pruned_col;
        APP<Dtype>::num_pruned_row[L] = num_pruned_row;
        
    }
    UpdatePrunedRatio();
    
    Dtype pruned_ratio;
    if (APP<Dtype>::prune_unit == "Weight")   { pruned_ratio = APP<Dtype>::pruned_ratio[L];     }
    else if (APP<Dtype>::prune_unit == "Row") { pruned_ratio = APP<Dtype>::pruned_ratio_row[L]; }
    else if (APP<Dtype>::prune_unit == "Col") { pruned_ratio = APP<Dtype>::pruned_ratio_col[L]; }
    if (pruned_ratio >= APP<Dtype>::prune_ratio[L]) {
        APP<Dtype>::iter_prune_finished[L] = -1; /// To check multi-GPU
        cout << layer_name << " prune finished" << endl;
    } else { 
        if (APP<Dtype>::prune_coremthd == "SPP" || APP<Dtype>::prune_coremthd == "Reg") {
            RestorePruneProb(pruned_ratio);
        }
    }
    LOG(INFO) << "    Masks restored, num_pruned_col = " << APP<Dtype>::num_pruned_col[L]
              << "  num_pruned_row = " << APP<Dtype>::num_pruned_row[L]
              << "  pruned_ratio = "   << APP<Dtype>::pruned_ratio[L]
              << "  prune_ratio = "    << APP<Dtype>::prune_ratio[L];
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::PruneMinimals() {
    Dtype* muweight   = this->blobs_[0]->mutable_cpu_data();
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    const int group = APP<Dtype>::group[L];
    
    if (APP<Dtype>::prune_unit == "Weight") {
        for (int i = 0; i < count; ++i) {
            if (APP<Dtype>::IF_weight_pruned[L][i]) { continue; }
            if (fabs(muweight[i]) < APP<Dtype>::prune_threshold || APP<Dtype>::history_reg[L][i] >= APP<Dtype>::target_reg) {
                muweight[i] = 0;
                APP<Dtype>::masks[L][i] = 0;
                APP<Dtype>::num_pruned_weight[L] += 1;
                APP<Dtype>::IF_weight_pruned[L][i] = true;
                APP<Dtype>::hrank[L][i]  = APP<Dtype>::step_ - 1000000 - (APP<Dtype>::history_reg[L][i] - APP<Dtype>::target_reg);
                APP<Dtype>::hhrank[L][i] = APP<Dtype>::step_ - 1000000 - (APP<Dtype>::history_reg[L][i] - APP<Dtype>::target_reg); 
            }
        }
    } else if (APP<Dtype>::prune_unit == "Col") {
        for (int j = 0; j < num_col; ++j) {
            if (APP<Dtype>::IF_col_pruned[L][j][0]) { continue; }
            // bool IF_all_weights_are_small = true;
            Dtype sum = 0;
            for (int i = 0; i < num_row; ++i) {
                sum += fabs(muweight[i * num_col + j]);
            }
            sum /= num_row;
            if (sum < APP<Dtype>::prune_threshold ||  APP<Dtype>::history_reg[L][j] >= APP<Dtype>::target_reg) {
                for (int i = 0; i < num_row; ++i) {
                    muweight[i * num_col + j] = 0;
                    APP<Dtype>::masks[L][i * num_col + j] = 0; 
                }
                APP<Dtype>::num_pruned_col[L] += 1;
                for (int g = 0; g < group; ++g) {
                    APP<Dtype>::IF_col_pruned[L][j][g] = true;
                }
                APP<Dtype>::hrank[L][j] = APP<Dtype>::step_ - 1000000 - (APP<Dtype>::history_reg[L][j] - APP<Dtype>::target_reg);  // the worser column, the earlier pruned column will be ranked in fronter
                
            }        
        }
    } else {
        cerr << "prune_unit is not Col and Weight, please check: " << __FILE__ << __LINE__ << endl;
        exit(1);
    }
}



template <typename Dtype>
void ConvolutionLayer<Dtype>::RestorePruneProb(const Dtype& pruned_r) {
    const string layer_name = this->layer_param_.name();
    const int L = APP<Dtype>::layer_index[layer_name];    
    
    const string inFile = APP<Dtype>::snapshot_prefix + APP<Dtype>::prune_state_dir + layer_name + ".txt"; /// TODO: paramaters check
    ifstream state_stream;
    state_stream.open(inFile.data());
    if (!state_stream.is_open()) {
        LOG(INFO) << "Error, failed to restore prune_prob, the prune_prob file cannot be opened: `" 
             << inFile << "`" << endl;
    } else {
        string line;
        getline(state_stream, line); /// the first line is iteration
        vector<Dtype> state;
        while (getline(state_stream, line, ' ')) {
            state.push_back(atof(line.c_str()));
        }
        if (APP<Dtype>::prune_coremthd == "SPP"){
            assert(state.size() == APP<Dtype>::history_prob[L].size());
            for (int i = 0; i < state.size(); ++i) {
                APP<Dtype>::history_prob[L][i] = state[i];
            }
        } else if (APP<Dtype>::prune_coremthd == "Reg") {
            const int vsize = APP<Dtype>::hrank[L].size();
            assert(state.size() == 2 * vsize);
            for (int i = 0; i < vsize; ++i) {
                APP<Dtype>::hrank[L][i] = state[i];
                APP<Dtype>::history_reg[L][i] = state[vsize + i];
            }
        }
        LOG(INFO) << layer_name << " Restore prune state done!";
    }

}


template <typename Dtype>
Dtype ConvolutionLayer<Dtype>::normal_random() {
    static Dtype V1, V2, S;
    static int phase = 0;
    Dtype X;
    if (phase == 0) {
        do {
            Dtype U1 = (Dtype) rand() / RAND_MAX;
            Dtype U2 = (Dtype) rand() / RAND_MAX;
            V1 = 2 * U1 - 1;
            V2 = 2 * U2 - 1;
            S = V1 * V1 + V2 * V2;
        } while (S >= 1 || S == 0);  /// loop until 0<S<1
        X = V1 * sqrt(-2 * log(S) / S);
    } else {
        X = V2 * sqrt(-2 * log(S) / S);
    }
    phase = 1 - phase;
    return X * 0.05;
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    /// i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data(); 

    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
          top_data + n * this->top_dim_);

      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }
  }
}



template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  const Dtype* weight = this->blobs_[0]->cpu_data(); /// weight用来计算底层的梯度dx = dz * w
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

  for (int i = 0; i < top.size(); ++i) { /// 对于top层中的每个神经元
    const Dtype* top_diff = top[i]->cpu_diff(); /// top_diff是dz
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();

    /// Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) { /// num_是在base_conv中定义的
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }


    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        /// gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff); /// calculate weight_diff for this layer
        }
        
        /// gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_); /// dx = dz * w
        }
      }
    }
  }

}


#ifdef CPU_ONLY
STUB_GPU(ConvolutionLayer);
#endif

INSTANTIATE_CLASS(ConvolutionLayer);

}  /// namespace caffe
