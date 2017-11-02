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
        if (APP::layer_index.count(layer_name) == 0) {
            APP::layer_index[layer_name] = APP::layer_cnt;
            ++ APP::layer_cnt;
            cout << "A new layer registered: " << layer_name 
                 << "  Total #layer: " << APP::layer_cnt << endl;
        }
    } else { return; }
    const int L = APP::layer_index[layer_name];
    const string mthd = APP::prune_method;
    cout << "PruneSetUp: " << layer_name  
         << "  its layer_index: " << L
         << "  Total #layer: " << APP::layer_cnt << endl;
    
    
    // Note: the varibales below can ONLY be used in training.
    // Note: These varibales will be called for every GPU, whereas since we use `layer_index` to index, so it doesn't matter.
    // Set up prune parameters of layer
    APP::prune_ratio.push_back(prune_param.prune_ratio());
    APP::delta.push_back(prune_param.delta());
    APP::pruned_ratio.push_back(0);
    APP::pruned_ratio_col.push_back(0);
    APP::pruned_ratio_row.push_back(0);
    APP::GFLOPs.push_back(this->blobs_[0]->shape()[0] * this->blobs_[0]->shape()[1] 
                        * this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3]); /// further calculated in `net.cpp`, after layer SetUp
    
    // Info shared among different layers
    // Pruning state
    APP::masks.push_back( vector<bool>(count, 1) );
    APP::num_pruned_col.push_back(0);
    APP::num_pruned_row.push_back(0);
    APP::IF_row_pruned.push_back( vector<bool>(num_row, false) );
    vector<bool> vec_tmp(this->group_, false); /// initialization
    APP::IF_col_pruned.push_back( vector<vector<bool> >(num_col, vec_tmp) );
    
    const int num_ = (mthd == "PPr") ? num_row : num_col;
    APP::history_prob.push_back(  vector<float>(num_, 1) );
    APP::history_score.push_back( vector<float>(num_, 0) );
    
    
    // Info shared among layers
    APP::filter_area.push_back(this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3]);
    APP::group.push_back(this->group_);
    APP::priority.push_back(prune_param.priority());
    APP::iter_prune_finished.push_back(INT_MAX);

    
    // Logging
    if (APP::num_log) {
        const int num_log = APP::num_log;
        Dtype rands[num_log];
        caffe_rng_uniform(num_log, (Dtype)0, (Dtype)(num_col - 1), rands);
        APP::log_index.push_back( vector<int>(num_log) );
        for (int i = 0; i < num_log; ++i) {
            APP::log_index[L][i] = int(rands[i]);
        }
        APP::log_weight.push_back( vector<vector<float> >(num_log) );
        APP::log_diff.push_back( vector<vector<float> >(num_log) );
    }

    cout << "=== Masks etc. Initialized" << endl;
}


template <typename Dtype>
bool ConvolutionLayer<Dtype>::IF_hppf() {
    /** IF_higher_priority_prune_finished 
    */
    bool IF_hppf = true;
    const int L = APP::layer_index[this->layer_param_.name()];
    for (int i = 0; i < APP::layer_cnt; ++i) {
        if (APP::priority[i] < APP::priority[L] && APP::iter_prune_finished[i] == INT_MAX) {
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
    APP::IF_alpf = true;
    for (int i = 0; i < APP::layer_cnt; ++i) {
        if (APP::iter_prune_finished[i] == INT_MAX) {
            APP::IF_alpf = false;
            break;
        }
    }
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::Print(const int& L, char mode) {
    assert(mode == 'f' || mode = 'b'); /// forward, backward
    const int num_col = this->blobs_[0]->count() / this->blobs_[0]->shape()[0];
    const string mthd = APP::prune_method;
    const bool IF_prune_row = (mthd == "PPr" || mthd == "FP" || mthd == "TP");
    Dtype* blob;
    int length;
    string content;
    if (mode == 'f') {
        blob = this->blobs_[0]->mutable_cpu_data();
        length = 18;
        content = "WeightBeforeMasked";
    } else {
        blob = this->blobs_[0]->mutable_cpu_diff();
        length = 16;
        content = "DiffBeforeMasked";
    }
    cout << this->layer_param_.name() << endl;
    cout.width(5);  cout << "Index" << "   ";
    cout.width(length); cout << content << "   ";
    cout.width(4);  cout << "Mask" << "   ";
    cout.width(4);  cout << "Prob" << endl;
    for (int i = 0; i < 20; ++i) {
        cout.width(3);  
        if (IF_prune_row) {
            cout << "r";
            cout.width(2);  cout << i+1 << "   ";
            /// i denotes row
            cout.width(length); cout << blob[i * num_col] << "   ";
            cout.width(4);  cout << APP::masks[L][i * num_col] << "   ";
        } else {
            cout << "c";
            cout.width(2);  cout << i+1 << "   ";
            /// i denotes column
            cout.width(length); cout << blob[i] << "   ";
            cout.width(4);  cout << APP::masks[L][i] << "   ";
        }
        cout.width(4);  cout << APP::history_prob[L][i] << endl;
    }
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdatePrunedRatio() {
    const int L = APP::layer_index[this->layer_param_.name()];
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    
    APP::pruned_ratio_col[L] = APP::num_pruned_col[L] / num_col;
    APP::pruned_ratio_row[L] = APP::num_pruned_row[L] * 1.0 / num_row;
    APP::pruned_ratio[L] =  (APP::pruned_ratio_col[L] + APP::pruned_ratio_row[L]) 
                           - APP::pruned_ratio_col[L] * APP::pruned_ratio_row[L];
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
        const int L = APP::layer_index[this->layer_param_.name()];

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
            if (this->IF_row_pruned[c]) {
                fm_score[c].first = INT_MAX;
            }
        }
        sort(fm_score.begin(), fm_score.end());
        int num_once_prune = 1;
        if (APP::num_once_prune > 1) { num_once_prune = APP::num_once_prune; }
        for (int i = 0; i < num_once_prune; ++i) {
            const int c = fm_score[i].second;
            for (int j = 0; j < num_col; ++j) {
                muweight[c * num_col + j] = 0; /// Seems don't work
                APP::masks[L][c * num_col + j] = 0;
            }
            APP::IF_row_pruned[L][c] = true;
            // APP::pruned_rows.push_back(c); // To check if right
            ++ APP::num_pruned_row[L];
        }
    }
    
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::FilterPrune() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int L = APP::layer_index[this->layer_param_.name()];

    typedef pair<Dtype, int> mypair;
    vector<mypair> row_score(num_row);
    for (int i = 0; i < num_row; ++i) {
        row_score[i].second = i; /// index 
        if (this->IF_row_pruned[i]) { 
            row_score[i].first = INT_MAX; /// make those pruned row "float" up
            continue;
        } 
        row_score[i].first  = 0; /// score
        for (int j = 0; j < num_col; ++j) {
            row_score[i].first += fabs(muweight[i * num_col +j]);
        }
    }
    sort(row_score.begin(), row_score.end()); /// in ascending order
    for (int i = 0; i < APP::num_once_prune; ++i) {
        const int r = row_score[i].second;
        for (int j = 0; j < num_col; ++j) {
            muweight[r * num_col + j] = 0;
            APP::masks[L][r * num_col + j] = 0;
        }
        APP::IF_row_pruned[L][r] = true;
        APP::pruned_rows.push_back(r);
        ++ APP::num_pruned_row[L];
    }
    
    
} 

template <typename Dtype> 
void ConvolutionLayer<Dtype>::ProbPruneCol() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    const int L = APP::layer_index[layer_name];
    const int num_col_to_prune_ = ceil((APP::prune_ratio[L] + APP::delta[L]) * num_col); /// a little bit higher goal
    const int iter_size = APP::iter_size;
    const Dtype rgamma = APP::rgamma;
    const Dtype rpower = APP::rpower;
    const Dtype cgamma = APP::cgamma;
    const Dtype cpower = APP::cpower;
    const int group = APP::group[L];

    /// Calculate history score
    typedef std::pair<Dtype, int> mypair;
    vector<mypair> col_score(num_col);
    for (int j = 0; j < num_col; ++j) {
        col_score[j].second = j; /// index
        Dtype score = 0;
        for (int i = 0; i < num_row; ++i) {
            score += fabs(muweight[i * num_col +j]);
        }
        APP::history_score[L][j] = APP::score_decay * APP::history_score[L][j] + score;
        col_score[j].first = APP::history_score[L][j];
        if (APP::IF_col_pruned[L][j][0]) { col_score[j].first = INT_MAX; } /// make the pruned columns "float" up
    }
    sort(col_score.begin(), col_score.end());
    
    /// Recover the best columns, according to some probabilities
    Dtype p_recover;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_recover);
    if (rgamma > 0 && pow(rgamma + 0.00027 * APP::step_, rpower) > p_recover * iter_size) {

        /// Print and check
        cout << "recover prob: " << layer_name << "  step: " << APP::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << APP::history_prob[L][col_score[j].second] << " "; }
        cout << "\n";                    
        
        for (int j = num_col_to_prune_ - APP::num_pruned_col[L] - 1; 
                 j < num_col - APP::num_pruned_col[L]; ++j) {
            const int col_of_rank_j = col_score[j].second;
            //cout << "recover col: " << col_of_rank_j 
                 //<< "  its prob: " << APP::history_prob[L][col_of_rank_j] 
                 //<< "  step: " << APP::step_ << endl;
            APP::history_prob[L][col_of_rank_j] = 1;
        }
    }

    /// Update history_prob, according to some probabilities
    Dtype p_prune;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_prune);
    /// if ((APP::step_ - 1) % APP::prune_interval == 0) {  
    if (pow(cgamma + 0.0008 * APP::step_, cpower) > p_prune * iter_size) { 
    /// if (std::min(Dtype(APP::learning_speed), (Dtype)0.004) * 4 > p_prune) {  
    
        /// Print and check
        cout << "update prob: " << layer_name << " step: " << APP::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << APP::history_prob[L][col_score[j].second] << " "; }
        cout << "\n";
    
        /// Calculate functioning probability of each weight
        const Dtype AA = APP::AA; 
        const Dtype aa = AA / 10.0;
        const Dtype alpha = -log(aa/AA) / (num_col_to_prune_ - APP::num_pruned_col[L] - 1);  /// adjust alpha according to the remainder of cloumns
        for (int j = 0; j < num_col_to_prune_ - APP::num_pruned_col[L]; ++j) {               /// note the range of j: only undermine those not-good-enough columns
            const int col_of_rank_j = col_score[j].second;
            APP::history_prob[L][col_of_rank_j] = std::max(APP::history_prob[L][col_of_rank_j] - AA * exp(-j * alpha), (Dtype)0);  
            if (APP::history_prob[L][col_of_rank_j] == 0) {
                APP::num_pruned_col[L] += 1;
                for (int g = 0; g < group; ++g) {
                    APP::IF_col_pruned[L][col_of_rank_j][g] = true;
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
        const bool cond1 = rands[i % num_col] < APP::history_prob[L][i % num_col];
        const bool cond2 = !APP::IF_row_pruned[L][i / num_col];
        APP::masks[L][i] = (cond1 && cond2) ? 1 : 0;
        this->weight_backup[i] = muweight[i];
        muweight[i] *= APP::masks[L][i];
    }
    this->IF_restore = true;
    for (int i = 0; i < count; ++i) { 
        muweight[i] *= APP::masks[L][i]; 
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
    const int L = APP::layer_index[this->layer_param_.name()];
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int num_row_per_g = num_row / APP::group[L];
    
    for (int i = 0; i < count; ++i) {
        const int k = (APP::prune_method == "PPr") ? i / num_col : i % num_col;
        const bool cond = (APP::prune_method == "PPr") ? APP::IF_col_pruned[L][i % num_col][i / num_col / num_row_per_g]
                                                       : APP::IF_row_pruned[L][i / num_col]; /// if i is pruned
        if (APP::history_prob[L][k] > 0) {
            muweight[i] *= APP::history_prob[L][k];
            APP::history_prob[L][k] = 1; 
            APP::masks[L][i] = cond ? 0 : 1;
        }
    }
}


template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdateNumPrunedRow() {
    const int L = APP::layer_index[this->layer_param_.name()];
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int fanl = APP::filter_area[L+1]; /// filter_area_next_layer
    const int num_col = count / num_row;
    const int num_row_per_g = num_row / APP::group[L+1];
    
    cout << this->layer_param_.name() << " in UpdateNumPrunedRow" << endl;
    for (int i = 0; i < num_row; ++i) {
        if (!APP::IF_row_pruned[L][i]) {
            const int chl = i % num_row_per_g;
            const int g   = i / num_row_per_g;
            bool IF_consecutive_pruned = true; /// If the corresponding columns in next layer are pruned consecutively, 
                                               /// then this row can be removed.
            for (int j = chl * fanl; j < (chl + 1) * fanl; ++j) {
                if (!APP::IF_col_pruned[L + 1][j][g]) { 
                    IF_consecutive_pruned = false; 
                    break;
                }
            }
            if (IF_consecutive_pruned) {
                for (int j = 0; j < num_col; ++j) {
                    muweight[i * num_col + j] = 0;
                    APP::masks[L][i * num_col + j] = 0;
                }
                APP::IF_row_pruned[L][i] = true;
                ++ APP::num_pruned_row[L];
                cout << " " << this->layer_param_.name() << " prune a row successfully: " << i << endl;
            }
        }
    }
    
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdateNumPrunedCol() {
    if (!APP::pruned_rows.size()) { return; }
    
    const int L = APP::layer_index[this->layer_param_.name()];
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int num_chl = this->blobs_[0]->shape()[1];
    const int num_row_per_g = num_row / APP::group[L];
    const int filter_area = this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3];
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    
    cout << this->layer_param_.name() << " in UpdateNumPrunedCol" << endl;
    vector<int>::iterator it;
    for (it = APP::pruned_rows.begin(); it != APP::pruned_rows.end(); ++it) {
        const int chl = *it % num_chl;
        const int g   = *it / num_chl;
        for (int i = g * num_row_per_g; i < (g + 1) * num_row_per_g; ++i) {
            for (int j = chl * filter_area; j < (chl + 1) * filter_area; ++j) {
                APP::masks[L][i * num_col + j] = 0;
                muweight[i * num_col + j] = 0;
                APP::IF_col_pruned[L][j][g] = true;
            }
        }
        APP::num_pruned_col[L] += filter_area * 1.0 / APP::group[L];
        cout << "  " << this->layer_param_.name() << " prune a channel successfully: " << chl << endl;
    }
    APP::pruned_rows.clear();
}


template <typename Dtype>
void ConvolutionLayer<Dtype>::ComputeBlobMask(float ratio) {
    /** Restore pruning state when retrain
    */
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const Dtype *weight = this->blobs_[0]->cpu_data();
    const string layer_name = this->layer_param_.name();
    const int L = APP::layer_index[layer_name];
    const int group = APP::group[L];
    const int num_row_per_g = num_row / group;
    const string mthd = APP::prune_method;

    Dtype num_pruned_col = 0;
    int   num_pruned_row = 0;

    // Column
    for (int j = 0; j < num_col; ++j) {
        for (int g = 0; g < group; ++g) {
            Dtype sum = 0;
            for (int i = g * num_row_per_g; i < (g+1) * num_row_per_g; ++i) { 
                sum += fabs(weight[i * num_col + j]);
            }
            if (sum == 0) { 
                num_pruned_col += 1.0 / group; /// note that num_pruned_row is always integer while num_pruned_col can be non-integer.
                APP::IF_col_pruned[L][j][g] = true;
                for (int i = g * num_row_per_g; i < (g+1) * num_row_per_g; ++i) { 
                    APP::masks[L][i * num_col + j] = 0; 
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
            APP::IF_row_pruned[L][i] = true;
            for (int j = 0; j < num_col; ++j) { 
                APP::masks[L][i * num_col + j] = 0; 
            }
        }
    }
    
    APP::num_pruned_col[L] = num_pruned_col;
    APP::num_pruned_row[L] = num_pruned_row;
    UpdatePrunedRatio();
    const Dtype pruned_ratio = (mthd == "PPr" || mthd == "FP" || mthd == "TP") 
                                    ? APP::pruned_ratio_row[L] : APP::pruned_ratio_col[L];
    if (pruned_ratio >= APP::prune_ratio[L]) {
        APP::iter_prune_finished[L] = APP::step_ - 1; /// To check multi-GPU
    } else if (APP::prune_method.substr(0, 2) == "PP") { 
        // Restore prune prob
        const string infile = APP::snapshot_prefix + "prob_snapshot/prob_" + layer_name + ".txt";
        ifstream prob;
        prob.open(infile.data());
        string line;
        vector<float> pr;
        if (!prob.is_open()) {
            if (pruned_ratio) {
                cout << "Error: the layer's prune suspended, while the prune_prob file cannot be opened! " 
                     << infile << endl;
            }
        } else {
            getline(prob, line); /// the first line is iteration
            while (getline(prob, line, ' ')) {
                pr.push_back(atof(line.c_str()));
            }
            assert(pr.size() == APP::history_prob[L].size());
            for (int i = 0; i < pr.size(); ++i) {
                APP::history_prob[L][i] = pr[i];
            }
            cout << "  Prune Prob Restored!" << endl;
        }
    }
    LOG(INFO) << "    Masks restored, num_pruned_col = " << APP::num_pruned_col[L]
              << "  num_pruned_row = " << APP::num_pruned_row[L]
              << "  pruned_ratio = "   << APP::pruned_ratio[L]
              << "  prune_ratio = "    << APP::prune_ratio[L];
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
