#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/deep_compression.hpp"
#include <cstdlib>
#include <cmath>
#define NSUM 50
#define SHOW_INTERVAL 20
#define SHOW_NUM 20

namespace caffe {
using namespace std;

template <typename Dtype>
void ConvolutionLayer<Dtype>::PruneSetUp(const PruneParameter& prune_param) {
    /// set up prune parameters
    this->prune_ratio = prune_param.prune_ratio();
    this->delta = prune_param.delta();
    this->pruned_ratio = 0;
    
    /// set up masks etc.
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    
    CHECK_EQ(layer_name[0], '[') 
        << "Error: The layer name is not started with layer_index in the format of '[3]conv4', please convert your prototxt.";
    
    int i = 1;
    this->layer_index = 0;
    while (layer_name[i] != ']') { 
        this->layer_index = this->layer_index * 10 + layer_name[i] - '0';
        ++ i; 
    }
    DeepCompression::filter_area[this->layer_index] = this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3];
    DeepCompression::group[this->layer_index] = this->group_;
    
    this->masks_.resize(count, 1);
    this->weight_backup.resize(count);
    this->history_prob.resize(num_col, 1);
    DeepCompression::history_prob[this->layer_index].resize(count, 1);
    this->history_score.resize(num_col, 0);

    /// Pruning state info
    this->num_pruned_col = 0;
    this->num_pruned_row = 0;
    this->IF_col_pruned.resize(num_col, false);
    this->IF_row_pruned.resize(num_row, false);
    DeepCompression::IF_row_pruned[this->layer_index].resize(num_row, false);
    DeepCompression::IF_col_pruned[this->layer_index].resize(num_col, false);
    this->num_pruned_weight = 0;
    this->candidate_window.resize(DeepCompression::window_size, -1);

    this->drop_column_masks.resize(count, 1);
    this->history_diff.resize(count, 0);
    this->history_weight.resize(count, 0);

    this->blobs_[0]->mutable_cpu_second_diff = new Dtype[count];
    for (int i = 0; i < count; ++i) {
        this->blobs_[0]->mutable_cpu_second_diff[i] = 0;
    }
    if (num_col * this->prune_ratio > DeepCompression::max_num_column_to_prune) {
        DeepCompression::max_num_column_to_prune = num_col * this->prune_ratio;
    }
    cout << "=== Masks etc. Initialized." << endl;
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
        if (DeepCompression::num_once_prune > 1) { num_once_prune = DeepCompression::num_once_prune; }
        for (int i = 0; i < num_once_prune; ++i) {
            const int c = fm_score[i].second;
            for (int j = 0; j < num_col; ++j) {
                muweight[c * num_col + j] = 0; /// Seems don't work
                this->masks_[c * num_col + j] = 0;
            }
            this->IF_row_pruned[c] = true;
            DeepCompression::IF_row_pruned[this->layer_index][c] = true;
            ++ this->num_pruned_row;
            ++ DeepCompression::num_pruned_row[this->layer_index];
        }
    }
}
    

template <typename Dtype> 
void ConvolutionLayer<Dtype>::ProbPrune() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const string layer_name = this->layer_param_.name();
    const int num_col_to_prune_ = ceil((this->prune_ratio + this->delta) * num_col); /// a little bit higher goal
    const int iter_size = DeepCompression::iter_size;
    const Dtype rgamma = DeepCompression::rgamma;
    const Dtype rpower = DeepCompression::rpower;
    const Dtype cgamma = DeepCompression::cgamma;
    const Dtype cpower = DeepCompression::cpower;
    
    
    /// Calculate history score
    typedef std::pair<Dtype, int> mypair;
    vector<mypair> col_score(num_col);
    for (int j = 0; j < num_col; ++j) {
        col_score[j].second = j; /// index
        Dtype score = 0;
        for (int i = 0; i < num_row; ++i) {
            score += fabs(muweight[i * num_col +j]);
        }
        this->history_score[j] = DeepCompression::score_decay * this->history_score[j] + score;
        col_score[j].first = this->history_score[j];
        if (this->IF_col_pruned[j]) { col_score[j].first = INT_MAX; } /// make the pruned columns "float" up
    }
    sort(col_score.begin(), col_score.end());
    
    /// Recover the best columns, according to some probabilities
    Dtype p_recover;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_recover);
    if (pow(rgamma + 0.00027 * DeepCompression::step_, rpower) > p_recover * iter_size) {

        /// Print and check
        cout << "recover prob: " << layer_name << "  step: " << DeepCompression::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << DeepCompression::history_prob[this->layer_index][col_score[j].second] << " "; }
        cout << "\n";                    
        
        for (int j = num_col_to_prune_ - DeepCompression::num_pruned_col[this->layer_index] - 1; 
                 j < num_col - DeepCompression::num_pruned_col[this->layer_index]; ++j) {
            const int col_of_rank_j = col_score[j].second;
            cout << "recover col: " << col_of_rank_j 
                 << "  its prob: " << DeepCompression::history_prob[this->layer_index][col_of_rank_j] 
                 << "  step: " << DeepCompression::step_ << endl;
            DeepCompression::history_prob[this->layer_index][col_of_rank_j] = 1;
        }
    }

    /// Update history_prob, according to some probabilities
    Dtype p_prune;
    caffe_rng_uniform(1, (Dtype)0, (Dtype)1, &p_prune);
    /// if ((DeepCompression::step_ - 1) % DeepCompression::prune_interval == 0) {  
    if (pow(cgamma + 0.0008 * DeepCompression::step_, cpower) > p_prune * iter_size) { 
    /// if (std::min(Dtype(DeepCompression::learning_speed), (Dtype)0.004) * 4 > p_prune) {  
    
        /// Print and check
        cout << "update prob: " << layer_name << " step: " << DeepCompression::step_ << endl;
        cout << " score: ";   for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].first  << " "; }
        cout << "\ncolumn: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << col_score[j].second << " "; }
        cout << "\n  prob: "; for (int j = 0; j < SHOW_NUM; ++j) { cout << DeepCompression::history_prob[this->layer_index][col_score[j].second] << " "; }
        cout << "\n";
    
        /// Calculate functioning probability of each weight
        const Dtype AA = 0.05; 
        const Dtype aa = 0.0041;
        const Dtype alpha = -log(aa/AA) / (num_col_to_prune_ - DeepCompression::num_pruned_col[this->layer_index] - 1);  /// adjust alpha according to the remainder of cloumns
        for (int j = 0; j < num_col_to_prune_ - DeepCompression::num_pruned_col[this->layer_index]; ++j) {               /// note the range of j: only undermine those not-good-enough columns
            const int col_of_rank_j = col_score[j].second;
            DeepCompression::history_prob[this->layer_index][col_of_rank_j] = std::max(DeepCompression::history_prob[this->layer_index][col_of_rank_j] - AA * exp(-j * alpha), (Dtype)0);
            
            if (DeepCompression::history_prob[this->layer_index][col_of_rank_j] == 0) {
                ++ DeepCompression::num_pruned_col[this->layer_index];
                ++ this->num_pruned_col;
                this->IF_col_pruned[col_of_rank_j] = true;
                DeepCompression::IF_col_pruned[this->layer_index][col_of_rank_j] = true;
                
                for (int i = 0; i < num_row; ++i) { 
                    muweight[i * num_col + col_of_rank_j] = 0; 
                } /// once pruned, zero out weights
            }
        } 
    }

    
    /// With probability updated, generate masks
    Dtype rands[num_col];
    caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
    for (int i = 0; i < count; ++i) {
        this->masks_[i] = rands[i % num_col] < DeepCompression::history_prob[this->layer_index][i % num_col] ? 1 : 0; /// generate masks
    }              
    for (int i = 0; i < count; ++i) { this->weight_backup[i] = muweight[i]; }
    this->IF_restore = true;
    
    /// Print and check (before pruning)
    if (this->layer_index == 1 && DeepCompression::step_ % SHOW_INTERVAL == 0) {
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
            cout.width(4);  cout << DeepCompression::history_prob[this->layer_index][i] << endl;
        }
    }
    for (int i = 0; i < count; ++i) { muweight[i] *= this->masks_[i]; } /// do pruning

}
template <typename Dtype> 
void ConvolutionLayer<Dtype>::CleanWorkForPP() {
    /// Once the pruning ratio reached, set all the masks of non-zero prob to 1 and adjust their weights.
    /// Get into here ONLY ONCE.
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    for (int i = 0; i < count; ++i) {
        if (DeepCompression::history_prob[this->layer_index][i % num_col] > 0) {
            muweight[i] *= DeepCompression::history_prob[this->layer_index][i % num_col];
            DeepCompression::history_prob[this->layer_index][i % num_col] = 1;
        }
    }
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdateNumPrunedRow() {
    if (this->layer_index == DeepCompression::max_layer_index || DeepCompression::IF_prune_finished[this->layer_index + 1]) { return; }
    
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int filter_area_next_layer = DeepCompression::filter_area[this->layer_index+1]; 
    const int num_col = count / num_row;
    
    cout << "conv" << this->layer_index + 1 << " in UpdateNumPrunedRow" << endl;
    for (int i = 0; i < num_row; ++i) {
        if (!this->IF_row_pruned[i]) {
            const int i_ = i % (num_row / DeepCompression::group[this->layer_index + 1]);
            bool IF_consecutive_pruned = true;
            for (int j = i_ * filter_area_next_layer; j < (i_ + 1) * filter_area_next_layer; ++j) {
                if (!DeepCompression::IF_col_pruned[this->layer_index + 1][j]) { 
                    IF_consecutive_pruned = false; 
                    break;
                }
            }
            if (IF_consecutive_pruned) {
                for (int j = 0; j < num_col; ++j) {
                    muweight[i * num_col + j] = 0;
                    this->masks_[i * num_col + j] = 0;
                }
                this->IF_row_pruned[i] = true;
                DeepCompression::IF_row_pruned[this->layer_index][i] = true;
                ++ this->num_pruned_row;
                ++ DeepCompression::num_pruned_row[this->layer_index];
                cout << "conv" << this->layer_index + 1 << " prune a row succeed: " << i << endl;
            }
        }
    }
    
    
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::UpdateNumPrunedCol() {
    if (this->layer_index == 0 || DeepCompression::IF_prune_finished[this->layer_index - 1]) { return; }
        
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int filter_area = this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3];
    const int num_col = count / num_row;
    
    cout << "conv" << this->layer_index+1 << " in UpdateNumPrunedCol" << endl;
    for (int j = 0; j < num_col; ++j) {
        if (!(this->IF_col_pruned[j]) && DeepCompression::IF_row_pruned[this->layer_index - 1][j / filter_area]) {
            for (int i = 0; i < num_row; ++i) { 
                muweight[i * num_col + j] = 0;
                this->masks_[i * num_col + j] = 0;
            }
            this->IF_col_pruned[j] = true;
            DeepCompression::IF_col_pruned[this->layer_index][j] = true;
            ++ this->num_pruned_col;
            ++ DeepCompression::num_pruned_col[this->layer_index];
            cout << "conv" << this->layer_index+1 << " prune a col succeed: " << j << endl;
        }
    }
    
}

template <typename Dtype> 
void ConvolutionLayer<Dtype>::FilterPrune() {
    Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
     
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
    for (int i = 0; i < DeepCompression::num_once_prune; ++i) {
        for (int j = 0; j < num_col; ++j) {
            muweight[row_score[i].second * num_col + j] = 0;
            this->masks_[row_score[i].second * num_col + j] = 0;
        }
        this->IF_row_pruned[row_score[i].second] = true;
        DeepCompression::IF_row_pruned[this->layer_index][row_score[i].second] = true;
        ++ this->num_pruned_row;
        ++ DeepCompression::num_pruned_row[this->layer_index];
    }
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
void ConvolutionLayer<Dtype>::ComputeBlobMask(float ratio) {
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const Dtype *weight = this->blobs_[0]->cpu_data();
    const string layer_name = this->layer_param_.name();

    for (int j = 0; j < num_col; ++j) {
        Dtype sum = 0;
        for (int i = 0; i < num_row; ++i) { sum += fabs(weight[i * num_col + j]); }
        if (sum == 0) { 
            ++ this->num_pruned_col;
            this->IF_col_pruned[j] = true;
            DeepCompression::IF_col_pruned[this->layer_index][j] = true;
            for (int i = 0; i < num_row; ++i) { this->masks_[i * num_col + j] = 0; }
        }
    } 
    for (int i = 0; i < num_row; ++i) { 
        Dtype sum = 0;
        for (int j = 0; j < num_col; ++j) { sum += fabs(weight[i * num_col + j]); }
        if (sum == 0) {
            ++ this->num_pruned_row;
            this->IF_row_pruned[i] = true;
            DeepCompression::IF_row_pruned[this->layer_index][i] = true;
            for (int j = 0; j < num_col; ++j) { this->masks_[i * num_col + j] = 0; }
        }
    }
    DeepCompression::num_pruned_col[this->layer_index] = this->num_pruned_col;
    DeepCompression::num_pruned_row[this->layer_index] = this->num_pruned_row;
    this->pruned_ratio = 1 - (1 - this->num_pruned_col * 1.0 / num_col) * (1 - this->num_pruned_row * 1.0 / num_row);
    if (this->pruned_ratio >= this->prune_ratio) {
        DeepCompression::IF_prune_finished[this->layer_index] = true;
    }
    LOG(INFO) << "    Masks restored, num_pruned_col = " << this->num_pruned_col
              << "  num_pruned_row = " << this->num_pruned_row
              << "  pruned_ratio = " << this->pruned_ratio
              << "  prune_ratio = " << this->prune_ratio;
    /// sleep(1);
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
