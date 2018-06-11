#include <string>
#include <vector>
#include <numeric>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <ctime>
#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"

namespace caffe {

// Return the current learning rate. The currently implemented learning rate
// policies are as follows:
//    - fixed: always return base_lr.
//    - step: return base_lr * gamma ^ (floor(iter / step))
//    - exp: return base_lr * gamma ^ iter
//    - inv: return base_lr * (1 + gamma * iter) ^ (- power)
//    - multistep: similar to step but it allows non uniform steps defined by
//      stepvalue
//    - poly: the effective learning rate follows a polynomial decay, to be
//      zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)
//    - sigmoid: the effective learning rate follows a sigmod decay
//      return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
//
// where base_lr, max_iter, gamma, step, stepvalue and power are defined
// in the solver parameter protocol buffer, and iter is the current iteration.
template <typename Dtype>
Dtype SGDSolver<Dtype>::GetLearningRate() {
  Dtype rate;
  const string& lr_policy = this->param_.lr_policy();
  if (lr_policy == "fixed") {
    rate = this->param_.base_lr();
  } else if (lr_policy == "step") {
    this->current_step_ = this->iter_ / this->param_.stepsize();
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "exp") {
    rate = this->param_.base_lr() * pow(this->param_.gamma(), this->iter_);
  } else if (lr_policy == "inv") {
    rate = this->param_.base_lr() *
        pow(Dtype(1) + this->param_.gamma() * this->iter_,
            - this->param_.power());
  } else if (lr_policy == "multistep") {
    if (this->current_step_ < this->param_.stepvalue_size() &&
          this->iter_ >= this->param_.stepvalue(this->current_step_)) {
      this->current_step_++;
      LOG(INFO) << "MultiStep Status: Iteration " <<
      this->iter_ << ", step = " << this->current_step_;
    }
    rate = this->param_.base_lr() *
        pow(this->param_.gamma(), this->current_step_);
  } else if (lr_policy == "poly") {
    rate = this->param_.base_lr() * pow(Dtype(1.) -
        (Dtype(this->iter_) / Dtype(this->param_.max_iter())),
        this->param_.power());
  } else if (lr_policy == "sigmoid") {
    rate = this->param_.base_lr() * (Dtype(1.) /
        (Dtype(1.) + exp(-this->param_.gamma() * (Dtype(this->iter_) -
          Dtype(this->param_.stepsize())))));
  } else {
    LOG(FATAL) << "Unknown learning rate policy: " << lr_policy;
  }
  return rate;
}

template <typename Dtype>
void SGDSolver<Dtype>::PreSolve() {
  // Initialize the history
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  history_.clear();
  update_.clear();
  temp_.clear();
  /// @mingsuntse, for pruning
  tmp_.clear();
  
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));  
    
    /// @mingsuntse, for pruning
    tmp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  // cout << "clip_gradients: " << clip_gradients << endl; // WANGHUAN
  if (clip_gradients < 0) { return; }
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  Dtype sumsq_diff = 0;
  for (int i = 0; i < net_params.size(); ++i) {
    sumsq_diff += net_params[i]->sumsq_diff();
  }
  const Dtype l2norm_diff = std::sqrt(sumsq_diff);
  if (l2norm_diff > clip_gradients) {
    Dtype scale_factor = clip_gradients / l2norm_diff;
    LOG(INFO) << "Gradient clipping: scaling down gradients (L2 norm "
        << l2norm_diff << " > " << clip_gradients << ") "
        << "by scale factor " << scale_factor;
    for (int i = 0; i < net_params.size(); ++i) {
      net_params[i]->scale_diff(scale_factor);
    }
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ApplyUpdate() {
  #ifdef ShowTimingLog
  cout << "ApplyUpdate begins timing" << endl;
  clock_t t1 = clock();
  #endif
    
  CHECK(Caffe::root_solver()); /// 更新梯度是由主solver来做的
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  ClipGradients();
  #ifdef ShowTimingLog
  cout << "  after ClipGradients: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
  #endif
  
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    Normalize(param_id);
    Regularize(param_id);
    ClearHistory(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  #ifdef ShowTimingLog
  cout << "  after ComputeUpdateValue etc.: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
  #endif
  
  this->net_->Update();
  #ifdef ShowTimingLog
  cout << "Time in ApplyUpdate: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
  #endif
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) { return; }
  // Scale gradient to counterbalance accumulation.
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const Dtype accum_normalization = Dtype(1.) / this->param_.iter_size();
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    caffe_gpu_scal(net_params[param_id]->count(), accum_normalization,
        net_params[param_id]->mutable_gpu_diff());
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {    
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();
  // ------------------------------------------------
  /// Decrease-Weight-Decay Mode, @mingsuntse
  Dtype current_wd = weight_decay; // default
  if (this->param_.dwd_mode() != "None") {
      CHECK_GE(this->param_.wd_end(), 0) << "Error: wd_end must be in [0, 1]";
      // CHECK_LE(this->param_.wd_end(), 1) << "Error: wd_end must be in [0, 1]"; // weight decay can go up, when wd_end > 1
      
      const int begin = this->param_.dwd_begin_iter();
      if (this->iter_ >= begin) {
          if (this->param_.dwd_mode() == "linearly") {
            const int end   = this->param_.dwd_end_iter();
            CHECK_GT(end, begin) << "Error: dwd_end_iter must be larger than dwd_begin_iter.";
            current_wd = weight_decay * (1 - (1 - this->param_.wd_end()) / (end - begin) * (std::min(this->iter_, end) - begin));
          
          } else if (this->param_.dwd_mode() == "step_linearly") {
            const int end   = this->param_.dwd_end_iter();
            CHECK_GT(end, begin) << "Error: dwd_end_iter must be larger than dwd_begin_iter.";
            const int tmp_iter = (std::min(this->iter_, end) - begin) / this->param_.dwd_step() * this->param_.dwd_step();
            current_wd = weight_decay * (1 - (1 - this->param_.wd_end()) / (end - begin) * tmp_iter);

          }
      }
  }
  // ------------------------------------------------
  Dtype local_decay = current_wd * net_params_weight_decay[param_id];
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            net_params[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());

      } else if (regularization_type == "L1") {
        caffe_cpu_sign(net_params[param_id]->count(),
            net_params[param_id]->cpu_data(),
            temp_[param_id]->mutable_cpu_data());

        // compute sign, saved in temp_[param_id]->mutable_cpu_data()
        caffe_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->cpu_data(),
            net_params[param_id]->mutable_cpu_diff());

      } else {
        LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    if (local_decay) {
      if (regularization_type == "L2") {
        // add weight decay
        caffe_gpu_axpy(net_params[param_id]->count(),
                       local_decay,
                       net_params[param_id]->gpu_data(),
                       net_params[param_id]->mutable_gpu_diff());

      } else if (regularization_type == "L1") {
        caffe_gpu_sign(net_params[param_id]->count(),
            net_params[param_id]->gpu_data(),
            temp_[param_id]->mutable_gpu_data());

        caffe_gpu_axpy(net_params[param_id]->count(),
            local_decay,
            temp_[param_id]->gpu_data(),
            net_params[param_id]->mutable_gpu_diff());

      } else if (regularization_type == "SelectiveReg" || regularization_type == "Reg_Col") {
        // add weight decay, weight decay still used
        caffe_gpu_axpy(net_params[param_id]->count(),
                       local_decay,
                       net_params[param_id]->gpu_data(),
                       net_params[param_id]->mutable_gpu_diff());    
        // If return
        const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[param_id].first];
        const int L = GetLayerIndex(param_id);
        if (L == -1) { return; }
        
        #ifdef ShowTimingLog
        cout << layer_name << " Reg_Col start timing" << endl;
        clock_t t1 = clock();
        #endif
        
        /* print for check
        cout << param_id << "  layer_name: " << layer_name << endl;
        cout << "  num_param_layer_indices: " << this->net_->param_layer_indices().size() << endl; 
        cout << "  num_layer: "               << this->net_->layer_names().size() << endl;
        cout << "  num_blob: "                << this->net_->blob_names().size()  << endl;
        cout << "  num_learnable_param: "     << this->net_->learnable_params().size() << endl;

        // My layer_index only contains conv and fc layers, while caffe's layer_index contains literally all layers.
        map<string, int> layer_names_index = this->net_->layer_names_index();
        cout << "my layer_index: " << L
             << "  caffe's layer_index: " << layer_names_index[layer_name] << endl;
        */
        Dtype* muhistory_score  = this->net_->layer_by_name(layer_name)->history_score()[0]->mutable_cpu_data();
        Dtype* muhistory_punish = this->net_->layer_by_name(layer_name)->history_punish()[0]->mutable_cpu_data();
        Dtype* mumasks          = this->net_->layer_by_name(layer_name)->masks()[0]->mutable_cpu_data();
        const Dtype* weight = net_params[param_id]->cpu_data();
        Dtype* muweight = net_params[param_id]->mutable_cpu_data();
        const int count = net_params[param_id]->count();
        const int num_row = net_params[param_id]->shape()[0];
        const int num_col = count / num_row;
        const int num_pruned_col = APP<Dtype>::num_pruned_col[L];
        const int num_col_to_prune_ = ceil(num_col * APP<Dtype>::prune_ratio_step * 2); // Going to prune APP<Dtype>::prune_ratio_step, but set target as APP<Dtype>::prune_ratio_step * 2.
        const int num_col_ = num_col - num_pruned_col;
        if (num_col_to_prune_ <= 0) {
            LOG(FATAL) << "num_col_to_prune_ <= 0";
            exit(1);
        }
        const Dtype AA = APP<Dtype>::AA;
        if (APP<Dtype>::step_ % APP<Dtype>::prune_interval == 0) {
            if (APP<Dtype>::prune_coremthd == "Reg-rank" || APP<Dtype>::prune_coremthd == "Reg") {
                // Sort 01: sort by L1-norm
                typedef std::pair<Dtype, int> mypair;
                vector<mypair> col_score(num_col);
                for (int j = 0; j < num_col; ++j) {
                    col_score[j].second = j;
                    if (APP<Dtype>::IF_col_pruned[L][j][0]) { // TODO(mingsuntse): fix this [0]
                        col_score[j].first = muhistory_score[j]; // make the pruned sink down
                        continue;
                    }
                    col_score[j].first  = 0;
                    for (int i = 0; i < num_row; ++i) {
                        col_score[j].first += fabs(weight[i * num_col + j]);
                    }
                }
                sort(col_score.begin(), col_score.end());
                #ifdef ShowTimingLog
                cout  << "  after 1st sort: " << (double)(clock() - t1)/CLOCKS_PER_SEC << "s" << endl;
                #endif
                
                // Make new criteria by rank: history_rank
                const int n = this->iter_ + 1; // No.n iter (n starts from 1)
                for (int rk = 0; rk < num_col; ++rk) {
                    const int col_of_rank_rk = col_score[rk].second;
                    if (APP<Dtype>::IF_col_pruned[L][col_of_rank_rk][0]) { continue; }
                    muhistory_score[col_of_rank_rk] = ((n-1) * muhistory_score[col_of_rank_rk] + rk) / n;
                }
                
                // Sort 02: sort by history_rank
                vector<mypair> col_hrank(num_col); // the history_rank of each column, history_rank is like the new score
                // cout << "ave-magnitude_col " << this->iter_ << " " << layer_name << ":";
                for (int j = 0; j < num_col; ++j) {
                    Dtype sum = 0; // for print ave magnitude
                    for (int i = 0; i < num_row; ++i) {
                        sum += fabs(weight[i * num_col + j]);
                    }
                    // cout << " " << sum/num_row;
                    col_hrank[j].first  = muhistory_score[j];
                    col_hrank[j].second = j;
                }
                sort(col_hrank.begin(), col_hrank.end());
                //cout << endl;
                #ifdef ShowTimingLog
                cout  << "  after 2nd sort: " << (double)(clock() - t1)/CLOCKS_PER_SEC << "s" << endl;
                #endif
                
                /*
                // Print: Check rank, j is column number
                if (this->iter_ % 20 == 0) {
                    char iter[10]; sprintf(iter, "%6d", this->iter_ + 1); // max_iter should be in [0, 999999]
                    cout << iter << "-" << layer_name << "hrank:";
                    for (int j = 0; j < num_col; ++j) {
                        cout << "  ";
                        char s[50];
                        if (APP<Dtype>::IF_col_pruned[L][j][0]) { 
                            sprintf(s, "%7.0f", muhistory_score[j]);
                        } else {
                            sprintf(s, "%7.2f", muhistory_score[j]);
                        }
                        cout << s;
                    }
                    cout << endl;
                    
                    cout << iter << "-" << layer_name << "rank(by_hrank):";
                    for (int rk = 0; rk < num_col; ++rk) {
                        cout << "  ";
                        char s[50];
                        const int prune_mark = APP<Dtype>::IF_col_pruned[L][col_hrank[rk].second][0] ? 0 : 1;
                        sprintf(s, "%4d-%d", col_hrank[rk].second, prune_mark);
                        cout << s;
                    }
                    cout << endl;
                }
                */

                // scheme 1, the exponential center-symmetrical function
                const Dtype kk = APP<Dtype>::kk; // u in the paper
                const Dtype alpha = log(2/kk) / (num_col_to_prune_);
                const Dtype N1 = -log(kk)/alpha; // symmetry point
                // scheme 2, the dis-continual function
                const Dtype kk2 = APP<Dtype>::kk2;
                const Dtype alpha1 = (num_col_to_prune_ == 1)          ? 0 : log(1/kk2) / (num_col_to_prune_ - 1);
                const Dtype alpha2 = (num_col_to_prune_ == num_col_-1) ? 0 : log(1/kk2) / (num_col_ - 1 - num_col_to_prune_);
                for (int j = 0; j < num_col_; ++j) { // j: rank 
                    const int col_of_rank_j = col_hrank[j + num_pruned_col].second; // Note the real rank is j + num_pruned_col
                    const Dtype Delta = APP<Dtype>::IF_scheme1_when_Reg_rank
                                          ? (j < N1                ? AA * exp(-alpha  * j) : 2*kk*AA - AA * exp(-alpha  * (2 * N1     - j)))
                                          : (j < num_col_to_prune_ ? AA * exp(-alpha1 * j) :         - AA * exp(-alpha2 * (num_col_-1 - j)));
                    const Dtype old_reg = muhistory_punish[col_of_rank_j];
                    const Dtype new_reg = std::max(old_reg + Delta, Dtype(0));
                    for (int i = 0; i < num_row; ++i) {
                        muhistory_punish[i * num_col + col_of_rank_j] = new_reg;
                    }
                    if (new_reg >= APP<Dtype>::target_reg) {
                        for (int g = 0; g < APP<Dtype>::group[L]; ++g) { 
                          APP<Dtype>::IF_col_pruned[L][col_of_rank_j][g] = true;
                        }
                        APP<Dtype>::num_pruned_col[L] += 1;
                        for (int i = 0; i < num_row; ++i) {
                            mumasks[i * num_col + col_of_rank_j] = 0;
                            muweight[i* num_col + col_of_rank_j] = 0;
                        }
                        muhistory_score[col_of_rank_j] = APP<Dtype>::step_ - 1000000 - (muhistory_punish[col_of_rank_j] - APP<Dtype>::target_reg);
                        
                        // Check whether the corresponding row in the last layer could be pruned
                        if (L != 0 && L != APP<Dtype>::conv_layer_cnt) { // Not the fist Conv and first FC layer
                          const int filter_spatial_size = net_params[param_id]->count(2);
                          const int channel = col_of_rank_j / filter_spatial_size;
                          bool IF_consecutive_pruned = true;
                          for (int j = channel * filter_spatial_size; j < (channel+1) * filter_spatial_size; ++j) {
                            
                            if (!APP<Dtype>::IF_col_pruned[L][j][0]) {
                              IF_consecutive_pruned = false;
                              break;
                            }
                          }
                          if (IF_consecutive_pruned) {
                            const int num_chl_per_g = num_col / filter_spatial_size;
                            for (int g = 0; g < APP<Dtype>::group[L]; ++g) {
                              APP<Dtype>::rows_to_prune[L - 1].push_back(channel + g * num_chl_per_g);
                            }
                          }
                        }
                    }
                    if (new_reg < old_reg) {
                        cout << "reduce reg: " << layer_name << "-" << col_of_rank_j 
                             << "   old reg: "  << old_reg
                             << "   new reg: "  << new_reg << endl;
                    }
                }
            } else if (APP<Dtype>::prune_coremthd == "Reg-L1") {
                typedef std::pair<Dtype, int> mypair;
                vector<mypair> col_score(num_col);
                for (int j = 0; j < num_col; ++j) {
                    col_score[j].second = j;
                    if (APP<Dtype>::IF_col_pruned[L][j][0]) {
                        col_score[j].first = INT_MAX;
                        continue;
                    }
                    col_score[j].first  = 0;
                    for (int i = 0; i < num_row; ++i) {
                        col_score[j].first += fabs(weight[i * num_col + j]);
                    }
                }
                sort(col_score.begin(), col_score.end());
                
                const Dtype k_L1 = (num_col_to_prune_ == 0) ? 0 : AA / (col_score[num_col_to_prune_].first - col_score[0].first);
                cout << "step: " << this->iter_ + 1 << "  " << layer_name << " k_L1: " << k_L1 << endl;
                for (int rk = 0; rk < num_col_; ++rk) {
                    const int col_of_rank_rk = col_score[rk].second;
                    const Dtype Delta = AA - k_L1 * (col_score[rk].first - col_score[0].first);
                    const Dtype old_reg = muhistory_punish[col_of_rank_rk];
                    const Dtype new_reg = std::max(old_reg + Delta, Dtype(0));
                    for (int i = 0; i < num_row; ++i) {
                        muhistory_punish[i * num_col + col_of_rank_rk] = new_reg;
                    }
                    if (new_reg < old_reg) {
                        cout << "reduce reg: " << layer_name << "-" << col_of_rank_rk 
                             << "   old reg: "  << old_reg
                             << "   new reg: "  << new_reg << endl;
                    }
                }
                const int num_show = APP<Dtype>::show_num_weight > num_col ? num_col : APP<Dtype>::show_num_weight;
                cout << layer_name << "-score: "; for (int rk = 0; rk < num_show; ++rk) { cout << col_score[rk].first  << " "; } cout << endl;
                cout << layer_name << "  -col: "; for (int rk = 0; rk < num_show; ++rk) { cout << col_score[rk].second << " "; } cout << endl;
                cout << layer_name << "  -reg: "; for (int rk = 0; rk < num_show; ++rk) { cout << muhistory_punish[col_score[rk].second] << " "; } cout << endl;
            }
        }
        #ifdef ShowTimingLog
        cout  << "  after calculate reg term: " << (double)(clock() - t1)/CLOCKS_PER_SEC << "s" << endl;
        #endif
        
        //Apply Reg
        caffe_gpu_mul(count,
                      net_params[param_id]->gpu_data(),
                      muhistory_punish,
                      tmp_[param_id]->mutable_gpu_data());
        #ifdef ShowTimingLog
        cout << "  after gpu mul: " << (double)(clock() - t1)/CLOCKS_PER_SEC << "s" << endl;
        #endif
        caffe_gpu_add(count,
                      tmp_[param_id]->gpu_data(),
                      net_params[param_id]->gpu_diff(),
                      net_params[param_id]->mutable_gpu_diff()); 
        #ifdef ShowTimingLog
        cout << "  after gpu add, end of Reg_Col: " << (double)(clock() - t1)/CLOCKS_PER_SEC << "s" << endl;
        #endif
      } else if (regularization_type == "Reg-Optimal_Col") {
         
        // If return
        const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[param_id].first];
        const int L = GetLayerIndex(param_id);
        if (L == -1) { return; }
        
        #ifdef ShowTimingLog
        cout << layer_name << " Reg-Optimal_Col start timing" << endl;
        clock_t t1 = clock();
        #endif
        
        Dtype* muhistory_punish = this->net_->layer_by_name(layer_name)->history_punish()[0]->mutable_cpu_data();
        Dtype* mumasks = this->net_->layer_by_name(layer_name)->masks()[0]->mutable_cpu_data();
        const Dtype* weight = net_params[param_id]->cpu_data();
        const Dtype* S      = net_params[param_id]->cpu_secdiff();
        const int count     = net_params[param_id]->count();
        const int num_row   = net_params[param_id]->shape()[0];
        const int num_col   = count / num_row;
        const int num_pruned_col    = APP<Dtype>::num_pruned_col[L];
        const int num_col_to_prune_ = ceil(num_col * APP<Dtype>::prune_ratio[L]) - num_pruned_col;
        if (num_col_to_prune_ <= 0) {
            LOG(FATAL) << "num_col_to_prune_ <= 0";
            exit(1);
        }
        const Dtype AA = APP<Dtype>::AA; // The fixed reg multiplier
        
        cout << layer_name << endl;
        if (APP<Dtype>::step_ % APP<Dtype>::prune_interval == 0) {
            // Get alpha
            Dtype D, E, F;
            int ix = 19; // fixed_reg_weight_index
            D = S[ix] + muhistory_punish[ix];
            E = weight[ix] * weight[ix] * D * D;
            F = E * muhistory_punish[ix];
            const Dtype alpha = (-E * AA - F) / ((D + AA) * (D + AA) * (D + AA));
            
            // vector<Dtype> compr(num_col, 0); // magnitude compression ratio
            // typedef std::pair<Dtype, int> mypair;
            // vector<mypair> col_score(num_col);
            for (int i = 0; i < count; ++i) {
                D = S[i] + muhistory_punish[i];
                E = weight[i] * weight[i] * D * D;
                F = E * muhistory_punish[i];
                
                // solve the 3-order equation
                const Dtype p = (F - D * E) / alpha;
                const Dtype q = E / alpha;
                
                // Check 
                Dtype pq0 = q*q/4 + p*p*p/27;
                if (pq0 < 0) {
                    // cout << "Wrong: negative " << pq0 << endl;
                    pq0 = - pq0;
                }
                const Dtype pq1 = -q/2 + sqrt(pq0);
                const Dtype pq2 = -q/2 - sqrt(pq0);
                const int sign1 = pq1 > 0 ? 1 : -1;
                const int sign2 = pq2 > 0 ? 1 : -1;
                
                const Dtype x = sign1 * pow(fabs(pq1), 1.0/3)
                              + sign2 * pow(fabs(pq2), 1.0/3); // if q*q/4 + p*p*p/27 > 0?
                
                if (i < 10) {
                    cout << "alpha: " << alpha
                         << "  S: " << S[i] 
                         << "  D: " << D 
                         << "  E: " << E 
                         << "  F: " << F 
                         << "  p: " << p 
                         << "  q: " << q 
                         << "  pq:" << q*q/4 + p*p*p/27
                         << "  pq1:" << pq1
                         << "  pq2:" << pq2
                         << "  x-D: " << x-D
                         << endl;
                }
                
                muhistory_punish[i] += min(APP<Dtype>::AA, fabs(x - D));
                if (muhistory_punish[i] >= APP<Dtype>::target_reg) {
                    mumasks[i] = 0;
                }
                
                // compr[i % num_col] += fabs((x - D) / x); // |\Delta w / w|
            }
            cout << endl;
            
            /* 
            // Sort col_score
            for (int j = 0; j < num_col; ++j) {
                col_score[j].first  = compr[j];
                col_score[j].second = j;
            }
            sort(col_score.begin(), col_score.end());
            
            
            for (int rk = 0; rk < num_col; ++rk) {
                const int col_of_rank_rk = col_score[rk].second;
                if (rk < num_col_to_prune_) { // rank is not good
                    
                } else { // rank is good
                    
                }
                // Get 
            */
        }

        #ifdef ShowTimingLog
        cout  << "  after calculate reg term: " << (double)(clock() - t1)/CLOCKS_PER_SEC << "s" << endl;
        #endif
        
        //Apply Reg
        caffe_gpu_mul(net_params[param_id]->count(),
                      net_params[param_id]->gpu_data(),
                      muhistory_punish,
                      tmp_[param_id]->mutable_gpu_data());
        #ifdef ShowTimingLog
        cout << "  after gpu mul: " << (double)(clock() - t1)/CLOCKS_PER_SEC << "s" << endl;
        #endif
        caffe_gpu_add(net_params[param_id]->count(),
                      tmp_[param_id]->gpu_data(),
                      net_params[param_id]->gpu_diff(),
                      net_params[param_id]->mutable_gpu_diff()); 
        #ifdef ShowTimingLog
        cout << "  after gpu add, end of Reg-Optimal_Col: " << (double)(clock() - t1)/CLOCKS_PER_SEC << "s" << endl;
        #endif
      
      // ******************************************************************************************
      // Got idea from cvpr rebuttal, improve SelectiveReg: 1) use L1-norm rather than rank, 2) row prune
      } else if (regularization_type == "Reg_Row") {
        // add weight decay, weight decay still used
        caffe_gpu_axpy(net_params[param_id]->count(),
                       local_decay,
                       net_params[param_id]->gpu_data(),
                       net_params[param_id]->mutable_gpu_diff());
        // If return
        const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[param_id].first];
        const int L = GetLayerIndex(param_id);
        if (L == -1) { return; }
        
        Dtype* muhistory_score  = this->net_->layer_by_name(layer_name)->history_score()[0]->mutable_cpu_data();
        Dtype* muhistory_punish = this->net_->layer_by_name(layer_name)->history_punish()[0]->mutable_cpu_data();
        Dtype* gpu_muhistory_punish = this->net_->layer_by_name(layer_name)->history_punish()[0]->mutable_gpu_data();
        Dtype* mumasks          = this->net_->layer_by_name(layer_name)->masks()[0]->mutable_cpu_data();
        const Dtype* weight = net_params[param_id]->cpu_data();
        const int count     = net_params[param_id]->count();
        const int num_row   = net_params[param_id]->shape()[0];
        const int num_col   = count / num_row;
        const Dtype AA = APP<Dtype>::AA;
        const int num_pruned_row    = APP<Dtype>::num_pruned_row[L];
        const int num_row_to_prune_ = ceil(num_row * APP<Dtype>::prune_ratio[L]) - num_pruned_row;
        const int num_row_          = num_row - num_pruned_row;
        
        if (APP<Dtype>::prune_coremthd == "Reg-rank") {
            // Sort 01: sort by L1-norm
            typedef std::pair<Dtype, int> mypair;
            vector<mypair> row_score(num_row);
            for (int i = 0; i < num_row; ++i) {
                row_score[i].second = i;
                if (APP<Dtype>::IF_row_pruned[L][i]) {
                    row_score[i].first = muhistory_score[i * num_col]; // make the pruned row sink down
                    continue;
                }
                row_score[i].first  = 0;
                for (int j = 0; j < num_col; ++j) {
                    row_score[i].first += fabs(weight[i * num_col + j]);
                }
            }
            sort(row_score.begin(), row_score.end()); // in ascending order
            
            // Make new criteria by rank: history_rank
            const int n = this->iter_ + 1; // No.n iter (n starts from 1)
            for (int rk = 0; rk < num_row; ++rk) {
                const int row_of_rank_rk = row_score[rk].second;
                if (APP<Dtype>::IF_row_pruned[L][row_of_rank_rk]) { continue; }
                muhistory_score[row_of_rank_rk * num_col] = ((n-1) * muhistory_score[row_of_rank_rk * num_col] + rk) / n;
            }
            
            // Sort 02: sort by history_rank
            vector<mypair> row_hrank(num_row);
            // cout << "ave-magnitude_row " << this->iter_ << " " << layer_name << ":";
            for (int i = 0; i < num_row; ++i) {
                row_hrank[i].first  = muhistory_score[i * num_col];
                row_hrank[i].second = i;
            }
            sort(row_hrank.begin(), row_hrank.end());
            
            // Punishment Function 
            assert (num_row_to_prune_ > 0);
            if (APP<Dtype>::IF_scheme1_when_Reg_rank) {
                // scheme 1
                const Dtype kk = APP<Dtype>::kk;
                const Dtype alpha = log(2/kk) / (num_row_to_prune_ + 1);
                const Dtype N1 = -log(kk)/alpha;
                for (int rk = 0; rk < num_row_; ++rk) {
                    const int row_of_rank_rk = row_hrank[rk + num_pruned_row].second; // Note the real rank is j + num_pruned_col
                    const Dtype Delta = rk < N1 ? AA * exp(-alpha * rk) : -AA * exp(-alpha * (2*N1-rk)) + 2*kk*AA;
                    const Dtype old_reg = muhistory_punish[row_of_rank_rk * num_col];
                    const Dtype new_reg = std::max(old_reg + Delta, Dtype(0));
                    caffe_gpu_set(num_col, new_reg, gpu_muhistory_punish + row_of_rank_rk * num_col);
                    if (new_reg < old_reg) {
                        cout << "reduce reg: " << layer_name << "-" << row_of_rank_rk 
                             << "  old reg: "  << old_reg
                             << "  new reg: "  << new_reg << endl;
                    }
                }
            } else {
                // scheme 2, the dis-continual function
                const Dtype kk2 = APP<Dtype>::kk2;
                const Dtype alpha1 = (num_row_to_prune_ == 1)          ? 0 : log(1/kk2) / (num_row_to_prune_ - 1);
                const Dtype alpha2 = (num_row_to_prune_ == num_row_-1) ? 0 : log(1/kk2) / (num_row_-1 - num_row_to_prune_);
                for (int rk = 0; rk < num_row_; ++rk) {
                    const int row_of_rank_rk = row_hrank[rk + num_pruned_row].second; // Note the real rank is j + num_pruned_col
                    const Dtype Delta = rk < num_row_to_prune_ ? AA * exp(-alpha1 * rk) : -AA * exp(-alpha2 * (num_row_-1 - rk));
                    const Dtype old_reg = muhistory_punish[row_of_rank_rk * num_col];
                    const Dtype new_reg = std::max(old_reg + Delta, Dtype(0));
                    caffe_gpu_set(num_col, new_reg, gpu_muhistory_punish + row_of_rank_rk * num_col);
                }
            }
        
        // use L1-norm rather than rank
        } else if (APP<Dtype>::prune_coremthd == "Reg-L1") {
            // sort by L1-norm
            typedef std::pair<Dtype, int> mypair;
            vector<mypair> row_score(num_row);
            for (int i = 0; i < num_row; ++i) {
                row_score[i].second = i;
                if (APP<Dtype>::IF_row_pruned[L][i]) {
                    row_score[i].first = INT_MAX; // make the pruned row float up
                    continue;
                }
                row_score[i].first  = 0;
                for (int j = 0; j < num_col; ++j) {
                    row_score[i].first += fabs(weight[i * num_col + j]);
                }
            }
            sort(row_score.begin(), row_score.end()); // in ascending order
            
            // Punishment Function
            assert (num_row_to_prune_ > 0 && num_row_to_prune_ < num_row);
            cout << "num_row_to_prune_: " <<num_row_to_prune_ << endl;
            const Dtype k_L1 = num_row_to_prune_ == 0 ? 0 : AA / (row_score[num_row_to_prune_].first - row_score[0].first);
            cout << "k_L1: " << k_L1 << endl;
            for (int rk = 0; rk < num_row_; ++rk) {
                const int row_of_rank_rk = row_score[rk].second;
                const Dtype Delta = AA - k_L1 * (row_score[rk].first - row_score[0].first);
                const Dtype old_reg = muhistory_punish[row_of_rank_rk * num_col];
                const Dtype new_reg = std::max(old_reg + Delta, Dtype(0));
                caffe_gpu_set(num_col, new_reg, gpu_muhistory_punish + row_of_rank_rk * num_col);
                if (new_reg < old_reg) {
                    cout << "reduce reg: " << layer_name << "-" << row_of_rank_rk 
                         << "  old reg: "  << old_reg
                         << "  new reg: "  << new_reg << endl;
                }
            }
            const int num_show = 10000 > num_row ? num_row : APP<Dtype>::show_num_weight;
            cout << "score: ";   for (int rk = 0; rk < num_show; ++rk) { cout << row_score[rk].first  << " "; }
            cout << "\n  row: "; for (int rk = 0; rk < num_show; ++rk) { cout << row_score[rk].second << " "; }
            cout << "\n  reg: "; for (int rk = 0; rk < num_show; ++rk) { cout << muhistory_punish[row_score[rk].second * num_col] << " "; }
            cout << endl;
        }
        //Apply Reg
        caffe_gpu_mul(net_params[param_id]->count(),
                      net_params[param_id]->gpu_data(),
                      muhistory_punish,
                      tmp_[param_id]->mutable_gpu_data());
        caffe_gpu_add(net_params[param_id]->count(),
                      tmp_[param_id]->gpu_data(),
                      net_params[param_id]->gpu_diff(),
                      net_params[param_id]->mutable_gpu_diff()); 
                      
      } else if (regularization_type == "Reg_Weight") {
        // add weight decay, weight decay still used
        caffe_gpu_axpy(net_params[param_id]->count(),
                       local_decay,
                       net_params[param_id]->gpu_data(),
                       net_params[param_id]->mutable_gpu_diff());    
        // If return
        const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[param_id].first];
        const int L = GetLayerIndex(param_id);
        if (L == -1) { return; }
        
        Dtype* muhistory_score  = this->net_->layer_by_name(layer_name)->history_score()[0]->mutable_cpu_data();
        Dtype* muhistory_punish = this->net_->layer_by_name(layer_name)->history_punish()[0]->mutable_cpu_data();
        Dtype* mumasks          = this->net_->layer_by_name(layer_name)->masks()[0]->mutable_cpu_data();
        const Dtype* weight = net_params[param_id]->cpu_data();
        Dtype* muweight = net_params[param_id]->mutable_cpu_data();
        const int count = net_params[param_id]->count();
        
        // estimate threhold score
        Dtype score_min = 999;
        Dtype score_max = -1;
        for (int i = 0; i < count; ++i) {
            if (i < 20) {
                cout << muweight[i] << " ";
            }
            if (fabs(muweight[i]) > score_max) {
                score_max = fabs(muweight[i]);
            }
            if (fabs(muweight[i]) < score_min) {
                score_min = fabs(muweight[i]);
            }
        }
        cout << endl;
        // Dtype score_max=0.246919, score_min=1.91997e-05;
        const Dtype u = (score_max + score_min) / 2; // mean
        const Dtype sigma = (score_max - score_min) / 8; //stddev assumption: all weights are included in 4-sigma scope
        const Dtype prune_ratio = (APP<Dtype>::prune_ratio[L] < 0.5) ? 1 - APP<Dtype>::prune_ratio[L] : APP<Dtype>::prune_ratio[L]; // the lookup table only contains half of the normal distribution
        const Dtype normalized_prune_ratio = round(prune_ratio / 0.05) * 0.05; // e.g. 0.63 -> 0.65; 0.05 is the prune ratio step 
        const int index = int((normalized_prune_ratio - 0.5) / 0.05);
        const Dtype normal_lookup_table[10] = {0, 0.125, 0.255, 0.385, 0.525, 0.675, 0.845, 1.035, 1.285, 1.645};
        const Dtype score_thr = APP<Dtype>::prune_ratio[L] > 0.5
                                    ? u + normal_lookup_table[index] * sigma
                                    : u - normal_lookup_table[index] * sigma;
        assert(score_thr < score_max && score_thr > score_min);
        cout << layer_name << " u=" << u << " sigma=" << sigma 
                           << " | score_thr=" << score_thr << " score_max=" << score_max << " score_min=" << score_min << endl;

        // assign Delta
        const Dtype AA = APP<Dtype>::AA;
        const Dtype k1 = AA / (score_thr - score_min);
        const Dtype k2 = AA / (score_max - score_thr);
        for (int i = 0; i < count; ++i) {
            const Dtype Delta = fabs(weight[i]) < score_thr 
                                    ? AA - k1 * (fabs(weight[i]) - score_min)
                                    : k2 * (score_thr - fabs(weight[i]));
            const Dtype old_reg = muhistory_punish[i];
            const Dtype new_reg = max(old_reg + Delta, Dtype(0));
            muhistory_punish[i] = new_reg;
            if (new_reg < old_reg) {
                cout << "recover reg: " << layer_name << "-" << i 
                     << "  old reg: " << old_reg
                     << "  new reg: " << new_reg << endl;
            }
        }
        caffe_gpu_mul(net_params[param_id]->count(),
                      net_params[param_id]->gpu_data(),
                      muhistory_punish,
                      tmp_[param_id]->mutable_gpu_data());
        caffe_gpu_add(net_params[param_id]->count(),
                      tmp_[param_id]->gpu_data(),
                      net_params[param_id]->gpu_diff(),
                      net_params[param_id]->mutable_gpu_diff());
      } else {
          LOG(FATAL) << "Unknown regularization type: " << regularization_type;
      }
    }
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

#ifndef CPU_ONLY
template <typename Dtype>
void sgd_update_gpu(int N, Dtype* g, Dtype* h, Dtype momentum,
    Dtype local_rate);
#endif

template <typename Dtype>
void SGDSolver<Dtype>::ComputeUpdateValue(int param_id, Dtype rate) {
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_lr = this->net_->params_lr();
  Dtype momentum = this->param_.momentum();
  Dtype local_rate = rate * net_params_lr[param_id];
  // Compute the update to history, then copy it to the parameter diff.
  switch (Caffe::mode()) {
  case Caffe::CPU: {
    caffe_cpu_axpby(net_params[param_id]->count(), local_rate,
              net_params[param_id]->cpu_diff(), momentum,
              history_[param_id]->mutable_cpu_data()); /// history = momentum * history + lrate * diff
    caffe_copy(net_params[param_id]->count(),
        history_[param_id]->cpu_data(),
        net_params[param_id]->mutable_cpu_diff());
    break;
  }
  case Caffe::GPU: {
#ifndef CPU_ONLY
    sgd_update_gpu(net_params[param_id]->count(),
        net_params[param_id]->mutable_gpu_diff(),
        history_[param_id]->mutable_gpu_data(),
        momentum, local_rate);
#else
    NO_GPU;
#endif
    break;
  }
  default:
    LOG(FATAL) << "Unknown caffe mode: " << Caffe::mode();
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::ClearHistory(const int& param_id) {
    const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[param_id].first];
    if (APP<Dtype>::layer_index.count(layer_name) == 0 || history_[param_id]->shape().size() == 1) { 
        return; 
    } // bias not pruned for now
    const int L = APP<Dtype>::layer_index[layer_name];
    if (APP<Dtype>::pruned_ratio[L] == 0) {
        return;
    }
    caffe_gpu_mul(history_[param_id]->count(),
                  this->net_->layer_by_name(layer_name)->masks()[0]->gpu_data(),
                  history_[param_id]->gpu_data(),
                  history_[param_id]->mutable_gpu_data());
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename) {
  switch (this->param_.snapshot_format()) {
    case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
      SnapshotSolverStateToBinaryProto(model_filename);
      break;
    case caffe::SolverParameter_SnapshotFormat_HDF5:
      SnapshotSolverStateToHDF5(model_filename);
      break;
    default:
      LOG(FATAL) << "Unsupported snapshot format.";
  }
}

template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToBinaryProto(
    const string& model_filename) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  state.clear_history();
  state.clear_history_score(); /// @mingsuntse, for pruning
  state.clear_history_punish();
  
  string previous_layer_name = "";
  int local_blob_index = 0;
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    BlobProto* history_score_blob = state.add_history_score(); /// @mingsuntse, for pruning
    BlobProto* history_punish_blob = state.add_history_punish();
    history_[i]->ToProto(history_blob);
    const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[i].first];
    if (APP<Dtype>::layer_index.count(layer_name) && 
            (APP<Dtype>::prune_coremthd.substr(0, 3) == "Reg" or APP<Dtype>::prune_coremthd.substr(0, 2) == "PP")) { // i: param_id
        local_blob_index = layer_name == previous_layer_name ? local_blob_index + 1 : 0;
        this->net_->layer_by_name(layer_name)->history_score()[local_blob_index]->ToProto(history_score_blob);
        this->net_->layer_by_name(layer_name)->history_punish()[local_blob_index]->ToProto(history_punish_blob);
        previous_layer_name = layer_name;
    }
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

// TODO(mingsuntse): Add history_score etc. in .h5
template <typename Dtype>
void SGDSolver<Dtype>::SnapshotSolverStateToHDF5(
    const string& model_filename) {
  string snapshot_filename =
      Solver<Dtype>::SnapshotFilename(".solverstate.h5");
  LOG(INFO) << "Snapshotting solver state to HDF5 file " << snapshot_filename;
  hid_t file_hid = H5Fcreate(snapshot_filename.c_str(), H5F_ACC_TRUNC,
      H5P_DEFAULT, H5P_DEFAULT);
  CHECK_GE(file_hid, 0)
      << "Couldn't open " << snapshot_filename << " to save solver state.";
  hdf5_save_int(file_hid, "iter", this->iter_);
  hdf5_save_string(file_hid, "learned_net", model_filename);
  hdf5_save_int(file_hid, "current_step", this->current_step_);
  hid_t history_hid = H5Gcreate2(file_hid, "history", H5P_DEFAULT, H5P_DEFAULT,
      H5P_DEFAULT);
  CHECK_GE(history_hid, 0)
      << "Error saving solver state to " << snapshot_filename << ".";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_save_nd_dataset<Dtype>(history_hid, oss.str(), *history_[i]);
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromBinaryProto(
    const string& state_file) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  APP<Dtype>::step_ = this->iter_ + 1;
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  if (APP<Dtype>::prune_method != "None") {
      CHECK_EQ(state.history_score_size(), history_.size())
          << "Incorrect length of history score blobs.";
      LOG(INFO) << "SGDSolver: restoring history score";
      CHECK_EQ(state.history_punish_size(), history_.size())
          << "Incorrect length of history punish blobs.";
      LOG(INFO) << "SGDSolver: restoring history punish";
  }
  string previous_layer_name = "";
  int local_blob_index = 0;
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
    const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[i].first];
    if (APP<Dtype>::layer_index.count(layer_name) && 
            (APP<Dtype>::prune_coremthd.substr(0, 3) == "Reg" or APP<Dtype>::prune_coremthd.substr(0, 2) == "PP")) {
        local_blob_index = layer_name == previous_layer_name ? local_blob_index + 1 : 0;
        this->net_->layer_by_name(layer_name)->history_score()[local_blob_index]->FromProto(state.history_score(i));
        this->net_->layer_by_name(layer_name)->history_punish()[local_blob_index]->FromProto(state.history_punish(i));
        previous_layer_name = layer_name;
    }
  }
}

template <typename Dtype>
const int SGDSolver<Dtype>::GetLayerIndex(const int& param_id) {
    // Four occasions to return, `-1` means return
    // 1. Get layer index and layer name, if not registered, don't reg it.
    const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[param_id].first];
    if (APP<Dtype>::layer_index.count(layer_name) == 0) { return -1; }
    const int L = APP<Dtype>::layer_index[layer_name];

    // 2. Do not reg biases
    const vector<int>& shape = this->net_->learnable_params()[param_id]->shape();
    if (shape.size() == 1) { return -1; }
    
    // 3.
    const bool IF_want_prune     = APP<Dtype>::prune_method != "None" && APP<Dtype>::prune_ratio[L] > 0;
    const bool IF_been_pruned    = APP<Dtype>::pruned_ratio[L] > 0;
    const bool IF_enough_iter    = APP<Dtype>::step_ >= APP<Dtype>::prune_begin_iter + 1;
    const bool IF_not_recovering = APP<Dtype>::IF_acc_recovered;
    const bool IF_prune = IF_want_prune && (IF_been_pruned || IF_enough_iter) && IF_not_recovering;
    if (!(IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX)) { return -1; }

    return L;
}

template <typename Dtype>
void SGDSolver<Dtype>::RestoreSolverStateFromHDF5(const string& state_file) {
  hid_t file_hid = H5Fopen(state_file.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  CHECK_GE(file_hid, 0) << "Couldn't open solver state file " << state_file;
  this->iter_ = hdf5_load_int(file_hid, "iter");
  if (H5LTfind_dataset(file_hid, "learned_net")) {
    string learned_net = hdf5_load_string(file_hid, "learned_net");
    this->net_->CopyTrainedLayersFrom(learned_net);
  }
  this->current_step_ = hdf5_load_int(file_hid, "current_step");
  hid_t history_hid = H5Gopen2(file_hid, "history", H5P_DEFAULT);
  CHECK_GE(history_hid, 0) << "Error reading history from " << state_file;
  int state_history_size = hdf5_get_num_links(history_hid);
  CHECK_EQ(state_history_size, history_.size())
      << "Incorrect length of history blobs.";
  for (int i = 0; i < history_.size(); ++i) {
    ostringstream oss;
    oss << i;
    hdf5_load_nd_dataset<Dtype>(history_hid, oss.str().c_str(), 0,
                                kMaxBlobAxes, history_[i].get());
  }
  H5Gclose(history_hid);
  H5Fclose(file_hid);
}

INSTANTIATE_CLASS(SGDSolver);
REGISTER_SOLVER_CLASS(SGD);

}  // namespace caffe
