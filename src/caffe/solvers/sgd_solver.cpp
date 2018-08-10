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
  const int num_learnable_layer = APP<Dtype>::layer_index.size();
  vector<int> shape2(1, num_learnable_layer);
  current_prune_ratio_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape2)));
  last_feasible_prune_ratio_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape2)));
  last_infeasible_prune_ratio_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape2)));
}

template <typename Dtype>
void SGDSolver<Dtype>::ClipGradients() {
  const Dtype clip_gradients = this->param_.clip_gradients();
  if (clip_gradients < 0) {
    return;
  }
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
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();
  if (APP<Dtype>::prune_method != "None") {
    if (APP<Dtype>::learning_rate == 0) {
      APP<Dtype>::learning_rate = rate;
    } else {
      rate = APP<Dtype>::learning_rate;
    }
  }
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  ClipGradients();
  for (int param_id = 0; param_id < this->net_->learnable_params().size(); ++param_id) {
    Normalize(param_id);
    Regularize(param_id);
    ClearHistory(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  this->net_->Update();
}

template <typename Dtype>
void SGDSolver<Dtype>::Normalize(int param_id) {
  if (this->param_.iter_size() == 1) {
    return;
  }
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
  Dtype local_decay = weight_decay * net_params_weight_decay[param_id];
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

      } else if (regularization_type == "SelectiveReg") {
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
        Dtype* muweight   = net_params[param_id]->mutable_cpu_data();
        const int count   = net_params[param_id]->count();
        const int num_row = net_params[param_id]->shape()[0];
        const int num_col = count / num_row;
        const int num_pruned_col = APP<Dtype>::num_pruned_col[L];
        const int real_num_col_to_prune_ = ceil(num_col * APP<Dtype>::current_prune_ratio[L]) - num_pruned_col;
        int num_col_to_prune_ = real_num_col_to_prune_;
        const int num_col_ = num_col - num_pruned_col;
        
        // This should not happen, but check here in case.
        // If it happens unluckily, it usually does not affect the result.
        if (num_col_to_prune_ <= 0) {
          cout << "BUG: num_col_to_prune_ = " << num_col_to_prune_ << endl;
          return;
        }
        
        const Dtype AA = APP<Dtype>::AA;
        if (APP<Dtype>::step_ % APP<Dtype>::prune_interval == 0) {
          if (APP<Dtype>::prune_coremthd == "Reg-rank" || APP<Dtype>::prune_coremthd == "Reg") {
            // Sort 01: sort by L1-norm
            typedef std::pair<Dtype, int> mypair;
            vector<mypair> col_score(num_col);
            for (int j = 0; j < num_col; ++j) {
              col_score[j].second = j;
              if (APP<Dtype>::IF_col_pruned[L][j][0]) {
                col_score[j].first = muhistory_score[j]; // make the pruned sink down
                continue;
              }
              col_score[j].first  = 0;
              for (int i = 0; i < num_row; ++i) {
                col_score[j].first += fabs(muweight[i * num_col + j]);
              }
            }
            sort(col_score.begin(), col_score.end());

            // Make new criteria, i.e. history_rank, by rank
            const int n = this->iter_ + 1; // No.n iter (n starts from 1)
            for (int rk = 0; rk < num_col; ++rk) {
              const int col_of_rank_rk = col_score[rk].second;
              if (APP<Dtype>::IF_col_pruned[L][col_of_rank_rk][0]) { continue; }
              muhistory_score[col_of_rank_rk] = ((n-1) * muhistory_score[col_of_rank_rk] + rk) / n;
            }

            // Sort 02: sort by history_rank
            vector<mypair> col_hrank(num_col); // the history_rank of each column, history_rank is like the new score
            for (int j = 0; j < num_col; ++j) {
              col_hrank[j].first  = muhistory_score[j];
              col_hrank[j].second = j;
            }
            sort(col_hrank.begin(), col_hrank.end());

            // scheme 1, the exponential center-symmetrical function
            const Dtype kk = APP<Dtype>::kk; // u in the paper
            const Dtype alpha = log(2/kk) / (num_col_to_prune_);
            const Dtype N1 = -log(kk)/alpha; // the symmetry point
            
            // scheme 2, the dis-continual function
            const Dtype kk2 = max((APP<Dtype>::pruned_ratio_for_comparison[L] - APP<Dtype>::last_feasible_prune_ratio[L]) 
                  / (APP<Dtype>::current_prune_ratio[L] - APP<Dtype>::last_feasible_prune_ratio[L]) * (Dtype)0.75, (Dtype)0.05) ; // feasible range: (0, 1)
            const Dtype alpha21 = (num_col_to_prune_ == 1)          ? 0 : log(1/kk2) / (num_col_to_prune_-1);
            const Dtype alpha22 = (num_col_to_prune_ == num_col_-1) ? 0 : log(1/kk2) / (num_col_-1 - num_col_to_prune_);
            
            APP<Dtype>::IF_scheme1_when_Reg_rank = true; // scheme 2 is the default. 
            for (int j = 0; j < num_col_; ++j) { // j: rank
              const int col_of_rank_j = col_hrank[j + num_pruned_col].second; // Note the real rank is j + num_pruned_col
              const Dtype Delta = APP<Dtype>::IF_scheme1_when_Reg_rank
                                  ? (j < N1                ? AA * exp(-alpha   * j) : 2*kk*AA - AA * exp(-alpha   * (2 * N1     - j)))
                                  : (j < num_col_to_prune_ ? AA * exp(-alpha21 * j) :         - AA * exp(-alpha22 * (num_col_-1 - j)));

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
                muhistory_score[col_of_rank_j] = APP<Dtype>::step_ - 1000000 - (muhistory_punish[col_of_rank_j] - APP<Dtype>::target_reg); // This is to 
                // make the pruned weight group sorted in left in sort 01 and 02 above, and the earlier pruned the lefter sorted

                // Check whether the corresponding row in the last layer could be pruned
                if (L != 0 && L != APP<Dtype>::conv_layer_cnt) { // Not the fist Conv and first FC layer
                  const int filter_spatial_size = net_params[param_id]->count(2);
                  const int channel = col_of_rank_j / filter_spatial_size;
                  bool IF_consecutively_pruned = true;
                  for (int j = channel * filter_spatial_size; j < (channel+1) * filter_spatial_size; ++j) {
                    if (!APP<Dtype>::IF_col_pruned[L][j][0]) {
                      IF_consecutively_pruned = false;
                      break;
                    }
                  }
                  if (IF_consecutively_pruned) {
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
          }
        }
        
        //Apply Reg
        caffe_gpu_mul(count,
                      net_params[param_id]->gpu_data(),
                      muhistory_punish,
                      tmp_[param_id]->mutable_gpu_data());
        caffe_gpu_add(count,
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
void SGDSolver<Dtype>::SnapshotSolverState(const string& model_filename, const string& prefix) {
  switch (this->param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    SnapshotSolverStateToBinaryProto(model_filename, prefix);
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
  const string& model_filename, const string& prefix) {
  SolverState state;
  state.set_iter(this->iter_);
  state.set_learned_net(model_filename);
  state.set_current_step(this->current_step_);
  
  state.set_prune_state(APP<Dtype>::prune_state);
  state.set_prune_stage(APP<Dtype>::prune_stage);
  state.set_stage_iter_prune_finished(APP<Dtype>::stage_iter_prune_finished);
  state.set_last_feasible_prune_iter(APP<Dtype>::last_feasible_prune_iter);
  
  BlobProto* current_prune_ratio_blob = state.add_current_prune_ratio();
  BlobProto* last_feasible_prune_ratio_blob = state.add_last_feasible_prune_ratio();
  BlobProto* last_infeasible_prune_ratio_blob = state.add_last_infeasible_prune_ratio();
  for (int L = 0; L < APP<Dtype>::layer_index.size(); ++L) {
    current_prune_ratio_[0]->mutable_cpu_data()[L] = APP<Dtype>::current_prune_ratio[L];
    last_feasible_prune_ratio_[0]->mutable_cpu_data()[L] = APP<Dtype>::last_feasible_prune_ratio[L];
    last_infeasible_prune_ratio_[0]->mutable_cpu_data()[L] = APP<Dtype>::last_infeasible_prune_ratio[L];
  }
  current_prune_ratio_[0]->ToProto(current_prune_ratio_blob);
  last_feasible_prune_ratio_[0]->ToProto(last_feasible_prune_ratio_blob);
  last_infeasible_prune_ratio_[0]->ToProto(last_infeasible_prune_ratio_blob);
  
  state.clear_history();
  state.clear_history_score(); /// @mingsuntse, for pruning
  state.clear_history_punish();
  
  string previous_layer_name = "";
  int local_blob_index = 0;
  for (int i = 0; i < history_.size(); ++i) { // i: param_id
    // Add history
    BlobProto* history_blob = state.add_history();
    BlobProto* history_score_blob = state.add_history_score();
    BlobProto* history_punish_blob = state.add_history_punish();
    history_[i]->ToProto(history_blob);
    
    // Add history_score and history_punish
    const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[i].first];
    if (APP<Dtype>::layer_index.count(layer_name)
        && (APP<Dtype>::prune_coremthd.substr(0, 3) == "Reg" or APP<Dtype>::prune_coremthd.substr(0, 2) == "PP")) {
      const int L = APP<Dtype>::layer_index[layer_name];
      if (APP<Dtype>::prune_ratio[L] > 0) { // Only add the layers which want to be pruned.
        local_blob_index = layer_name == previous_layer_name ? local_blob_index + 1 : 0;
        this->net_->layer_by_name(layer_name)->history_score()[local_blob_index]->ToProto(history_score_blob); // Save history_score for weights as well as biases
        this->net_->layer_by_name(layer_name)->history_punish()[local_blob_index]->ToProto(history_punish_blob);
        previous_layer_name = layer_name;
      }
    }
  }    
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate", prefix);
  LOG(INFO)
      << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

// TODO(@mingsuntse): Add history_score etc. in .h5
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
  const string& state_file, const bool& restore_prune_state) {
  SolverState state;
  ReadProtoFromBinaryFile(state_file, &state);
  this->iter_ = state.iter();
  APP<Dtype>::step_ = this->iter_ + 1;
  
  if (restore_prune_state) {
    APP<Dtype>::prune_state = state.prune_state();
    APP<Dtype>::prune_stage = state.prune_stage();
    APP<Dtype>::stage_iter_prune_finished = state.stage_iter_prune_finished();
    APP<Dtype>::last_feasible_prune_iter = state.last_feasible_prune_iter();
    current_prune_ratio_[0]->FromProto(state.current_prune_ratio(0));
    last_feasible_prune_ratio_[0]->FromProto(state.last_feasible_prune_ratio(0));
    last_infeasible_prune_ratio_[0]->FromProto(state.last_infeasible_prune_ratio(0));
    for (int L = 0; L < APP<Dtype>::layer_index.size(); ++L) {
      APP<Dtype>::current_prune_ratio[L] = current_prune_ratio_[0]->mutable_cpu_data()[L];
      APP<Dtype>::last_feasible_prune_ratio[L] = last_feasible_prune_ratio_[0]->mutable_cpu_data()[L];
      APP<Dtype>::last_infeasible_prune_ratio[L] = last_infeasible_prune_ratio_[0]->mutable_cpu_data()[L];
    }
  }
  
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  if (APP<Dtype>::prune_method != "None") { // Check the size of history_score and history_punish
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
    
    // Restore history_punish, history_score
    const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[i].first];
    if (APP<Dtype>::layer_index.count(layer_name) &&
        (APP<Dtype>::prune_coremthd.substr(0, 3) == "Reg" or APP<Dtype>::prune_coremthd.substr(0, 2) == "PP")) {
      const int L = APP<Dtype>::layer_index[layer_name];
      if (APP<Dtype>::prune_ratio[L] > 0) {
        local_blob_index = layer_name == previous_layer_name ? local_blob_index + 1 : 0;
        this->net_->layer_by_name(layer_name)->history_score()[local_blob_index]->FromProto(state.history_score(i));
        this->net_->layer_by_name(layer_name)->history_punish()[local_blob_index]->FromProto(state.history_punish(i));
        previous_layer_name = layer_name;
      }
    }
  }
}

template <typename Dtype>
const int SGDSolver<Dtype>::GetLayerIndex(const int& param_id) {
  // Four occasions to return, `-1` means return
  // 1. Get layer index and layer name, if not registered, don't reg it.
  const string& layer_name = this->net_->layer_names()[this->net_->param_layer_indices()[param_id].first];
  if (APP<Dtype>::layer_index.count(layer_name) == 0) {
    return -1;
  }
  const int L = APP<Dtype>::layer_index[layer_name];

  // 2. Do not reg biases
  const vector<int>& shape = this->net_->learnable_params()[param_id]->shape();
  if (shape.size() == 1) {
    return -1;
  }

  // 3.
  const bool IF_want_prune  = APP<Dtype>::prune_method != "None" && APP<Dtype>::prune_ratio[L] > 0;
  const bool IF_been_pruned = APP<Dtype>::pruned_ratio[L] > 0;
  const bool IF_enough_iter = APP<Dtype>::step_ >= APP<Dtype>::prune_begin_iter + 1;
  const bool IF_in_prune    = APP<Dtype>::prune_state == "prune";
  const bool IF_prune = IF_want_prune && (IF_been_pruned || IF_enough_iter) && IF_in_prune;
  if (!(IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX)) {
    return -1;
  }

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
