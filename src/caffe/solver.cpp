#include <cstdio>

#include <string>
#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#include <ctime>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <numeric>

#include "boost/algorithm/string.hpp"
#define MUL_LR_DECAY 0.1 // the multiplier of lr decay
#define MAX_CNT_LR_DECAY 2 // the max number of lr decay
#define ACCURACY_GAP_THRESHOLD 0.0005
#define INCRE_PR_BOTTOMLINE 0.01
#define CNT_AFTER_MAX_ACC 4
#define COEEF_ACC_2_PR 10 // multiplier of acc margin to incre_pr
#define TR_MUL_BOTTOM 0.25 // the bottomline of target_reg multiplier
#define STANDARD_INCRE_PR 0.05

namespace caffe {

template<typename Dtype>
void Solver<Dtype>::SetActionFunction(ActionCallback func) {
  action_request_function_ = func;
}

template<typename Dtype>
SolverAction::Enum Solver<Dtype>::GetRequestedAction() {
  if (action_request_function_) {
    // If the external request function has been set, call it.
    return action_request_function_();
  }
  return SolverAction::NONE;
}

template <typename Dtype>
//- construct solver with SolverParameter
Solver<Dtype>::Solver(const SolverParameter& param, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  Init(param);
}

template <typename Dtype>
// construct solver with param_file
Solver<Dtype>::Solver(const string& param_file, const Solver* root_solver)
    : net_(), callbacks_(), root_solver_(root_solver),
      requested_early_exit_(false) {
  SolverParameter param;
  ReadSolverParamsFromTextFileOrDie(param_file, &param);
  Init(param);
}

template <typename Dtype>
void Solver<Dtype>::Init(const SolverParameter& param) {
  CHECK(Caffe::root_solver() || root_solver_)
      << "root_solver_ needs to be set for all non-root solvers";
  LOG_IF(INFO, Caffe::root_solver()) << "Initializing solver from parameters: "
    << std::endl << param.DebugString();
  param_ = param;
  
  // ------------------------------------------
  // copy prune params
  APP<Dtype>::prune_method = param_.prune_method();
  if (APP<Dtype>::prune_method != "None") {
      char* mthd = new char[strlen(APP<Dtype>::prune_method.c_str()) + 1];
      strcpy(mthd, APP<Dtype>::prune_method.c_str());
      APP<Dtype>::prune_coremthd = strtok(mthd, "_"); // mthd is like "Reg_Col", the first split is `Reg`
      APP<Dtype>::prune_unit = strtok(NULL, "_"); // TODO(@mingsuntse): put this in APP's member function

      char* coremthd = new char[strlen(APP<Dtype>::prune_coremthd.c_str()) + 1];
      strcpy(coremthd, APP<Dtype>::prune_coremthd.c_str());
      APP<Dtype>::prune_coremthd_ = strtok(coremthd, "-");
  }
  APP<Dtype>::prune_interval = 1; // param_.prune_interval();
  APP<Dtype>::prune_begin_iter = -1;
  APP<Dtype>::AA = param_.aa();
  APP<Dtype>::target_reg = min(param_.target_reg() * IncrePR_2_TRMul(APP<Dtype>::prune_ratio_begin_ave), (Dtype)10);
  APP<Dtype>::kk  = 0.25;
  APP<Dtype>::speedup = param_.speedup();
  APP<Dtype>::compRatio = param_.compratio();
  APP<Dtype>::IF_update_row_col = param.if_update_row_col();
  APP<Dtype>::IF_speedup_count_fc = param.if_speedup_count_fc();
  APP<Dtype>::IF_compr_count_conv = param.if_compr_count_conv();
  APP<Dtype>::IF_scheme1_when_Reg_rank = param.if_scheme1_when_reg_rank();
  APP<Dtype>::IF_eswpf = param_.if_eswpf(); // if early stop when prune finished
  
  APP<Dtype>::iter_size = (APP<Dtype>::prune_method == "None") ? 1 : param_.iter_size_prune();
  APP<Dtype>::baseline_acc = param_.baseline_acc();
  APP<Dtype>::acc_borderline = param_.acc_borderline();
  CHECK_GE(param_.baseline_acc(), param_.acc_borderline()); // if acc_borderline > baseline_acc, it probably will cause bugs later.
  APP<Dtype>::retrain_test_interval = param_.retrain_test_interval();
  APP<Dtype>::losseval_interval = param_.losseval_interval();
  // ------------------------------------------

  CHECK_GE(param_.average_loss(), 1) << "average_loss should be non-negative.";
  CheckSnapshotWritePermissions();
  if (Caffe::root_solver() && param_.random_seed() >= 0) {
    Caffe::set_random_seed(param_.random_seed());
  }
  // Scaffolding code
  InitTrainNet();

  if (Caffe::root_solver()) {
    InitTestNets();
    LOG(INFO) << "Solver scaffolding done.";
  }
  iter_ = 0;
  current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::InitTrainNet() {
  const int num_train_nets = param_.has_net() + param_.has_net_param() +
      param_.has_train_net() + param_.has_train_net_param();
  const string& field_names = "net, net_param, train_net, train_net_param";
  CHECK_GE(num_train_nets, 1) << "SolverParameter must specify a train net "
      << "using one of these fields: " << field_names;
  CHECK_LE(num_train_nets, 1) << "SolverParameter must not contain more than "
      << "one of these fields specifying a train_net: " << field_names;
  NetParameter net_param;
  if (param_.has_train_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in train_net_param.";
    net_param.CopyFrom(param_.train_net_param());
  } else if (param_.has_train_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from train_net file: " << param_.train_net();
    ReadNetParamsFromTextFileOrDie(param_.train_net(), &net_param);
  }
  if (param_.has_net_param()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net specified in net_param.";
    net_param.CopyFrom(param_.net_param());
  }
  if (param_.has_net()) {
    LOG_IF(INFO, Caffe::root_solver())
        << "Creating training net from net file: " << param_.net();
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param);
    APP<Dtype>::model_prototxt = param_.net();
  }
  
  // Set the correct NetState.  We start with the solver defaults (lowest
  // precedence); then, merge in any NetState specified by the net_param itself;
  // finally, merge in any NetState specified by the train_state (highest
  // precedence).
  NetState net_state;
  net_state.set_phase(TRAIN);
  net_state.MergeFrom(net_param.state());
  net_state.MergeFrom(param_.train_state());
  net_param.mutable_state()->CopyFrom(net_state);
  if (Caffe::root_solver()) {
    net_.reset(new Net<Dtype>(net_param));
  } else {
    net_.reset(new Net<Dtype>(net_param, root_solver_->net_.get()));
  }
}

template <typename Dtype>
void Solver<Dtype>::InitTestNets() {
  CHECK(Caffe::root_solver());
  const bool has_net_param = param_.has_net_param();
  const bool has_net_file = param_.has_net();
  const int num_generic_nets = has_net_param + has_net_file;
  CHECK_LE(num_generic_nets, 1)
      << "Both net_param and net_file may not be specified.";
  const int num_test_net_params = param_.test_net_param_size();
  const int num_test_net_files = param_.test_net_size();
  const int num_test_nets = num_test_net_params + num_test_net_files;
  if (num_generic_nets) {
      CHECK_GE(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  } else {
      CHECK_EQ(param_.test_iter_size(), num_test_nets)
          << "test_iter must be specified for each test network.";
  }
  // If we have a generic net (specified by net or net_param, rather than
  // test_net or test_net_param), we may have an unlimited number of actual
  // test networks -- the actual number is given by the number of remaining
  // test_iters after any test nets specified by test_net_param and/or test_net
  // are evaluated.
  const int num_generic_net_instances = param_.test_iter_size() - num_test_nets;
  const int num_test_net_instances = num_test_nets + num_generic_net_instances;
  if (param_.test_state_size()) {
    CHECK_EQ(param_.test_state_size(), num_test_net_instances)
        << "test_state must be unspecified or specified once per test net.";
  }
  if (num_test_net_instances) {
    if (APP<Dtype>::prune_method != "None" && APP<Dtype>::test_gpu_id != -1) {
      return;
    }
    CHECK_GT(param_.test_interval(), 0);
  }
  int test_net_id = 0;
  vector<string> sources(num_test_net_instances);
  vector<NetParameter> net_params(num_test_net_instances);
  for (int i = 0; i < num_test_net_params; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net_param";
      net_params[test_net_id].CopyFrom(param_.test_net_param(i));
  }
  for (int i = 0; i < num_test_net_files; ++i, ++test_net_id) {
      sources[test_net_id] = "test_net file: " + param_.test_net(i);
      ReadNetParamsFromTextFileOrDie(param_.test_net(i),
          &net_params[test_net_id]);
  }
  const int remaining_test_nets = param_.test_iter_size() - test_net_id;
  if (has_net_param) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net_param";
      net_params[test_net_id].CopyFrom(param_.net_param());
    }
  }
  if (has_net_file) {
    for (int i = 0; i < remaining_test_nets; ++i, ++test_net_id) {
      sources[test_net_id] = "net file: " + param_.net();
      ReadNetParamsFromTextFileOrDie(param_.net(), &net_params[test_net_id]);
    }
  }
  test_nets_.resize(num_test_net_instances);
  for (int i = 0; i < num_test_net_instances; ++i) {
    // Set the correct NetState.  We start with the solver defaults (lowest
    // precedence); then, merge in any NetState specified by the net_param
    // itself; finally, merge in any NetState specified by the test_state
    // (highest precedence).
    NetState net_state;
    net_state.set_phase(TEST);
    net_state.MergeFrom(net_params[i].state());
    if (param_.test_state_size()) {
      net_state.MergeFrom(param_.test_state(i));
    }
    net_params[i].mutable_state()->CopyFrom(net_state);
    LOG(INFO)
        << "Creating test net (#" << i << ") specified by " << sources[i];
    if (Caffe::root_solver()) {
      test_nets_[i].reset(new Net<Dtype>(net_params[i]));
    } else {
      test_nets_[i].reset(new Net<Dtype>(net_params[i],
          root_solver_->test_nets_[i].get()));
    }
    test_nets_[i]->set_debug_info(param_.debug_info());
  }
}

template <typename Dtype>
void Solver<Dtype>::Step(int iters) {
  const int start_iter = iter_;
  const int stop_iter = iter_ + iters;
  int average_loss = this->param_.average_loss();
  losses_.clear();
  smoothed_loss_ = 0;
  
  first_retrain_finished_iter_ = 0;
  current_max_acc_ = 0;
  current_max_acc_iter_ = 0;
  current_max_acc_index_ = 0;
  max_acc_ = 0;
  max_acc_iter_ = 0;
  lr_state_start_ = 0;
  last_retrain_lr_ = 0;
  cnt_decay_lr_ = 0;
  
  time_t rawtime;
  time(&rawtime);
  const struct tm* timeinfo = localtime(&rawtime);
  strftime(time_buffer_, 50, " (%Y/%m/%d-%H:%M)", timeinfo);
  Dtype current_speedup, current_compRatio, GFLOPs_origin, num_param_origin;
  if (APP<Dtype>::prune_method != "None") {
    
    GetPruneProgress(&current_speedup,
                     &current_compRatio,
                     &GFLOPs_origin,
                     &num_param_origin);
    cout << "[app] Training starts, iter: " << iter_ << time_buffer_
         << ", speedup: " << current_speedup 
         << ", compRatio: " << current_compRatio << endl;
  
    // This is the begining of training, so snapshot as stage0's output.
    UpdateSnapshotNaming();
    if (iter_ == 0) {
      Snapshot("_stage0");
      ++ APP<Dtype>::prune_stage;
      UpdateSnapshotNaming();
      APP<Dtype>::last_feasible_acc = APP<Dtype>::baseline_acc;
      APP<Dtype>::accumulated_ave_incre_pr = APP<Dtype>::prune_ratio_begin_ave;
      APP<Dtype>::last_prune_ratio_incre = APP<Dtype>::prune_ratio_begin_ave;
    }
  }
  
  while (iter_ < stop_iter) {
    APP<Dtype>::step_ = iter_ + 1;
    time(&rawtime);
    const struct tm* timeinfo = localtime(&rawtime);
    strftime(time_buffer_, 50, " (%Y/%m/%d-%H:%M)", timeinfo);
    
    // zero-init the params
    net_->ClearParamDiffs();
    if (param_.test_interval() && iter_ % param_.test_interval() == 0
        && (iter_ > 0 || param_.test_initialization())
        && Caffe::root_solver()) {
      TestAll();
      if (requested_early_exit_) {
        // Break out of the while loop because stop was requested while testing.
        break;
      }
    }
    // std::cout << "call_backs_.size(): " << callbacks_.size() << std::endl;
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;

    // Speed check
    cout << "--- Solver begins timing" << endl;
    clock_t t1 = clock();
    
    APP<Dtype>::inner_iter = 0;
    for (int i = 0; i < APP<Dtype>::iter_size * param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
      ++ APP<Dtype>::inner_iter;
    }
    cout << "--- after ForwardBackward: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
    
    loss /= (APP<Dtype>::iter_size * param_.iter_size());
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    
    if (display) {
      // -----------------------------------------------------------------------------------
      // calculate training speed
      const time_t current_time = time(NULL);
      if (APP<Dtype>::last_time == 0) {
          APP<Dtype>::first_time = current_time;
          APP<Dtype>::first_iter = iter_;
      }
      char train_speed[50];
      sprintf(train_speed, "%.3f(%.3f)s/iter", (current_time - APP<Dtype>::last_time ) * 1.0 / param_.display(),
                                               (current_time - APP<Dtype>::first_time) * 1.0 / (iter_ - APP<Dtype>::first_iter));
      APP<Dtype>::last_time = current_time;
      // -----------------------------------------------------------------------------------
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", smoothed loss = " << smoothed_loss_
          << ", speed = " << train_speed;
      const vector<Blob<Dtype>*>& result = net_->output_blobs();
      int score_index = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        const string& output_name =
            net_->blob_names()[net_->output_blob_indices()[j]];
        const Dtype loss_weight =
            net_->blob_loss_weights()[net_->output_blob_indices()[j]];
        for (int k = 0; k < result[j]->count(); ++k) {
          ostringstream loss_msg_stream;
          if (loss_weight) {
            loss_msg_stream << " (* " << loss_weight
                            << " = " << loss_weight * result_vec[k] << " loss)";
          }
          LOG_IF(INFO, Caffe::root_solver()) << "    Train net output #"
              << score_index++ << ": " << output_name << " = "
              << result_vec[k] << loss_msg_stream.str();
        }
      }
    }
    
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    
    ApplyUpdate();
    cout << "--- after ApplyUpdate: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;

    // -----------------------------------------------------------------
    if (APP<Dtype>::prune_method != "None") {
      // Prune finished
      if(APP<Dtype>::prune_state == "prune" && APP<Dtype>::IF_current_target_achieved) {
        GetPruneProgress(&current_speedup,
                         &current_compRatio,
                         &GFLOPs_origin,
                         &num_param_origin);
                         
        cout << "[app]\n[app] Current pruning stage (stage = "
             << APP<Dtype>::prune_stage << ") finished. Go on training for a little while before checking accuracy."
             << " speedup: " << current_speedup
             << ", iter: " << APP<Dtype>::stage_iter_prune_finished << time_buffer_ << endl;
             
        for (int L = 0; L < APP<Dtype>::layer_index.size(); ++L) {
          if (APP<Dtype>::prune_ratio[L] == 0) { continue; }
          cout << "[app]    " << L << " - pruned_ratio: " << APP<Dtype>::pruned_ratio_col[L] << endl;
        }
        SetPruneState("losseval"); // Going to prune_state 'losseval'
        
        // Check reg
        map<string, int>::iterator map_it;
        for (int L = 0; L < APP<Dtype>::layer_index.size(); ++L) {
          if (APP<Dtype>::prune_ratio[L] == 0) { continue; }
          string layer_name = "";
          for (map_it = APP<Dtype>::layer_index.begin(); map_it != APP<Dtype>::layer_index.end(); ++map_it) {
            if (map_it->second == L) {
              layer_name = map_it->first;
              break;
            }
          }
          Dtype* muhistory_punish = this->net_->layer_by_name(layer_name)->history_punish()[0]->mutable_cpu_data();
          const int num_col = this->net_->layer_by_name(layer_name)->blobs()[0]->count(1);
          const int num_row = this->net_->layer_by_name(layer_name)->blobs()[0]->shape()[0];
          vector<Dtype> left_reg;
          for (int j = 0; j < num_col; ++j) {
            if (APP<Dtype>::IF_col_pruned[L][j][0] == false && 0 < muhistory_punish[j] && muhistory_punish[j] < APP<Dtype>::target_reg) {
              left_reg.push_back(muhistory_punish[j]);
              for (int i = 0; i < num_row; ++i) {
                muhistory_punish[i * num_col + j] = 0;
              }
            }
          }
          if (left_reg.size()) {
            cout << "[app]    " << L << " - " << left_reg.size() << " columns' left reg not cleared, now cleared:";
            for (int i = 0; i < left_reg.size(); ++i) {
              cout << " " << left_reg[i];
            }
            cout << endl;
          }
        }
      }

      // Check acc based on loss
      if (APP<Dtype>::prune_state == "losseval" && iter_ - APP<Dtype>::stage_iter_prune_finished == APP<Dtype>::losseval_interval) {
        cout << "[app]    'losseval' done, retrain to check accuracy before starting a new pruning stage. iter: " << iter_ << time_buffer_ << endl;
        SetPruneState("retrain");
      }
      
      // Retrain, check acc
      if (APP<Dtype>::prune_state == "retrain" 
            && APP<Dtype>::retrain_test_interval 
            && iter_ % APP<Dtype>::retrain_test_interval == 0) {
        if (APP<Dtype>::acc_borderline <= 0) {
          CheckMaxAcc("retrain", CNT_AFTER_MAX_ACC + 2);
        } else {
          CheckMaxAcc("retrain", CNT_AFTER_MAX_ACC);
        }
      }
      
      // Final retrain, check acc
      if (APP<Dtype>::prune_state == "final_retrain"
            && APP<Dtype>::retrain_test_interval
            && iter_ % APP<Dtype>::retrain_test_interval == 0
            && state_begin_iter_ != iter_) { // do not test on the the first 'final_retrain' iter, because it's unnecessary and harmful
        CheckMaxAcc("final_retrain", CNT_AFTER_MAX_ACC + 4);
      }

      // Print speedup & compression ratio each iter
      GetPruneProgress(&current_speedup,
                       &current_compRatio,
                       &GFLOPs_origin,
                       &num_param_origin);
      if (start_iter == iter_) {
        cout << "IF_speedup_count_fc: " << APP<Dtype>::IF_speedup_count_fc
             << "  Total GFLOPs_origin: " << GFLOPs_origin
             << " | IF_compr_count_conv: " << APP<Dtype>::IF_compr_count_conv
             << "  Total num_param_origin: " << num_param_origin << endl;
      }
      cout << "**** Step " << APP<Dtype>::step_ << " (after update): " 
           << current_speedup   << "/" << APP<Dtype>::speedup << " "
           << current_compRatio << "/" << APP<Dtype>::compRatio
           << " ****" << "\n" << endl;
    }
    // -----------------------------------------------------------------
    
    // Increment the internal iter_ counter -- its value should always indicate
    // the number of times the weights have been updated.
    ++iter_;
    
    SolverAction::Enum request = GetRequestedAction();

    // Save a snapshot if needed.
    if ((param_.snapshot()
         && iter_ % param_.snapshot() == 0
         && Caffe::root_solver()) ||
         (request == SolverAction::SNAPSHOT)) {
      Snapshot();
      // Remove useless caffemodels and solverstates to save disk space
      if (APP<Dtype>::prune_method != "None") {
        if (snapshot_iters_.size()) {
          RemoveUselessSnapshot("", snapshot_iters_.back());
        }
        snapshot_iters_.push_back(iter_);
      }
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::CheckMaxAcc(const string& prune_state, const int& cnt_after_max_acc) {
  if(APP<Dtype>::test_gpu_id != -1) {
    OfflineTest();
  } else {
    TestAll();
  }
  const Dtype acc1 = *min_element(test_accuracy_.begin(), test_accuracy_.end());
  const Dtype acc5 = *max_element(test_accuracy_.begin(), test_accuracy_.end());
  APP<Dtype>::retrain_test_acc1.push_back(acc1);
  APP<Dtype>::retrain_test_acc5.push_back(acc5);
  retrain_accs_.push_back(acc1);
  saved_retrain_iters_.push_back(iter_);

  if (acc1 > current_max_acc_) {
    current_max_acc_ = acc1;
    current_max_acc_index_ = APP<Dtype>::retrain_test_acc1.size();
    current_max_acc_iter_ = iter_;
  }
  if (acc1 > max_acc_) {
    max_acc_ = acc1;
    max_acc_iter_ = iter_;
  }
  char logstr[500];
  sprintf(logstr, "[app]    '%s' going on, current acc1 = %f, iter: %d", prune_state.c_str(), acc1, iter_);
  cout << logstr << time_buffer_ << endl;
  
  // Current lr period done
  const int CntDelta = (cnt_decay_lr_ == 0) * (prune_state == "retrain") * 2; // Give the first lr period more time to train, which is good for accuracy recovery.
  if (APP<Dtype>::retrain_test_acc1.size() - current_max_acc_index_ > cnt_after_max_acc + CntDelta) {
    const string prefix = (prune_state == "retrain") ? retrain_prefix_ : finalretrain_prefix_;
    if (first_retrain_finished_iter_ == 0) {
      first_retrain_finished_iter_ = current_max_acc_iter_;
    }
    
    // Decay lr
    APP<Dtype>::learning_rate *= MUL_LR_DECAY; // When current learning rate has reached its ceiling accuracy, decay it.
    ++ cnt_decay_lr_;
    sprintf(logstr, "[app]    '%s' of current lr period finished, final acc1 = %f, iter = %d, decay lr (new: %.7f)",
          prune_state.c_str(), current_max_acc_, current_max_acc_iter_, APP<Dtype>::learning_rate);
    cout << logstr << time_buffer_ << endl;
    
    // Resume
    const string resume_file = param_.snapshot_prefix() + prefix + "_iter_" + caffe::format_int(current_max_acc_iter_) + ".solverstate";
    Restore(resume_file.c_str(), false); // Restore to the best model in this lr period
    // Remove useless caffemodels
    for (int i = 0; i < saved_retrain_iters_.size(); ++i) {
      if (saved_retrain_iters_[i] != first_retrain_finished_iter_ && saved_retrain_iters_[i] != max_acc_iter_) { // spare these two caffemodels, because they may be restored later
        RemoveUselessSnapshot(prefix, saved_retrain_iters_[i]);
      }
    }
    saved_retrain_iters_.clear();
    saved_retrain_iters_.push_back(max_acc_iter_);

    // "final_retrain" use fixed lr, so once this lr period finished, all "final_retrain" done.
    if (prune_state == "final_retrain") {
      const Dtype final_output_acc = max(max_acc_, APP<Dtype>::last_feasible_acc2);
      const int final_output_iter = (max_acc_ > APP<Dtype>::last_feasible_acc2) ? max_acc_iter_ : APP<Dtype>::last_feasible_prune_iter2;
      sprintf(logstr, "[app]    All '%s' done. Output the best caffemodel, iter = %d, acc1 = %f", prune_state.c_str(), max_acc_iter_, max_acc_);
      cout << logstr << endl;
      sprintf(logstr, "[app]    All prune done. Output the best caffemodel, iter = %d, acc1 = %f", final_output_iter, final_output_acc);
      cout << logstr << endl;
      PrintFinalPrunedRatio();
      RemoveUselessSnapshot("", snapshot_iters_.back());
      exit(0);
    }
    
    // Check if retraining can be stopped in "retrain" state
    if (cnt_decay_lr_ >= MAX_CNT_LR_DECAY + 1 || current_max_acc_ < max_acc_) {
      APP<Dtype>::learning_rate /= MUL_LR_DECAY; // restore to last lr, because this lr is not used actually.
      sprintf(logstr, "[app]    All '%s' done: lr has decayed enough OR max acc of this lr period is not better than the previous one.", prune_state.c_str());
      cout << logstr << " Output the best caffemodel, iter = " << max_acc_iter_ << ", acc1 = " << max_acc_
           << ". Resuming from iter = " << first_retrain_finished_iter_ << endl;
      // Resume
      const string resume_file = param_.snapshot_prefix() + prefix + "_iter_" + caffe::format_int(first_retrain_finished_iter_) + ".solverstate";
      Restore(resume_file.c_str(), false); // Restore before removing the useless caffemodel
      if (max_acc_iter_ != first_retrain_finished_iter_) {
        RemoveUselessSnapshot(prefix, first_retrain_finished_iter_);
      }
      // Clear for next cycle of retraining
      last_retrain_lr_ = APP<Dtype>::learning_rate; // for potential use in 'final_retrain'
      APP<Dtype>::learning_rate = lr_state_start_; // restore to previous learning_rate
      first_retrain_finished_iter_ = 0;
      cnt_decay_lr_ = 0;
      saved_retrain_iters_.clear();
      
      // Check accuracy
      sort(retrain_accs_.begin(), retrain_accs_.end(), greater<Dtype>()); // in descending order
      CheckPruneStage((retrain_accs_[0] + retrain_accs_[1] + retrain_accs_[2]) / 3, max_acc_iter_, max_acc_); // use averaged acc to alleviate the influence of acc impulse
      max_acc_ = 0;
      max_acc_iter_ = 0;
      retrain_accs_.clear();
      
      if (APP<Dtype>::acc_borderline <= 0) {
        cout << "[app]    'Given pr to get acc' task done!" << endl;
        exit(0);
      }
    }

    // Clear for next retraining cycle
    APP<Dtype>::retrain_test_acc1.clear();
    APP<Dtype>::retrain_test_acc5.clear();
    current_max_acc_ = 0;
    current_max_acc_index_ = 0;
    current_max_acc_iter_ = 0;
  }
}

template <typename Dtype>
void Solver<Dtype>::GetPruneProgress(Dtype* speedup, Dtype* compRatio, Dtype* GFLOPs_origin_, Dtype* num_param_origin_) {
  // speedup
  Dtype GFLOPs_left   = 0;
  Dtype GFLOPs_origin = 0;
  const int num_layer_count = APP<Dtype>::IF_speedup_count_fc ? APP<Dtype>::layer_index.size() : APP<Dtype>::conv_layer_cnt;
  for (int i = 0; i < num_layer_count; ++i) {
      const Dtype pr = APP<Dtype>::pruned_ratio_row[i];
      const Dtype pc = APP<Dtype>::pruned_ratio_col[i];
      GFLOPs_left   += APP<Dtype>::GFLOPs[i] * (1 - (pr + pc - pr * pc));
      GFLOPs_origin += APP<Dtype>::GFLOPs[i];
  }
  if (APP<Dtype>::prune_unit == "Col" || APP<Dtype>::prune_unit == "Row") {
      APP<Dtype>::IF_speedup_achieved = GFLOPs_origin / GFLOPs_left >= APP<Dtype>::speedup;
  }
  *speedup = GFLOPs_origin / GFLOPs_left;
  *GFLOPs_origin_ = GFLOPs_origin;
  
  // compression ratio
  Dtype num_param_left   = 0;
  Dtype num_param_origin = 0;
  const int num_layer_begin = APP<Dtype>::IF_compr_count_conv ? 0 : APP<Dtype>::conv_layer_cnt;
  for (int i = num_layer_begin; i < APP<Dtype>::layer_index.size(); ++i) {
      num_param_left   += APP<Dtype>::num_param[i] * (1 - APP<Dtype>::pruned_ratio[i]);
      num_param_origin += APP<Dtype>::num_param[i];
  }
  if (APP<Dtype>::prune_unit == "Weight") {
      APP<Dtype>::IF_compRatio_achieved = num_param_origin / num_param_left >= APP<Dtype>::compRatio;
  }
  *compRatio = num_param_origin / num_param_left;
  *num_param_origin_ = num_param_origin;
}

template <typename Dtype>
void Solver<Dtype>::SetPruneState(const string& prune_state) {
  APP<Dtype>::prune_state = prune_state;
  if (prune_state == "prune") {
    APP<Dtype>::iter_size = this->param_.iter_size_prune();
    for (int L = 0; L < APP<Dtype>::layer_index.size(); ++L) {
      if (APP<Dtype>::prune_ratio[L] == 0) { continue; }
      APP<Dtype>::iter_prune_finished[L] = INT_MAX;
    }
    APP<Dtype>::IF_current_target_achieved = false;
  } else if (prune_state == "losseval") {
    APP<Dtype>::iter_size = APP<Dtype>::acc_borderline > 0 ? this->param_.iter_size_losseval() : this->param_.iter_size_final_retrain();
  } else if (prune_state == "retrain") {
    lr_state_start_ = APP<Dtype>::learning_rate; // for potential restore later
    APP<Dtype>::iter_size = APP<Dtype>::acc_borderline > 0 ? this->param_.iter_size_retrain() : this->param_.iter_size_final_retrain();
  } else if (prune_state == "final_retrain") {
    state_begin_iter_ = iter_;
    APP<Dtype>::iter_size = this->param_.iter_size_final_retrain();
    APP<Dtype>::learning_rate = last_retrain_lr_;
  } else {
    LOG(INFO) << "Wrong: unknown prune_state, please check." << endl;
    exit(1);
  }
}

template <typename Dtype>
void Solver<Dtype>::CheckPruneStage(const Dtype& acc, const int& last_max_acc_iter, const Dtype& last_max_acc) {
  if (APP<Dtype>::acc_borderline - acc > ACCURACY_GAP_THRESHOLD) { // accuracy bad
    for (int L = 0; L < APP<Dtype>::layer_index.size(); ++L) {
      if (APP<Dtype>::prune_ratio[L] == 0) { continue; }
      APP<Dtype>::last_infeasible_prune_ratio[L] = APP<Dtype>::pruned_ratio_for_comparison[L];
    }
    cout << "[app]    accuracy **bad**, going to roll back weights to iter = " << APP<Dtype>::last_feasible_prune_iter 
         << " (stage = " << APP<Dtype>::prune_stage - 1 << "). Reassign the incre_pr:" << endl;
    const Dtype incre_pr = SetNewCurrentPruneRatio(true, acc);
    const string resume_file = param_.snapshot_prefix() + laststage_prefix_ + "_iter_" + caffe::format_int(APP<Dtype>::last_feasible_prune_iter) + ".solverstate";
    cout << "[app]    ===== resuming from: " << resume_file << endl;
    Restore(resume_file.c_str(), false); // Note to restore after SetNewCurrentPruneRatio, because restore will change the state of network, like num_pruned_col
    SetPruneState("prune");
    // Check if incre_pr is large enough
    if (incre_pr < INCRE_PR_BOTTOMLINE) {
      cout << "[app]\n[app] Stop: incre_pr is too small (<" << INCRE_PR_BOTTOMLINE << "), so another pruning stage is meaningless. Go to 'final_retrain'." << endl;
      const string resume_file = param_.snapshot_prefix() + lastretrain_prefix_ + "_iter_" + caffe::format_int(APP<Dtype>::last_feasible_prune_iter2) + ".solverstate";
      Restore(resume_file.c_str(), false);
      cout << "[app]    ===== resuming from: " << resume_file << endl;
      SetPruneState("final_retrain");
    }
  } else { // accuracy good
    APP<Dtype>::last_feasible_prune_iter = iter_;
    APP<Dtype>::last_feasible_prune_iter2= last_max_acc_iter;
    APP<Dtype>::last_feasible_acc = acc;
    APP<Dtype>::last_feasible_acc2= last_max_acc;
    for (int L = 0; L < APP<Dtype>::layer_index.size(); ++L) {
      if (APP<Dtype>::prune_ratio[L] == 0) { continue; }
      APP<Dtype>::last_feasible_prune_ratio[L] = APP<Dtype>::pruned_ratio_for_comparison[L];
    }
    cout << "[app]    accuracy **still good**, save caffemodel, start a new pruning stage." << endl;
    SetNewCurrentPruneRatio(false, acc);
    SetPruneState("prune");
    Snapshot(stage_prefix_);
    ++ APP<Dtype>::prune_stage;
    UpdateSnapshotNaming();
  }
  CheckIfFinalTargetAchieved();
}
           
template <typename Dtype>
void Solver<Dtype>::CheckIfFinalTargetAchieved() {
  bool all_layer_prune_finished = true;
  for (int L = 0; L < APP<Dtype>::layer_index.size(); ++L) {
    if (APP<Dtype>::pruned_ratio_for_comparison[L] < APP<Dtype>::prune_ratio[L]) {
      all_layer_prune_finished = false;
      break;
    }
  }
  const bool IF_final_target_achieved = all_layer_prune_finished || APP<Dtype>::IF_speedup_achieved || APP<Dtype>::IF_compRatio_achieved;
  if (IF_final_target_achieved) {
    cout << "[app]\n[app] All layer prune finished: iter = " << iter_;
    if (APP<Dtype>::IF_eswpf) {
      cout << " - early stopped." << endl;
      PrintFinalPrunedRatio();
      exit(0);
    } else {
      cout << " - Go to 'final_retrain'." << endl;
      const string resume_file = param_.snapshot_prefix() + lastretrain_prefix_ + "_iter_" + caffe::format_int(APP<Dtype>::last_feasible_prune_iter2) + ".solverstate";
      Restore(resume_file.c_str(), false);
      cout << "[app]    ===== resuming from: " << resume_file << endl;
      SetPruneState("final_retrain");
    }
  }
}
           
template <typename Dtype>
const Dtype Solver<Dtype>::SetNewCurrentPruneRatio(const bool& IF_roll_back, const Dtype& val_acc) {
  Dtype incre_pr = 0;
  if (IF_roll_back) {
    APP<Dtype>::accumulated_ave_incre_pr -= APP<Dtype>::last_prune_ratio_incre;
    incre_pr = APP<Dtype>::last_prune_ratio_incre / (APP<Dtype>::last_feasible_acc - val_acc) 
                                                  * (APP<Dtype>::last_feasible_acc - APP<Dtype>::acc_borderline);
  } else {
    incre_pr = min(max((Dtype)INCRE_PR_BOTTOMLINE, (val_acc - APP<Dtype>::acc_borderline) * COEEF_ACC_2_PR), (Dtype)0.2); // range: [INCRE_PR_BOTTOMLINE, 0.2]
  }
  // Check incre_pr
  APP<Dtype>::last_prune_ratio_incre = incre_pr;
  APP<Dtype>::accumulated_ave_incre_pr += incre_pr;
  APP<Dtype>::target_reg = min(param_.target_reg() * IncrePR_2_TRMul(incre_pr), (Dtype)10);
  cout << "[app]    incre_pr: " << incre_pr 
       << " (now ave_pr = " << APP<Dtype>::accumulated_ave_incre_pr
       << ", target_reg = " << APP<Dtype>::target_reg << ")" << endl;
  for (int L  = 0; L < APP<Dtype>::layer_index.size(); ++L) {
    if (APP<Dtype>::prune_ratio[L] == 0) { continue; }
    APP<Dtype>::current_prune_ratio[L] = APP<Dtype>::last_feasible_prune_ratio[L] 
          + incre_pr / APP<Dtype>::STANDARD_SPARSITY * APP<Dtype>::prune_ratio_step[L];
    APP<Dtype>::current_prune_ratio[L] = min(APP<Dtype>::current_prune_ratio[L], APP<Dtype>::prune_ratio[L]);
    APP<Dtype>::iter_prune_finished[L] = INT_MAX;
    cout << "[app]    " << L << " - current_prune_ratio: " 
         << APP<Dtype>::last_feasible_prune_ratio[L] << " -> " << APP<Dtype>::current_prune_ratio[L] 
         << " (+" << APP<Dtype>::current_prune_ratio[L] - APP<Dtype>::last_feasible_prune_ratio[L] << ")" << endl;
  }
  return incre_pr;
}

template <typename Dtype>
void Solver<Dtype>::Solve(const char* resume_file) {
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Solving " << net_->name();
  LOG(INFO) << "Learning Rate Policy: " << param_.lr_policy();

  // Initialize to false every time we start solving.
  requested_early_exit_ = false;

  if (resume_file) {
    LOG(INFO) << "Restoring previous solver status from " << resume_file;
    Restore(resume_file);
  }
  
  // After restoring, calculate GFLOPs and determine whether the prune finished
  // Note that, layer target has been check when restoring solverstate.
  Dtype current_speedup, current_compRatio, GFLOPs_origin, num_param_origin;
  GetPruneProgress(&current_speedup,
                   &current_compRatio,
                   &GFLOPs_origin,
                   &num_param_origin);
  if (APP<Dtype>::IF_speedup_achieved || APP<Dtype>::IF_compRatio_achieved) {
    for (int i = 0; i < APP<Dtype>::layer_index.size(); ++i) {
      APP<Dtype>::iter_prune_finished[i] = -1; 
    }
  }

  // For a network that is trained by the solver, no bottom or top vecs
  // should be given, and we will just provide dummy vecs.
  int start_iter = iter_;
  Step(param_.max_iter() - iter_);

  // If we haven't already, save a snapshot after optimization, unless
  // overridden by setting snapshot_after_train := false
  if (param_.snapshot_after_train()
      && (!param_.snapshot() || iter_ % param_.snapshot() != 0)) {
    Snapshot();
  }
  if (requested_early_exit_) {
    if (APP<Dtype>::prune_method != "None") { PrintFinalPrunedRatio(); }
    LOG(INFO) << "Optimization stopped early.";
    return;
  }
  // After the optimization is done, run an additional train and test pass to
  // display the train and test loss/outputs if appropriate (based on the
  // display and test_interval settings, respectively).  Unlike in the rest of
  // training, for the train net we only run a forward pass as we've already
  // updated the parameters "max_iter" times -- this final pass is only done to
  // display the loss, which is computed in the forward pass.
  if (param_.display() && iter_ % param_.display() == 0) {
    int average_loss = this->param_.average_loss();
    Dtype loss;
    net_->Forward(&loss);
    UpdateSmoothedLoss(loss, start_iter, average_loss);
    LOG(INFO) << "Iteration " << iter_ << ", loss = " << smoothed_loss_;
  }
  if (param_.test_interval() && iter_ % param_.test_interval() == 0) {
    TestAll();
  }
  if (APP<Dtype>::prune_method != "None") { PrintFinalPrunedRatio(); }
  LOG(INFO) << "Optimization Done.";
}

template <typename Dtype>
void Solver<Dtype>::PrintFinalPrunedRatio() {
  cout << "[app]\n[app] Print final pruned ratio of all layers:" << endl;
  map<string, int>::iterator it_m;
  for (it_m = APP<Dtype>::layer_index.begin(); it_m != APP<Dtype>::layer_index.end(); ++it_m) {
    const string layer_name = it_m->first;
    const int L = it_m->second;
    const string shape_str = this->net_->layer_by_name(layer_name)->blobs()[0]->shape_string();
    const int num_row = this->net_->layer_by_name(layer_name)->blobs()[0]->shape()[0];
    const int num_col = this->net_->layer_by_name(layer_name)->blobs()[0]->count(1);
    const int num_pruned_col = APP<Dtype>::num_pruned_col[L];
    const int num_pruned_row = APP<Dtype>::num_pruned_row[L];
    char logstr[500];
    sprintf(logstr, "[app]    %s, shape = %s | num_col = %d, num_pruned_col = %d (%f) | num_row = %d, num_pruned_row = %d (%f)",
           layer_name.c_str(), shape_str.c_str(), num_col, num_pruned_col, num_pruned_col*1.0/num_col, num_row, num_pruned_row, num_pruned_row*1.0/num_row);
    cout << logstr << endl;
  }
}

template <typename Dtype>
void Solver<Dtype>::RemoveUselessSnapshot(const string& prefix, const int& iter) {
  if (iter >= 0) {
    const string caffemodel_path = param_.snapshot_prefix() + prefix + "_iter_" + caffe::format_int(iter) + ".caffemodel";
    const string solverstate_path = param_.snapshot_prefix() + prefix + "_iter_" + caffe::format_int(iter) + ".solverstate";
    std::remove(caffemodel_path.c_str());
    std::remove(solverstate_path.c_str());
  }
}
// ----------------------------------------------------------------------------------
template <typename Dtype>
void Solver<Dtype>::OfflineTest() {
  // Switch GPU
  int gpu_id = APP<Dtype>::test_gpu_id;
  if (gpu_id == -1) {
    gpu_id = APP<Dtype>::original_gpu_id;
  }
  Caffe::SetDevice(gpu_id);
  Caffe::set_mode(Caffe::GPU);
  
  // Create test net
  string prefix;
  if (APP<Dtype>::prune_state == "retrain") {
    prefix = retrain_prefix_;
  } else if (APP<Dtype>::prune_state == "final_retrain") {
    prefix = finalretrain_prefix_;
  }
  Snapshot(prefix);
  const string& weights = param_.snapshot_prefix() + prefix + "_iter_" + caffe::format_int(iter_) + ".caffemodel";
  Net<Dtype> test_net(APP<Dtype>::model_prototxt, caffe::TEST);
  test_net.CopyTrainedLayersFrom(weights);
  
  LOG(INFO) << "-------------------------- retrain test begins --------------------------";
  LOG(INFO) << "Use GPU with device ID " << gpu_id;
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, gpu_id);
  LOG(INFO) << "GPU device name: " << device_prop.name;
 
  const int num_iter = param_.test_iter(0);
  LOG(INFO) << "Running for " << num_iter << " iterations.";
  test_accuracy_.clear();
  vector<int> test_score_output_id;
  vector<Dtype> test_score;
  Dtype loss = 0;
  for (int i = 0; i < num_iter; ++i) {
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net.Forward(&iter_loss);
    loss += iter_loss;
    int idx = 0;
    for (int j = 0; j < result.size(); ++j) {
      const Dtype* result_vec = result[j]->cpu_data();
      for (int k = 0; k < result[j]->count(); ++k, ++idx) {
        const Dtype score = result_vec[k];
        if (i == 0) {
          test_score.push_back(score);
          test_score_output_id.push_back(j);
        } else {
          test_score[idx] += score;
        }
        const std::string& output_name = test_net.blob_names()[test_net.output_blob_indices()[j]];
        if (i % 20 == 0) {
          LOG(INFO) << "Batch " << i << ", " << output_name << " = " << score;
        }
      }
    }
  }
  loss /= num_iter;
  LOG(INFO) << "Loss: " << loss;
  for (int i = 0; i < test_score.size(); ++i) {
    const std::string& output_name = test_net.blob_names()[
        test_net.output_blob_indices()[test_score_output_id[i]]];
    const Dtype loss_weight = test_net.blob_loss_weights()[
        test_net.output_blob_indices()[test_score_output_id[i]]];
    std::ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / num_iter;
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << output_name << " = " << mean_score << loss_msg_stream.str();
    if (output_name.find("ccuracy") != std::string::npos) { // TODO(mingsuntse): improve this
      test_accuracy_.push_back(mean_score);
    }
  }
  LOG(INFO) << "-------------------------- retrain test done --------------------------";
  Caffe::SetDevice(APP<Dtype>::original_gpu_id); // Change back to original gpu
}

template <typename Dtype>
const Dtype Solver<Dtype>::IncrePR_2_TRMul(const Dtype& incre_pr) {
  /*
  x: increment_prune_ratio
  y: multiplier * target_reg
  
  y0 = 1 / (1 + e^(-k(x-0.05))) // which is a sigmoid function
  y1 = 3 * (y0 - 0.5) + 1  constrain the range be in (-0.5, 2.5)

  s.t.
  x = INCRE_PR_BOTTOMLINE  ->  y1 = 0.2 
  x = STANDARD_INCRE_PR   ->  y1 = 1
  */
  const Dtype y0 = (TR_MUL_BOTTOM - 1) / 3 + 0.5;
  const Dtype k = log(1/y0 - 1) / (STANDARD_INCRE_PR - INCRE_PR_BOTTOMLINE);

  const Dtype y0_ = 1 / (1 + exp(-k * (incre_pr - STANDARD_INCRE_PR)));
  const Dtype y1_ = 3 * (y0_ - 0.5) + 1;
  return y1_;
}

template <typename Dtype>
void Solver<Dtype>::TestAll() {
  for (int test_net_id = 0;
       test_net_id < test_nets_.size() && !requested_early_exit_;
       ++test_net_id) {
    Test(test_net_id);
  }
}

template <typename Dtype>
void Solver<Dtype>::Test(const int test_net_id) {
  test_accuracy_.clear();
  CHECK(Caffe::root_solver());
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
  
  if (APP<Dtype>::prune_method != "None") {
    if (APP<Dtype>::prune_state == "retrain") {
      Snapshot(retrain_prefix_); // save caffemodel, because one of them will be restored later
      LOG(INFO) << "-------------------------- retrain test begins --------------------------";
    } else if (APP<Dtype>::prune_state == "final_retrain") {
      Snapshot(finalretrain_prefix_);
      LOG(INFO) << "-------------------------- retrain test begins --------------------------";
    }
  }
  
  for (int i = 0; i < param_.test_iter(test_net_id); ++i) {
    SolverAction::Enum request = GetRequestedAction();
    // Check to see if stoppage of testing/training has been requested.
    while (request != SolverAction::NONE) {
        if (SolverAction::SNAPSHOT == request) {
          Snapshot();
        } else if (SolverAction::STOP == request) {
          requested_early_exit_ = true;
        }
        request = GetRequestedAction();
    }
    if (requested_early_exit_) {
      // break out of test loop.
      break;
    }
    Dtype iter_loss;
    const vector<Blob<Dtype>*>& result =
        test_net->Forward(&iter_loss);
    if (param_.test_compute_loss()) {
      loss += iter_loss;
    }
    if (i == 0) {
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score.push_back(result_vec[k]);
          test_score_output_id.push_back(j);
        }
      }
    } else {
      int idx = 0;
      for (int j = 0; j < result.size(); ++j) {
        const Dtype* result_vec = result[j]->cpu_data();
        for (int k = 0; k < result[j]->count(); ++k) {
          test_score[idx++] += result_vec[k];
        }
      }
    }
  }
  if (requested_early_exit_) {
    LOG(INFO) << "Test interrupted.";
    return;
  }
  if (param_.test_compute_loss()) {
    loss /= param_.test_iter(test_net_id);
    LOG(INFO) << "Test loss: " << loss;
  }
  for (int i = 0; i < test_score.size(); ++i) {
    const int output_blob_index =
        test_net->output_blob_indices()[test_score_output_id[i]];
    const string& output_name = test_net->blob_names()[output_blob_index];
    const Dtype loss_weight = test_net->blob_loss_weights()[output_blob_index];
    ostringstream loss_msg_stream;
    const Dtype mean_score = test_score[i] / param_.test_iter(test_net_id);
    if (loss_weight) {
      loss_msg_stream << " (* " << loss_weight
                      << " = " << loss_weight * mean_score << " loss)";
    }
    LOG(INFO) << "    Test net output #" << i << ": " << output_name << " = "
              << mean_score << loss_msg_stream.str();
    if (output_name.find("ccuracy") != std::string::npos) { // TODO(mingsuntse): improve this
      test_accuracy_.push_back(mean_score);
    }
  }
  if (APP<Dtype>::prune_method != "None" && (APP<Dtype>::prune_state == "retrain" || APP<Dtype>::prune_state == "final_retrain")) {
    LOG(INFO) << "-------------------------- retrain test ends --------------------------";
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSnapshotNaming() {
  stringstream sstream1;
  sstream1 << "_stage" << APP<Dtype>::prune_stage;
  stage_prefix_ = sstream1.str();
  
  stringstream sstream2;
  sstream2 << "_stage" << APP<Dtype>::prune_stage - 1;
  laststage_prefix_ = sstream2.str();

  stringstream sstream3;
  sstream3 << "_stage" << APP<Dtype>::prune_stage << "-retrain";
  retrain_prefix_ = sstream3.str();
  
  stringstream sstream4;
  sstream4 << "_stage" << APP<Dtype>::prune_stage - 1 << "-retrain";
  lastretrain_prefix_ = sstream4.str();
  
  finalretrain_prefix_ = "_finalretrain";
}

template <typename Dtype>
void Solver<Dtype>::Snapshot(const string& prefix) {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto(prefix); // save caffemodel
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }
  SnapshotSolverState(model_filename, prefix); // save solverstate
}

template <typename Dtype>
void Solver<Dtype>::CheckSnapshotWritePermissions() {
  if (Caffe::root_solver() && param_.snapshot()) {
    CHECK(param_.has_snapshot_prefix())
        << "In solver params, snapshot is specified but snapshot_prefix is not";
    string probe_filename = SnapshotFilename(".tempfile");
    std::ofstream probe_ofs(probe_filename.c_str());
    if (probe_ofs.good()) {
      probe_ofs.close();
      std::remove(probe_filename.c_str());
    } else {
      LOG(FATAL) << "Cannot write to snapshot prefix '"
          << param_.snapshot_prefix() << "'.  Make sure "
          << "that the directory exists and is writeable.";
    }
  }
}

template <typename Dtype>
string Solver<Dtype>::SnapshotFilename(const string extension, const string& prefix) {
  return param_.snapshot_prefix() + prefix + "_iter_" + caffe::format_int(iter_) + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto(const string& prefix) {
  string model_filename = SnapshotFilename(".caffemodel", prefix);
  LOG(INFO) << "Snapshotting to binary proto file " << model_filename;
  NetParameter net_param;
  net_->ToProto(&net_param, param_.snapshot_diff());
  WriteProtoToBinaryFile(net_param, model_filename);
  return model_filename;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToHDF5() {
  string model_filename = SnapshotFilename(".caffemodel.h5");
  LOG(INFO) << "Snapshotting to HDF5 file " << model_filename;
  net_->ToHDF5(model_filename, param_.snapshot_diff());
  return model_filename;
}

template <typename Dtype>
void Solver<Dtype>::Restore(const char* state_file, const bool& restore_prune_state) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename, restore_prune_state);
    if (restore_prune_state) {
      SetPruneState(APP<Dtype>::prune_state);
    }
  }
}

template <typename Dtype>
void Solver<Dtype>::UpdateSmoothedLoss(Dtype loss, int start_iter,
    int average_loss) {
  if (losses_.size() < average_loss) {
    losses_.push_back(loss);
    int size = losses_.size();
    smoothed_loss_ = (smoothed_loss_ * (size - 1) + loss) / size;
  } else {
    int idx = (iter_ - start_iter) % average_loss;
    smoothed_loss_ += (loss - losses_[idx]) / average_loss;
    losses_[idx] = loss;
  }
}

INSTANTIATE_CLASS(Solver);

}  // namespace caffe