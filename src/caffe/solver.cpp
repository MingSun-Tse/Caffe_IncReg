#include <cstdio>

#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#include <ctime>

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
    << std::endl << param.DebugString(); // 这里打印出solver_params
  param_ = param;
  
  // ------------------------------------------
  // WANGHUAN, copy prune params
  APP::prune_method = param_.prune_method();
  APP::criteria = param_.criteria();
  APP::num_once_prune = param_.num_once_prune();
  APP::prune_interval = param_.prune_interval();
  APP::rgamma = 30;   //param_.rgamma();
  APP::rpower = -1.3; //param_.rpower();
  APP::cgamma = 70;   //param_.cgamma();
  APP::cpower = -1.2; //param_.cpower(); 
  APP::prune_begin_iter = param_.prune_begin_iter();
  APP::iter_size = param_.iter_size();
  APP::AA = param_.aa();
  APP::target_reg = param_.target_reg(); //param_.aa();
  APP::kk = 0.25; //param_.kk(); 
  APP::speedup = param_.speedup();
  APP::IF_update_row_col = param.if_update_row_col();
  APP::IF_eswpf = param_.if_eswpf(); /// if early stop when prune finished
  APP::prune_threshold = param_.prune_threshold();
  APP::num_iter_reg = 10000; // param_.num_iter_reg();
  
  // APP::score_decay = param_.score_decay();
  APP::snapshot_prefix = param_.snapshot_prefix();
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
    ReadNetParamsFromTextFileOrDie(param_.net(), &net_param); // 读入网络结构
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

  while (iter_ < stop_iter) {
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
    
    // 分发参数
    // std::cout << "call_backs_.size(): " << callbacks_.size() << std::endl; /// WANGHUAN
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_start();
    }
    
    const bool display = param_.display() && iter_ % param_.display() == 0;
    net_->set_debug_info(display && param_.debug_info());
    // accumulate the loss and gradient
    Dtype loss = 0;

    /// ----------------------------------------------------------------------
    // Before another forward, judge whether prune could be stopped
    if (APP::IF_alpf && APP::IF_eswpf) {
        cout << "all layer prune finished: " << iter_ << " -- early stopped." << endl;
        requested_early_exit_ = true;
        break;
    }

    // GFLOPs, since it measures the speedup of the whole net, so put it here rather than in layer.
    Dtype GFLOPs_left   = 0;
    Dtype GFLOPs_origin = 0;
    for (int i = 0; i < APP::layer_cnt; ++i) {
        GFLOPs_left   += APP::GFLOPs[i] * (1 - APP::pruned_ratio[i]);
        GFLOPs_origin += APP::GFLOPs[i];
    }
    APP::IF_speedup_achieved = (GFLOPs_origin / GFLOPs_left >= APP::speedup);
    
    APP::step_ = iter_ + 1;
    cout << "\n**** Step " << APP::step_ << ": " << GFLOPs_origin / GFLOPs_left << "/" << APP::speedup << " ****" << endl;
    cout << "Total GFLOPs_origin: " << GFLOPs_origin << endl;
    /// ----------------------------------------------------------------------
    
    APP::inner_iter = 0;
    for (int i = 0; i < param_.iter_size(); ++i) {
      loss += net_->ForwardBackward();
      ++ APP::inner_iter; /// WANGHUAN
    }

    loss /= param_.iter_size();
    // average the loss across iterations for smoothed reporting
    UpdateSmoothedLoss(loss, start_iter, average_loss); 
    
    /// ----------------------------------------------------------------------
    /// WANGHUAN, used for Adaptive SPP
    /// APP::Delta_loss_history =  APP::Delta_loss_history * APP::loss_decay + (smoothed_loss_ - APP::loss);
    /// APP::Delta_loss_history =  smoothed_loss_ - APP::loss;
    APP::learning_speed = APP::loss - smoothed_loss_;
    APP::loss = smoothed_loss_;
    cout << "learning_speed: " << APP::learning_speed << endl;
    /// ----------------------------------------------------------------------

    if (display) {
      LOG_IF(INFO, Caffe::root_solver()) << "Iteration " << iter_
          << ", smoothed loss = " << smoothed_loss_; 
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
    
    // 归约梯度，在Update之前，而在Update中有`Iteration, lr`的logging，合理解释了log中就只有一处这个logging
    for (int i = 0; i < callbacks_.size(); ++i) {
      callbacks_[i]->on_gradients_ready();
    }
    
    ApplyUpdate(); /// Virtual Function

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
    }
    if (SolverAction::STOP == request) {
      requested_early_exit_ = true;
      // Break out of training loop.
      break;
    }
  }
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
  

  // After restore, calculate GFLOPs and determine whether the prune finished
  Dtype GFLOPs_left   = 0;
  Dtype GFLOPs_origin = 0;
  for (int i = 0; i < APP::layer_cnt; ++i) {
      GFLOPs_left   += APP::GFLOPs[i] * (1 - APP::pruned_ratio[i]);
      GFLOPs_origin += APP::GFLOPs[i];
  }
  APP::IF_speedup_achieved = (GFLOPs_origin / GFLOPs_left >= APP::speedup);
  if (APP::IF_speedup_achieved) {
    for (int i = 0; i < APP::layer_index.size(); ++i) {
        APP::iter_prune_finished[i] = -1; 
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
    if (APP::num_log) { Logshot(); }
    if (APP::prune_method.substr(0, 2) == "PP") { PruneStateShot(); }
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
  if (APP::num_log) { Logshot(); }
  if (APP::prune_method.substr(0, 2) == "PP") { PruneStateShot(); }
  LOG(INFO) << "Optimization Done.";
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
  CHECK(Caffe::root_solver());  
  
  LOG(INFO) << "Iteration " << iter_
            << ", Testing net (#" << test_net_id << ")";
  CHECK_NOTNULL(test_nets_[test_net_id].get())->
      ShareTrainedLayersWith(net_.get());
  vector<Dtype> test_score;
  vector<int> test_score_output_id;
  const shared_ptr<Net<Dtype> >& test_net = test_nets_[test_net_id];
  Dtype loss = 0;
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
  }
}

template <typename Dtype>
void Solver<Dtype>::Snapshot() {
  CHECK(Caffe::root_solver());
  string model_filename;
  switch (param_.snapshot_format()) {
  case caffe::SolverParameter_SnapshotFormat_BINARYPROTO:
    model_filename = SnapshotToBinaryProto();
    break;
  case caffe::SolverParameter_SnapshotFormat_HDF5:
    model_filename = SnapshotToHDF5();
    break;
  default:
    LOG(FATAL) << "Unsupported snapshot format.";
  }

  SnapshotSolverState(model_filename);
}

template <typename Dtype>
void Solver<Dtype>::PruneStateShot() {
    map<string, int>::iterator it_m;
    for (it_m = APP::layer_index.begin(); it_m != APP::layer_index.end(); ++it_m) {
        const char* outfile = (param_.snapshot_prefix() + "prob_snapshot/prob_" + it_m->first + ".txt").c_str();
        if (!access(outfile, 0)) { /// outfile has already existed
            remove(outfile);
        } 
        ofstream prob(outfile, ofstream::app);
        if (!prob.is_open()) {
            cout << "Error: opening prob file failed: " << prob << endl; 
        } else {
            prob << iter_ << "\n"; 
            vector<float> pr = APP::history_prob[it_m->second];
            vector<float>::iterator it;
            for (it = pr.begin(); it != pr.end(); ++it) {
                prob << *it << " ";
            }
        }    
    }
    cout << "Save prune prob done!" << endl;
}

template <typename Dtype>
void Solver<Dtype>::Logshot() {
    const time_t t = time(NULL);
    struct tm* ctime = localtime(&t);
    ostringstream TIME;
    TIME << 1900 + ctime->tm_year;
    if (ctime->tm_mon < 9) { TIME << 0; }     
    TIME << 1 + ctime->tm_mon
         << ctime->tm_mday
         << "-"
         << ctime->tm_hour
         << ctime->tm_min;
    
    const string i_tmp = param_.snapshot_prefix() + TIME.str() + "_log_index.txt";
    const string w_tmp = param_.snapshot_prefix() + TIME.str() + "_log_weight.txt";
    const string d_tmp = param_.snapshot_prefix() + TIME.str() + "_log_diff.txt";
    const char* ii = i_tmp.c_str(); 
    const char* ww = w_tmp.c_str(); 
    const char* dd = d_tmp.c_str(); 
    ofstream log_i(ii, ofstream::app);
    ofstream log_w(ww, ofstream::app); 
    ofstream log_d(dd, ofstream::app);
    
    vector<vector<vector<float> > >::iterator it_l; /// it_layer
    vector<vector<float> >::iterator it_w; /// it_weight
    vector<float>::iterator it_i; /// it_iter
    vector<vector<int> >::iterator it_il;
    vector<int>::iterator it_iw;

    if (!log_i.is_open()) { 
        cout << "Error: opening file failed: " << ii << endl; 
    } else {
        for (it_il = APP::log_index.begin(); it_il != APP::log_index.end(); ++it_il) {
            for (it_iw = (*it_il).begin(); it_iw != (*it_il).end(); ++it_iw) {
                log_i << *it_iw << " ";
            }
            log_i << "\n";
        }
    }
                
    if (!log_w.is_open()) { 
        cout << "Error: opening file failed: " << ww << endl; 
    } else {
        for (it_l = APP::log_weight.begin(); it_l != APP::log_weight.end(); ++it_l) {
            for (it_w = (*it_l).begin(); it_w != (*it_l).end(); ++it_w) {
                for (it_i = (*it_w).begin(); it_i != (*it_w).end(); ++it_i) {
                    log_w << *it_i << " ";
                }
                log_w << "\n";
            }
            log_w << "\n";
        }
    }

    if (!log_d.is_open()) { 
        cout << "Error: opening file failed: " << dd << endl; 
    } else {
        for (it_l = APP::log_diff.begin(); it_l != APP::log_diff.end(); ++it_l) {
            for (it_w = (*it_l).begin(); it_w != (*it_l).end(); ++it_w) {
                for (it_i = (*it_w).begin(); it_i != (*it_w).end(); ++it_i) {
                    log_d << *it_i << " ";
                }
                log_d << "\n";
            }
            log_d << "\n";
        }
    }

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
string Solver<Dtype>::SnapshotFilename(const string extension) {
  return param_.snapshot_prefix() + "_iter_" + caffe::format_int(iter_)
    + extension;
}

template <typename Dtype>
string Solver<Dtype>::SnapshotToBinaryProto() {
  string model_filename = SnapshotFilename(".caffemodel");
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
void Solver<Dtype>::Restore(const char* state_file) {
  CHECK(Caffe::root_solver());
  string state_filename(state_file);
  if (state_filename.size() >= 3 &&
      state_filename.compare(state_filename.size() - 3, 3, ".h5") == 0) {
    RestoreSolverStateFromHDF5(state_filename);
  } else {
    RestoreSolverStateFromBinaryProto(state_filename);
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
