#include <string>
#include <vector>

#include "caffe/sgd_solvers.hpp"
#include "caffe/util/hdf5.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/upgrade_proto.hpp"
#include "caffe/util/math_functions.hpp"
#include <numeric>
#include "caffe/adaptive_probabilistic_pruning.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>

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
  for (int i = 0; i < net_params.size(); ++i) {
    const vector<int>& shape = net_params[i]->shape();
    history_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    update_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));
    temp_.push_back(shared_ptr<Blob<Dtype> >(new Blob<Dtype>(shape)));    
  }
  
  // initialize history reg, WANGHUAN
  for (int i = 0; i < net_params.size(); ++i) {
      const vector<int>& shape = net_params[i]->shape();  
      const int num_col = net_params[i]->count() / shape[0];
      if (shape.size() != 4) continue; // do not reg fc layers and biases
      vector<Dtype> tmp(num_col, 0);
      history_reg_.push_back(tmp);
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
  CHECK(Caffe::root_solver());
  Dtype rate = GetLearningRate();
  if (this->param_.display() && this->iter_ % this->param_.display() == 0) {
    LOG(INFO) << "Iteration " << this->iter_ << ", lr = " << rate;
  }
  ClipGradients();
  ClearHistory(); // WANGHUAN
  for (int param_id = 0; param_id < this->net_->learnable_params().size();
       ++param_id) {

    // added by WANGHUAN
    // std::cout << "learnable_params().size(): " << this->net_->learnable_params().size() << std::endl;
    // std::cout << this->net_->name() << std::endl;

    Normalize(param_id);
    Regularize(param_id);
    ComputeUpdateValue(param_id, rate);
  }
  this->net_->Update();
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
// typedef std::pair<Dtype, int> mypair;
bool SGDSolver<Dtype>::Comparator(const std::pair<Dtype, int>& left, const std::pair<Dtype, int>& right) { 
    return (left.first < right.first); 
}    

template <typename Dtype>
void SGDSolver<Dtype>::Regularize(int param_id) {    
  const vector<Blob<Dtype>*>& net_params = this->net_->learnable_params();
  const vector<float>& net_params_weight_decay =
      this->net_->params_weight_decay();
  Dtype weight_decay = this->param_.weight_decay();
  string regularization_type = this->param_.regularization_type();

  // ------------------------------------------------
  // Decrease-Weight-Decay Mode, WANGHUAN
  Dtype current_wd = weight_decay; // default
  if (this->param_.dwd_mode() != "None") {
      CHECK_GE(this->param_.wd_end(), 0) << "Error: wd_end must be in [0, 1]";
      CHECK_LE(this->param_.wd_end(), 1) << "Error: wd_end must be in [0, 1]";
      
      const int begin = this->param_.dwd_begin_iter();
      // const int end   = this->param_.dwd_end_iter();
      // cout << "begin: " << this->param_.dwd_begin_iter() << "  " << end << endl; //???
      
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

          } else if (this->param_.dwd_mode() == "adaptive") {
            const int num_pruned = *std::max_element(APP::num_pruned_col.begin(), APP::num_pruned_col.end()); // 9 is the size, TODO: replace it using vector
            const int num_to_prune = APP::max_num_column_to_prune;
            current_wd = weight_decay * (1 - (1 - this->param_.wd_end()) / num_to_prune * num_pruned);
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

        // print cpu_diff after L1 reg
        for (int i = 0; i < 10; i++) {
          std::cout << *(net_params[param_id]->cpu_diff() + i) << " ";
        }
        std::cout << std::endl;
    
      } else if (regularization_type == "StrColReg") {
          const Dtype *weight = net_params[param_id]->cpu_data();
          const int count = net_params[param_id]->count();
          vector<Dtype> local_decay_StrColReg(count);
          const int num_filter = net_params[param_id]->shape()[0];
          const int num_col = count / num_filter;
          vector<Dtype> abscol(num_filter);
          vector<Dtype> absvar_col(num_col);
          vector<Dtype> absmean_col(num_col);

          // compute abs variance for each column
          for (int j = 0; j < num_col; ++j) {             
              absmean_col[j] = 0;
              for (int i = 0; i < num_filter; ++i) {
                  abscol[i] = fabs(weight[i * num_col + j]);
                  absmean_col[j] += abscol[i];
              }
              absmean_col[j] /= num_filter;

              absvar_col[j] = 0;
              for (int i = 0; i < num_filter; ++i) {
                  absvar_col[j] += (abscol[i] - absmean_col[j]) * (abscol[i] - absmean_col[j]);
              }
              absvar_col[j] /= num_filter;
          }

          // Penalty: the further weight is away from its column abs mean, the harsher its penalty is.
          for (int i = 0; i < num_filter; ++i) {
              for (int j = 0; j < num_col; ++j) {
                  Dtype a1 = fabs(weight[i * num_col + j]);
                  Dtype a2 = absmean_col[j];
                  if (a1 < 1e-30 || a1 > 1e30 || a2 < 1e-30 || a2 > 1e30) {
                      local_decay_StrColReg[i * num_col + j] = 0;
                  } else {
                      local_decay_StrColReg[i * num_col + j] = a1 - a2; // var: the mean deviation, if some weight deviates further than var, its penalty should be harsher.
                  }
                  // std::cout << local_decay_StrColReg[i * num_col + j] << std::endl;
              }
          }
          

          // update diff
          int conv_index = 3;      
          local_decay = net_params_weight_decay[param_id] * (weight_decay -  APP::num_pruned_col[conv_index] * 1.5e-6);
          for (int i = 0; i < count; ++i) {     
              int sign = 0;
              if (weight[i] >0) sign = 1;
              else if (weight[i] < 0) sign = -1;
              net_params[param_id]->mutable_cpu_diff()[i] += local_decay * (weight[i] + sign * local_decay_StrColReg[i]);
          }
          
      } else if (regularization_type == "SSL") {
        // add weight decay, weight decay still used
        caffe_gpu_axpy(net_params[param_id]->count(),
                       local_decay,
                       net_params[param_id]->gpu_data(),
                       net_params[param_id]->mutable_gpu_diff());
                       
        const Dtype* weight = net_params[param_id]->cpu_data();
        const Dtype col_reg = APP::col_reg;
        const int count = net_params[param_id]->count();
        const int num_filter = net_params[param_id]->shape()[0];
        const int num_col = count / num_filter;

        const vector<int>& shape = net_params[param_id]->shape();
        if (shape.size() != 4) { return; } // do not reg biases and fc layer
        if (APP::num_pruned_col[param_id / 2] >= num_col * APP::prune_ratio[param_id / 2]) { return; }
      
        Dtype* sqrted_energy = (Dtype*) malloc (sizeof(Dtype*) * count); // demoninator of SSL reg
        for (int j = 0; j < num_col; ++j) {
          Dtype sum = 0;
          for (int i = 0; i < num_filter; ++i) { sum += weight[i * num_col + j] * weight[i * num_col + j]; } 
          for (int i = 0; i < num_filter; ++i) { sqrted_energy[i * num_col + j] = sum < 1e-30 ? 1 : std::sqrt(sum); } // If some column is pruned, its "sum" will be very small, "sqrt(sum)" causing nan. 
        }
        const Dtype* sqrted_energy_const =  sqrted_energy;
          
        // add SSL reg
        Dtype* scaled_weight = (Dtype*) malloc (sizeof(Dtype*) * count);
        caffe_div(count, 
                  net_params[param_id]->cpu_data(), 
                  sqrted_energy_const, 
                  scaled_weight); // degug here
        const Dtype* scaled_weight_const = scaled_weight;
        
        caffe_axpy(count, 
                   col_reg, 
                   scaled_weight_const, 
                   net_params[param_id]->mutable_cpu_diff()); // degug here
        
        free(scaled_weight);
        free(sqrted_energy);
          
      } else if (regularization_type == "SelectiveReg") {        
        // add weight decay, weight decay still used
        caffe_gpu_axpy(net_params[param_id]->count(),
                       local_decay,
                       net_params[param_id]->gpu_data(),
                       net_params[param_id]->mutable_gpu_diff());    
        
        
        if (this->iter_ < APP::when_to_col_reg) { return; }// if iter < when_to_col_reg, do not apply column reg
        const vector<int>& shape = net_params[param_id]->shape();
        if (shape.size() != 4) { return; } // do not reg biases and fc layers
        
        const Dtype* weight = net_params[param_id]->cpu_data();
        const int count = net_params[param_id]->count();
        const int num_filter = net_params[param_id]->shape()[0];
        const int num_col = count / num_filter;
        const int num_col_to_prune = int(num_col * (APP::prune_ratio[param_id / 2] + 0.05)); // 0.05 is a margin 
        if (APP::num_pruned_col[param_id / 2] >= num_col_to_prune) { return; }
        
        // calculate column score as well as denominator of SSL reg    
        typedef std::pair<Dtype, int> mypair;
        vector<mypair> col_score(num_col);     // score of every column
        Dtype* sqrted_energy = (Dtype*) malloc (sizeof(Dtype*) * count); // demoninator of SSL reg
        for (int j = 0; j < num_col; ++j) {
            col_score[j].second = j;
            col_score[j].first = 0;
            Dtype sum = 0;
            for (int i = 0; i < num_filter; ++i) {
                sum += weight[i * num_col + j] * weight[i * num_col + j];
                col_score[j].first += fabs(weight[i * num_col +j]);
            }
            for (int i = 0; i < num_filter; ++i) {            
                sqrted_energy[i * num_col + j] = sum < 1e-30 ? 1 : std::sqrt(sum); // If some column is pruned, its "sum" will be very small, "sqrt(sum)" causing nan.
            }          
        }
        const Dtype* sqrted_energy_const =  sqrted_energy;
        
        // check sort. without Comparator, seems it works fine. quite odd...
        //for (int i = 0; i < 15; ++i) {
        //    std::cout << "before sort, col_score: " << col_score[i].first << " " << col_score[i].second  << std::endl;
        //}    
        sort(col_score.begin(), col_score.end()); // in ascending order
        //for (int i = 0; i < 15; ++i) {
            //std::cout << "after sort, col_score: " << col_score[i].first << " " << col_score[i].second << std::endl;
        //}
        
        // check order
        /*
        ostringstream stream;
        stream << param_id << "_order.txt";
        const char* dd = stream.str().c_str();
        ofstream col_score_order(dd, ofstream::app); // dd must be char*
        if (!col_score_order.is_open()) { 
            cout << "file open failed: param_id = " << param_id << endl; 
        } else { 
            for (int j = 0; j < num_col; ++j) {
                col_score_order <<     col_score[j].second << " ";
            }
            col_score_order << "\n";
            col_score_order.close();
        }
        */
        
        // compute reg multiplier for those “bad” columns, "good" columns are spared with zero reg.
        const Dtype AA = 1e-6; // the probability increment at rank 0
        const Dtype aa = 1e-8; // the probability increment at rank num_col_to_prune
        const Dtype alpha = -log(aa/AA) / num_col_to_prune;  
        vector<Dtype> col_reg(count); // reg multiplier for every weight  
        for (int j = 0; j < num_col; ++j) {  // j: the jth column according to col score ranking (排名第j的列)
            const int col_of_rank_j = col_score[j].second;         
            
            //selective reg
            history_reg_[param_id/2][col_of_rank_j] = j < num_col_to_prune 
                                                        ? history_reg_[param_id/2][col_of_rank_j] + AA * exp(-j * alpha)
                                                        : 0;
            for (int i = 0; i < num_filter; ++i) {
                col_reg[i * num_col + col_of_rank_j] = history_reg_[param_id/2][col_of_rank_j];
                
                // reg scheme 1
                // col_reg[i * num_col + col_of_rank_j] = history_reg_[param_id/2][col_of_rank_j] * reg_decay 
                                                            // + col_reg_base - col_reg_base / num_col_to_prune * jj;
                
                // reg scheme 2        
                // col_reg[i * num_col + col_of_rank_j] = j < num_col_to_prune ? beta * exp(-j * alpha) : 0; // Only reg the worst num_col_to_prune columns                    
            }     
        
        }          
        
        // check reg
        std::cout << "param_id: " << param_id << std::endl;
        std::cout << "selective reg for columns: " << std::endl;
        for (int j = 0; j < 10; ++j) {
            std::cout << "#" << j+1 << ": " << col_reg[j] << std::endl;
        }
        
        // add SelectiveReg
        Dtype* scaled_weight = (Dtype*) malloc (sizeof(Dtype*) * count);
        caffe_div(count, 
                  net_params[param_id]->cpu_data(), 
                  sqrted_energy_const, 
                  scaled_weight); // degug here
                
        for (int i = 0; i < count; ++i) {
            net_params[param_id]->mutable_cpu_diff()[i] += col_reg[i] * scaled_weight[i];
        }
            
        free(scaled_weight);
        free(sqrted_energy);
        
      } else if (regularization_type == "SSL+Diff") {
        // wanghuan, reg decays with the number of pruned column
        int conv_index = 3;
        int num_pruned_column = APP::num_pruned_col[conv_index];
        local_decay = net_params_weight_decay[param_id] * (weight_decay -  num_pruned_column * 6.5e-6);
        const Dtype COL_REG = (num_pruned_column >= 800 * APP::prune_ratio[conv_index]) ? 0 : APP::col_reg; // - num_pruned_column * 4.83e-6;
        const Dtype DIFF_REG = (num_pruned_column >= 800 * APP::prune_ratio[conv_index]) ? 0 : APP::diff_reg;
        
        // add weight decay, weight decay still used
        caffe_gpu_axpy(net_params[param_id]->count(),
                       local_decay,
                       net_params[param_id]->gpu_data(),
                       net_params[param_id]->mutable_gpu_diff());    
         
        // **** TODO **** 
        // without this "if", there will be “nan” error, why?
        if (param_id % 2 || param_id == 6) { return; } // do not reg biases and fc layer
        
        const Dtype* weight = net_params[param_id]->cpu_data();    
        const Dtype* weight_diff = net_params[param_id]->cpu_diff();        
        const int count = net_params[param_id]->count();
        const int num_filter = net_params[param_id]->shape()[0];
        const int num_col = count / num_filter;
        
        // add SSL reg
        Dtype* sqrted_energy =  new Dtype[count]; // demoninator of SSL reg        
        for (int j = 0; j < num_col; ++j) {
          Dtype sum1 = 0;
          for (int i = 0; i < num_filter; ++i) {
            sum1 += weight[i * num_col + j] * weight[i * num_col + j];
          } 
          for (int i = 0; i < num_filter; ++i) {            
            sqrted_energy[i * num_col + j] = sum1 < 1e-30 ? 1 : std::sqrt(sum1); // If some column is pruned, its "sum" will be very small, "sqrt(sum)" causing nan.
          }          
        }
        
        Dtype* scaled_weight = new Dtype[count];
        caffe_div(count, 
                  net_params[param_id]->cpu_data(), 
                  (const Dtype*) sqrted_energy, 
                  scaled_weight);         
        caffe_axpy(count, 
                   COL_REG, 
                   (const Dtype*) scaled_weight, 
                   net_params[param_id]->mutable_cpu_diff()); 
        
        delete[] scaled_weight;
        delete[] sqrted_energy;
        
        
        // add Diff reg
        Dtype* scaled_diff = new Dtype[count];
        const Dtype* second_diff = net_params[param_id]->mutable_cpu_second_diff;

        // ------------
        for (int i = 0; i < 20; ++i) {
            std::cout << second_diff[i] << " ";
        }
        std::cout << std::endl;
        
        caffe_mul(count,
                  second_diff, 
                  weight_diff,
                  scaled_diff);  // second_diff * diff         
        const Dtype* scaled_diff_2 = scaled_diff; 

        Dtype* sqrted_diff = new Dtype[count]; // demoninator of Diff reg
        for (int j = 0; j < num_col; ++j) {
          Dtype sum2 = 0;
          for (int i = 0; i < num_filter; ++i) {
            sum2 += weight_diff[i * num_col + j] * weight_diff[i * num_col + j];
          } 
          for (int i = 0; i < num_filter; ++i) {            
              sqrted_diff[i * num_col + j] = sum2 < 1e-30 ? 1 : std::sqrt(sum2);
          }          
        }
        //here is the problem
        caffe_div(count, 
                  scaled_diff_2, 
                  (const Dtype*) sqrted_diff, 
                  scaled_diff); 
    
    
        // 修改一下看是不是     -----------------------------        
        if (count == 25600) {
            std::cout << "\nscaled diff " << std::endl;
            for (int i = 0; i < 20; ++i) {
                std::cout << scaled_diff[i] << " ";
            }
            std::cout << std::endl;
        }
        // -----------------------------------------------
        
        caffe_axpy(count, 
                   DIFF_REG, 
                   (const Dtype*) second_diff, 
                   net_params[param_id]->mutable_cpu_diff()); 
        
        delete[] scaled_diff;
        delete[] sqrted_diff;
        
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
void SGDSolver<Dtype>::ClearHistory() {
    const vector<shared_ptr<Layer<Dtype> > >& layers = this->net_->layers();
    int param_id = 0;
    for (int i = 0; i < layers.size(); ++i) {
    /// As long as layer i has masks, its history_ should be cleared. But only clear history_ of weights, since we only have masks for weights.
    /// So the key is to relate layer i with corresponding param_id.
        const int count = layers[i]->masks_.size();
        
        if (count) { 
            while (history_[param_id]->count() != count) { 
                ++ param_id; /// jump over biases
            }
            Dtype* tmp = new Dtype[count]; /// TODEBUG: Why cannot use bool?
            for (int k = 0; k < count; ++k) {
                tmp[k] = layers[i]->masks_[k];
            }
            caffe_mul(count, 
                      (const Dtype*) tmp, 
                      history_[param_id]->cpu_data(), 
                      history_[param_id]->mutable_cpu_data());
            delete[] tmp;
            ++ param_id;
        }
    }
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
  for (int i = 0; i < history_.size(); ++i) {
    // Add history
    BlobProto* history_blob = state.add_history();
    history_[i]->ToProto(history_blob);
  }
  string snapshot_filename = Solver<Dtype>::SnapshotFilename(".solverstate");
  LOG(INFO)
    << "Snapshotting solver state to binary proto file " << snapshot_filename;
  WriteProtoToBinaryFile(state, snapshot_filename.c_str());
}

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
  if (state.has_learned_net()) {
    NetParameter net_param;
    ReadNetParamsFromBinaryFileOrDie(state.learned_net().c_str(), &net_param);
    this->net_->CopyTrainedLayersFrom(net_param);
  }
  this->current_step_ = state.current_step();
  CHECK_EQ(state.history_size(), history_.size())
      << "Incorrect length of history blobs.";
  LOG(INFO) << "SGDSolver: restoring history";
  for (int i = 0; i < history_.size(); ++i) {
    history_[i]->FromProto(state.history(i));
  }
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
