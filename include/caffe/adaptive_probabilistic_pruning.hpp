#ifndef ADAPTIVE_PROBABILISTIC_PRUNING_HPP_
#define ADAPTIVE_PROBABILISTIC_PRUNING_HPP_

#include <string>
#include <vector>
#include <map>
#include <climits>
// #define ShowTimingLog 1

namespace caffe {
using namespace std;

template<typename Dtype>
class APP {
public:
     APP() {};
    ~APP() {};

    /// --------------------------------
    /// pass params from solver.prototxt to layer
    static string prune_method;
    static string prune_unit;
    static string prune_coremthd;
    static string prune_coremthd_;  // if prune_method == "Reg-L1_Col", then prune_unit = "Col", prune_coremthd = "Reg-L1", prune_coremthd_ = "Reg"
    static Dtype ratio_once_prune;
    static int prune_interval;
    static int losseval_interval;
    static int retrain_test_interval;
    static int clear_history_interval;
    static int prune_begin_iter;
    static int iter_size;
    static Dtype learning_rate;
    static Dtype score_decay;
    static Dtype AA;
    static Dtype base_prune_ratio_step;
    static vector<Dtype> prune_ratio_step;
    static Dtype kk;
    static Dtype kk2;
    static Dtype speedup;
    static Dtype compRatio;
    static bool IF_update_row_col;
    static bool IF_speedup_count_fc;
    static bool IF_compr_count_conv;
    static bool IF_eswpf;
    static vector<bool> cnt_loss_cross_borderline;
    static int cnt_acc_hit;
    static int cnt_acc_bad;
    static bool IF_acc_far_from_borderline;
    static vector<bool> IF_layer_far_from_borderline;
    static Dtype prune_threshold;
    static Dtype target_reg;
    static vector<Dtype> last_feasible_prune_ratio;
    static Dtype         last_prune_ratio_incre;
    static Dtype         accumulated_ave_incre_pr;
    static Dtype         prune_ratio_begin_ave;
    static int           last_feasible_prune_iter;
    static vector<Dtype> last_infeasible_prune_ratio;
    static Dtype         last_feasible_acc;
    static string model_prototxt;
    static int original_gpu_id;
    static int test_gpu_id;
    static Dtype accu_borderline;
    static Dtype loss_borderline;
    static vector<Dtype> retrain_test_acc1;
    static vector<Dtype> retrain_test_acc5;
    static string prune_state;
    static int prune_stage;
    static Dtype STANDARD_SPARSITY;
    static Dtype STANDARD_INCRE_PR;
    
    static int inner_iter;
    static int step_;
    static int num_;
    static long last_time; // used to calculate training speed
    static long first_time;
    static int  first_iter;
    static bool IF_scheme1_when_Reg_rank;
    static bool IF_current_target_achieved;
    static bool IF_speedup_achieved;
    static bool IF_compRatio_achieved;
    
    static map<string, int> layer_index;
    static int fc_layer_cnt;
    static int conv_layer_cnt;
    static vector<int> filter_spatial_size;
    static vector<int> group;
    static vector<int> priority;
    
    static vector<vector<int> > rows_to_prune;
    static vector<vector<int> > pruned_rows;
    static vector<Dtype> num_pruned_col;
    static vector<int>   num_pruned_row;
    static vector<int>   num_pruned_weight;
    static vector<bool>  IF_update_row_col_layer;
    static vector<vector<bool> > IF_row_pruned;
    static vector<vector<vector<bool> > > IF_col_pruned;
    static vector<vector<bool> > IF_weight_pruned;
    static vector<int> iter_prune_finished;
    static int stage_iter_prune_finished;
    static vector<Dtype> prune_ratio;
    static vector<Dtype> current_prune_ratio; // The prune_ratio for current pruning iteration in multi-step pruning.
    static vector<Dtype> pruned_ratio;
    static vector<Dtype> pruned_ratio_col;
    static vector<Dtype> pruned_ratio_row;
    static vector<Dtype> pruned_ratio_for_comparison;
    static vector<Dtype> GFLOPs;
    static vector<Dtype> num_param;
    
    static int num_log;
    static string mask_generate_mechanism;
    static int input_length; // for 3D CNN
    static bool simulate_5d;
    static int h_off;
    static int w_off;
    
    static int show_interval;
    static string show_layer;
    static int show_num_layer;
    static int show_num_weight;
    static vector<Dtype> when_snapshot;
}; 

    template<typename Dtype>  string  APP<Dtype>::prune_method    = "None"; /// initialized for caffe test, which has no solver but this info is still needed in layer.
    template<typename Dtype>  string  APP<Dtype>::prune_unit      = "None";
    template<typename Dtype>  string  APP<Dtype>::prune_coremthd  = "None";
    template<typename Dtype>  string  APP<Dtype>::prune_coremthd_ = "None";
    template<typename Dtype>  Dtype   APP<Dtype>::ratio_once_prune;
    template<typename Dtype>  int     APP<Dtype>::prune_interval;
    template<typename Dtype>  int     APP<Dtype>::losseval_interval;
    template<typename Dtype>  int     APP<Dtype>::retrain_test_interval;
    template<typename Dtype>  int     APP<Dtype>::clear_history_interval;
    template<typename Dtype>  int     APP<Dtype>::prune_begin_iter;
    template<typename Dtype>  int     APP<Dtype>::iter_size;
    template<typename Dtype>  Dtype   APP<Dtype>::learning_rate = 0;
    template<typename Dtype>  Dtype   APP<Dtype>::score_decay = 0;
    template<typename Dtype>  Dtype   APP<Dtype>::AA;
    template<typename Dtype>  Dtype   APP<Dtype>::base_prune_ratio_step = 0.02;
    template<typename Dtype>  vector<Dtype> APP<Dtype>::prune_ratio_step;
    template<typename Dtype>  Dtype   APP<Dtype>::kk;
    template<typename Dtype>  Dtype   APP<Dtype>::kk2;
    template<typename Dtype>  Dtype   APP<Dtype>::speedup;
    template<typename Dtype>  Dtype   APP<Dtype>::compRatio;
    template<typename Dtype>  bool    APP<Dtype>::IF_update_row_col;
    template<typename Dtype>  bool    APP<Dtype>::IF_speedup_count_fc;
    template<typename Dtype>  bool    APP<Dtype>::IF_compr_count_conv;
    template<typename Dtype>  bool    APP<Dtype>::IF_eswpf;
    template<typename Dtype>  vector<bool> APP<Dtype>::cnt_loss_cross_borderline;
    template<typename Dtype>  int     APP<Dtype>::cnt_acc_hit = 0;
    template<typename Dtype>  int     APP<Dtype>::cnt_acc_bad = 0;
    template<typename Dtype>  bool    APP<Dtype>::IF_acc_far_from_borderline = true;
    template<typename Dtype>  vector<bool> APP<Dtype>::IF_layer_far_from_borderline;
    template<typename Dtype>  Dtype APP<Dtype>::prune_threshold;
    template<typename Dtype>  Dtype APP<Dtype>::target_reg;
    template<typename Dtype>  vector<Dtype> APP<Dtype>::last_feasible_prune_ratio;
    template<typename Dtype>  Dtype         APP<Dtype>::last_prune_ratio_incre = 0;
    template<typename Dtype>  Dtype         APP<Dtype>::accumulated_ave_incre_pr = 0;
    template<typename Dtype>  Dtype         APP<Dtype>::prune_ratio_begin_ave = 0.2; // the average sparsity for the first pruning stage
    template<typename Dtype>  int           APP<Dtype>::last_feasible_prune_iter = -1;
    template<typename Dtype>  vector<Dtype> APP<Dtype>::last_infeasible_prune_ratio;
    template<typename Dtype>  Dtype         APP<Dtype>::last_feasible_acc = 0;
    template<typename Dtype>  string APP<Dtype>::model_prototxt;
    template<typename Dtype>  int APP<Dtype>::original_gpu_id;
    template<typename Dtype>  int APP<Dtype>::test_gpu_id = -1;
    template<typename Dtype>  Dtype APP<Dtype>::accu_borderline;
    template<typename Dtype>  Dtype APP<Dtype>::loss_borderline;
    template<typename Dtype>  vector<Dtype> APP<Dtype>::retrain_test_acc1;
    template<typename Dtype>  vector<Dtype> APP<Dtype>::retrain_test_acc5;
    template<typename Dtype>  string APP<Dtype>::prune_state = "prune";
    template<typename Dtype>  int    APP<Dtype>::prune_stage = 1;
    template<typename Dtype>  Dtype APP<Dtype>::STANDARD_SPARSITY = 0.5; // If this changes, the prune_ratio_step should change accordingly.
    template<typename Dtype>  Dtype APP<Dtype>::STANDARD_INCRE_PR = 0.05;

    // 1.2 Info shared between solver and layer, initailized here
    template<typename Dtype>  int   APP<Dtype>::inner_iter = 0;
    template<typename Dtype>  int   APP<Dtype>::step_ = 1;
    template<typename Dtype>  int   APP<Dtype>::num_; // batch_size
    template<typename Dtype>  long  APP<Dtype>::last_time  = 0;
    template<typename Dtype>  long  APP<Dtype>::first_time = 0;
    template<typename Dtype>  int   APP<Dtype>::first_iter = 0;
    template<typename Dtype>  bool  APP<Dtype>::IF_scheme1_when_Reg_rank;
    template<typename Dtype>  bool  APP<Dtype>::IF_current_target_achieved = false; /// if all layer prune finished in the current pruning iteration
    template<typename Dtype>  bool  APP<Dtype>::IF_speedup_achieved   = false;
    template<typename Dtype>  bool  APP<Dtype>::IF_compRatio_achieved = false;

    // 2.1 Info shared among layers
    template<typename Dtype>  map<string, int>  APP<Dtype>::layer_index;
    template<typename Dtype>  int  APP<Dtype>::fc_layer_cnt   = 0;
    template<typename Dtype>  int  APP<Dtype>::conv_layer_cnt = 0;
    template<typename Dtype>  vector<int>  APP<Dtype>::filter_spatial_size;
    template<typename Dtype>  vector<int>  APP<Dtype>::group;
    template<typename Dtype>  vector<int>  APP<Dtype>::priority;
    
    // 2.2 Pruning state (key)
    template<typename Dtype>  vector<vector<int> > APP<Dtype>::rows_to_prune;
    template<typename Dtype>  vector<vector<int> > APP<Dtype>::pruned_rows;
    template<typename Dtype>  vector<bool>   APP<Dtype>::IF_update_row_col_layer;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::num_pruned_col;
    template<typename Dtype>  vector<int>    APP<Dtype>::num_pruned_row;
    template<typename Dtype>  vector<int>    APP<Dtype>::num_pruned_weight;
    template<typename Dtype>  vector<vector<bool> >   APP<Dtype>::IF_row_pruned;
    template<typename Dtype>  vector<vector<vector<bool> > >  APP<Dtype>::IF_col_pruned;
    template<typename Dtype>  vector<vector<bool> >   APP<Dtype>::IF_weight_pruned;
    template<typename Dtype>  vector<int>    APP<Dtype>::iter_prune_finished;
    template<typename Dtype>  int            APP<Dtype>::stage_iter_prune_finished = INT_MAX;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::prune_ratio;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::current_prune_ratio;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_col;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_row;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_for_comparison;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::GFLOPs;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::num_param;
    
    // 3. Logging
    template<typename Dtype>  int     APP<Dtype>::num_log = 0; // > 0: logging is true
    template<typename Dtype>  string  APP<Dtype>::mask_generate_mechanism = "group-wise";
    template<typename Dtype>  int     APP<Dtype>::input_length = 16; // TODO(mingsuntse): to use solver.prototxt or autoset.
    template<typename Dtype>  bool    APP<Dtype>::simulate_5d = false;
    template<typename Dtype>  int     APP<Dtype>::h_off = 0; // used in data transformation
    template<typename Dtype>  int     APP<Dtype>::w_off = 0;
    template<typename Dtype>  int APP<Dtype>::show_interval = 1; // the interval to print pruning progress log
    template<typename Dtype>  string APP<Dtype>::show_layer = "1111"; // '1' means to print the weights of the layer with the index
    template<typename Dtype>  int APP<Dtype>::show_num_layer = 100; // work with show_interval, how many layers get printed
    template<typename Dtype>  int APP<Dtype>::show_num_weight = 20; // work with show_layer, how many weights get printed
    template<typename Dtype>  vector<Dtype> APP<Dtype>::when_snapshot;
}

#endif
