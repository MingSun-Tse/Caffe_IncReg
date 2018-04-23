#ifndef ADAPTIVE_PROBABILISTIC_PRUNING_HPP_
#define ADAPTIVE_PROBABILISTIC_PRUNING_HPP_

#include <string>
#include <vector>
#include <map>
#define NUM_OF_WEIGHT_BUCKET 2
#define RATIO 0.5


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
    static string criteria; 
    static int num_once_prune;
    static int prune_interval;
    static int clear_history_interval;
    static Dtype rgamma;
    static Dtype rpower;
    static Dtype cgamma;
    static Dtype cpower;
    static int prune_begin_iter;
    static int iter_size;
    static Dtype score_decay;
    static Dtype AA;
    static Dtype kk;
    static Dtype kk2;
    static Dtype speedup;
    static Dtype compRatio;
    static bool IF_update_row_col;
    static bool IF_speedup_count_fc;
    static bool IF_compr_count_conv;
    static bool IF_eswpf;
    static Dtype prune_threshold;
    static Dtype target_reg;
    static int num_iter_reg;
    static int reg_cushion_iter; // In the beginning of reg, improve the reg little-by-little to mitigate the side-effect. `reg_cushion_iter` is the iter of this mitigation perioid
    static Dtype hrank_momentum;
    
    
    static int inner_iter;
    static int step_;
    static long last_time; // used to calculate training speed
    static long first_time;
    static int  first_iter;
    static bool IF_alpf;
    static bool IF_scheme1_when_Reg_rank;
    static bool IF_speedup_achieved;
    static bool IF_compRatio_achieved;
    
    
    static map<string, int> layer_index;
    static int fc_layer_cnt;
    static int conv_layer_cnt;
    static vector<int> filter_area;
    static vector<int> group;
    static vector<int> priority;
    
    
    
    static vector<Dtype> num_pruned_col;
    static vector<int>   num_pruned_row;
    static vector<int>   num_pruned_weight;
    static vector<int>   pruned_rows;
    static vector<bool>  IF_update_row_col_layer;
    static vector<vector<Dtype> > masks;
    static vector<vector<bool> > IF_row_pruned;
    static vector<vector<vector<bool> > > IF_col_pruned;
    static vector<vector<bool> > IF_weight_pruned;
    static vector<vector<Dtype> > history_prob;
    static vector<vector<Dtype> > history_reg;
    static vector<vector<Dtype> > history_reg_weight;
    static vector<vector<Dtype> > hscore;
    static vector<vector<Dtype> > hrank;
    static vector<vector<Dtype> > hhrank;
    static vector<int> iter_prune_finished;
    static vector<Dtype> prune_ratio;
    static vector<Dtype> delta;
    static vector<Dtype> pruned_ratio;
    static vector<Dtype> pruned_ratio_col;
    static vector<Dtype> pruned_ratio_row;
    static vector<Dtype> GFLOPs;
    static vector<Dtype> num_param;
    static vector<Dtype> reg_to_distribute;
    
    
    static int num_log;
    static vector<vector<vector<Dtype> > > log_weight;
    static vector<vector<vector<Dtype> > > log_diff;
    static vector<vector<int> > log_index;
    static string snapshot_prefix;
    static string prune_state_dir;
    static string mask_generate_mechanism;
    
    static int show_interval;
    static int show_layer;
    static int show_num_layer;
    static int show_num_weight;
    
    /// --------------------------------
    
    static int window_size;
    static Dtype score_decay_rate;
    static bool use_score_decay;
    
    // When to Prune or Reg etc.
    static int when_to_col_reg;
    static Dtype col_reg;
    static Dtype diff_reg;
    
    // Decrease-Weight_Decay
    static int max_num_column_to_prune;

    // Selective Reg
    static Dtype reg_decay;
    static bool use_selective_reg;
    
    // Adaptive SPP
    static Dtype loss; 
    static Dtype loss_decay;
    static Dtype Delta_loss_history;
    static Dtype learning_speed;
    
    const static int test = 10;

}; 

    template<typename Dtype>  string  APP<Dtype>::prune_method   = "None"; /// initialized for caffe test, which has no solver but this info is still needed in layer.
    template<typename Dtype>  string  APP<Dtype>::prune_unit     = "None";
    template<typename Dtype>  string  APP<Dtype>::prune_coremthd = "None";
    template<typename Dtype>  string  APP<Dtype>::prune_coremthd_ = "None";
    template<typename Dtype>  string  APP<Dtype>::criteria;
    template<typename Dtype>  int     APP<Dtype>::num_once_prune;
    template<typename Dtype>  int     APP<Dtype>::prune_interval;
    template<typename Dtype>  int     APP<Dtype>::clear_history_interval;
    template<typename Dtype>  Dtype   APP<Dtype>::rgamma;
    template<typename Dtype>  Dtype   APP<Dtype>::rpower;
    template<typename Dtype>  Dtype   APP<Dtype>::cgamma;
    template<typename Dtype>  Dtype   APP<Dtype>::cpower;
    template<typename Dtype>  int     APP<Dtype>::prune_begin_iter;
    template<typename Dtype>  int     APP<Dtype>::iter_size;
    template<typename Dtype>  Dtype   APP<Dtype>::score_decay = 0;
    template<typename Dtype>  Dtype   APP<Dtype>::AA;
    template<typename Dtype>  Dtype   APP<Dtype>::kk;
    template<typename Dtype>  Dtype   APP<Dtype>::kk2;
    template<typename Dtype>  Dtype   APP<Dtype>::speedup;
    template<typename Dtype>  Dtype   APP<Dtype>::compRatio;
    template<typename Dtype>  bool    APP<Dtype>::IF_update_row_col;
    template<typename Dtype>  bool    APP<Dtype>::IF_speedup_count_fc;
    template<typename Dtype>  bool    APP<Dtype>::IF_compr_count_conv;
    template<typename Dtype>  bool    APP<Dtype>::IF_eswpf;
    template<typename Dtype>  Dtype   APP<Dtype>::prune_threshold;
    template<typename Dtype>  Dtype   APP<Dtype>::target_reg;
    template<typename Dtype>  int     APP<Dtype>::num_iter_reg;
    template<typename Dtype>  int     APP<Dtype>::reg_cushion_iter;
    template<typename Dtype>  Dtype   APP<Dtype>::hrank_momentum;

    // 1.2 Info shared between solver and layer, initailized here
    template<typename Dtype>  int   APP<Dtype>::inner_iter = 0;
    template<typename Dtype>  int   APP<Dtype>::step_ = -1;
    template<typename Dtype>  long   APP<Dtype>::last_time  = 0;
    template<typename Dtype>  long   APP<Dtype>::first_time = 0;
    template<typename Dtype>  int    APP<Dtype>::first_iter = 0;
    template<typename Dtype>  bool  APP<Dtype>::IF_scheme1_when_Reg_rank;
    template<typename Dtype>  bool  APP<Dtype>::IF_alpf = false; /// if all layer prune finished
    template<typename Dtype>  bool  APP<Dtype>::IF_speedup_achieved   = false;
    template<typename Dtype>  bool  APP<Dtype>::IF_compRatio_achieved = false;

    
    // 2.1 Info shared among layers
    template<typename Dtype>  map<string, int>  APP<Dtype>::layer_index;
    template<typename Dtype>  int  APP<Dtype>::fc_layer_cnt   = 0;
    template<typename Dtype>  int  APP<Dtype>::conv_layer_cnt = 0;
    template<typename Dtype>  vector<int>  APP<Dtype>::filter_area;
    template<typename Dtype>  vector<int>  APP<Dtype>::group;
    template<typename Dtype>  vector<int>  APP<Dtype>::priority;
    
     
    
    // 2.2 Pruning state (key)
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::num_pruned_col;
    template<typename Dtype>  vector<int>    APP<Dtype>::num_pruned_row;
    template<typename Dtype>  vector<int>    APP<Dtype>::num_pruned_weight;
    template<typename Dtype>  vector<int>    APP<Dtype>::pruned_rows; /// used in UpdateNumCol
    template<typename Dtype>  vector<bool>   APP<Dtype>::IF_update_row_col_layer; /// used in UpdateNumCol
    template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::masks;
    template<typename Dtype>  vector<vector<bool> >   APP<Dtype>::IF_row_pruned;
    template<typename Dtype>  vector<vector<vector<bool> > >  APP<Dtype>::IF_col_pruned;
    template<typename Dtype>  vector<vector<bool> >   APP<Dtype>::IF_weight_pruned;
    template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::history_prob;
    template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::history_reg;
    template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::history_reg_weight; // for each weight, not column
    template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::hscore;
    template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::hrank;
    template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::hhrank;
    template<typename Dtype>  vector<int>    APP<Dtype>::iter_prune_finished;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::prune_ratio;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::delta;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_col;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_row;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::GFLOPs;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::num_param;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::reg_to_distribute;
    
    // 3. Logging
    template<typename Dtype>  int     APP<Dtype>::num_log = 0; // > 0: logging is true
    template<typename Dtype>  vector<vector<vector<Dtype> > >  APP<Dtype>::log_weight;
    template<typename Dtype>  vector<vector<vector<Dtype> > >  APP<Dtype>::log_diff;
    template<typename Dtype>  vector<vector<int> >    APP<Dtype>::log_index;
    template<typename Dtype>  string  APP<Dtype>::snapshot_prefix;
    template<typename Dtype>  string  APP<Dtype>::prune_state_dir = "/PruneStateSnapshot/";
    template<typename Dtype>  string  APP<Dtype>::mask_generate_mechanism = "group-wise";
    
    template<typename Dtype>  int APP<Dtype>::show_interval = 10; 
    template<typename Dtype>  int APP<Dtype>::show_layer = 0; // The layer index of which the weights will be printed.
    template<typename Dtype>  int APP<Dtype>::show_num_layer = 100;
    template<typename Dtype>  int APP<Dtype>::show_num_weight = 40;
    /// --------------------------------

    // use window proposal or score decay ----- legacy
    template<typename Dtype>  int    APP<Dtype>::window_size = 40;
    template<typename Dtype>  bool   APP<Dtype>::use_score_decay = true;
    template<typename Dtype>  Dtype  APP<Dtype>::score_decay_rate = 0.88;
    
    // selective reg ----- legacy
    template<typename Dtype>  bool   APP<Dtype>::use_selective_reg = false; // default is false
    template<typename Dtype>  Dtype  APP<Dtype>::reg_decay = 0.59;
    
    // the penalty ratio of column regularization
    template<typename Dtype>  Dtype  APP<Dtype>::col_reg = 0.05; //0.0075; // 0.0008;  
    template<typename Dtype>  Dtype  APP<Dtype>::diff_reg = 0.00001; 
        
    // Decrease-Weight-Decay ----- legacy
    template<typename Dtype>  int  APP<Dtype>::max_num_column_to_prune = 0; // If "adaptive" used, this must be provided.
    // When to Prune or Reg etc.
    template<typename Dtype>  int  APP<Dtype>::when_to_col_reg = 7654321; // when to apply col reg, SSL or SelectiveReg

    // Adaptive SPP
    template<typename Dtype>  Dtype  APP<Dtype>::loss = 0;
    template<typename Dtype>  Dtype  APP<Dtype>::loss_decay = 0.7;
    template<typename Dtype>  Dtype  APP<Dtype>::Delta_loss_history = 0;
    template<typename Dtype>  Dtype  APP<Dtype>::learning_speed = 0;
}

#endif
