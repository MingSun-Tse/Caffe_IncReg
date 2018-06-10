#ifndef ADAPTIVE_PROBABILISTIC_PRUNING_HPP_
#define ADAPTIVE_PROBABILISTIC_PRUNING_HPP_

#include <string>
#include <vector>
#include <map>
//#define ShowTimingLog 1

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
    static int num_once_prune;
    static int prune_interval;
    static int clear_history_interval;
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
    static int reg_cushion_iter; // In the beginning of reg, improve the reg little-by-little to mitigate the side-effect. `reg_cushion_iter` is the iter of this mitigation perioid
    
    static int inner_iter;
    static int step_;
    static int num_;
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
    static vector<Dtype> prune_ratio;
    static vector<Dtype> pruned_ratio;
    static vector<Dtype> pruned_ratio_col;
    static vector<Dtype> pruned_ratio_row;
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
}; 

    template<typename Dtype>  string  APP<Dtype>::prune_method   = "None"; /// initialized for caffe test, which has no solver but this info is still needed in layer.
    template<typename Dtype>  string  APP<Dtype>::prune_unit     = "None";
    template<typename Dtype>  string  APP<Dtype>::prune_coremthd = "None";
    template<typename Dtype>  string  APP<Dtype>::prune_coremthd_ = "None";
    template<typename Dtype>  int     APP<Dtype>::num_once_prune;
    template<typename Dtype>  int     APP<Dtype>::prune_interval;
    template<typename Dtype>  int     APP<Dtype>::clear_history_interval;
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
    template<typename Dtype>  int     APP<Dtype>::reg_cushion_iter;

    // 1.2 Info shared between solver and layer, initailized here
    template<typename Dtype>  int   APP<Dtype>::inner_iter = 0;
    template<typename Dtype>  int   APP<Dtype>::step_ = -1;
    template<typename Dtype>  int   APP<Dtype>::num_; // batch_size
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
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::prune_ratio;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_col;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_row;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::GFLOPs;
    template<typename Dtype>  vector<Dtype>  APP<Dtype>::num_param;
    
    // 3. Logging
    template<typename Dtype>  int     APP<Dtype>::num_log = 0; // > 0: logging is true
    template<typename Dtype>  string  APP<Dtype>::mask_generate_mechanism = "group-wise";
    template<typename Dtype>  int     APP<Dtype>::input_length = 16; // TODO(mingsuntse): to use solver.prototxt or autoset.
    template<typename Dtype>  bool    APP<Dtype>::simulate_5d = false;
    template<typename Dtype>  int     APP<Dtype>::h_off = 0; // used in data transformation
    template<typename Dtype>  int     APP<Dtype>::w_off = 0;
    template<typename Dtype>  int APP<Dtype>::show_interval = 1; 
    template<typename Dtype>  string APP<Dtype>::show_layer = "1001";
    template<typename Dtype>  int APP<Dtype>::show_num_layer = 100;
    template<typename Dtype>  int APP<Dtype>::show_num_weight = 20;
}

#endif
