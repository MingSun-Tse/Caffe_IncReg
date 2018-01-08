 #include "caffe/adaptive_probabilistic_pruning.hpp"


namespace caffe {
    /// --------------------------------
    // 1.1 Params passed from solver to layer, not initialized here
    // template<typename Dtype>  string  APP<Dtype>::prune_method   = "None"; /// initialized for caffe test, which has no solver but this info is still needed in layer.
    // template<typename Dtype>  string  APP<Dtype>::prune_unit     = "None";
    // template<typename Dtype>  string  APP<Dtype>::prune_coremthd = "None";
    // template<typename Dtype>  string  APP<Dtype>::criteria;
    // template<typename Dtype>  int     APP<Dtype>::num_once_prune;
    // template<typename Dtype>  int     APP<Dtype>::prune_interval;
    // template<typename Dtype>  Dtype   APP<Dtype>::rgamma;
    // template<typename Dtype>  Dtype   APP<Dtype>::rpower;
    // template<typename Dtype>  Dtype   APP<Dtype>::cgamma;
    // template<typename Dtype>  Dtype   APP<Dtype>::cpower;
    // template<typename Dtype>  int     APP<Dtype>::prune_begin_iter;
    // template<typename Dtype>  int     APP<Dtype>::iter_size;
    // template<typename Dtype>  Dtype   APP<Dtype>::score_decay = 0;
    // template<typename Dtype>  Dtype   APP<Dtype>::AA;
    // template<typename Dtype>  Dtype   APP<Dtype>::kk;
    // template<typename Dtype>  Dtype   APP<Dtype>::speedup;
    // template<typename Dtype>  Dtype   APP<Dtype>::compRatio;
    // template<typename Dtype>  bool    APP<Dtype>::IF_update_row_col;
    // template<typename Dtype>  bool    APP<Dtype>::IF_eswpf;
    // template<typename Dtype>  Dtype   APP<Dtype>::prune_threshold;
    // template<typename Dtype>  Dtype   APP<Dtype>::target_reg;
    // template<typename Dtype>  int     APP<Dtype>::num_iter_reg;
    // template<typename Dtype>  int     APP<Dtype>::reg_cushion_iter;
    // template<typename Dtype>  Dtype   APP<Dtype>::hrank_momentum;

    // // 1.2 Info shared between solver and layer, initailized here
    // template<typename Dtype>  int   APP<Dtype>::inner_iter = 0;
    // template<typename Dtype>  int   APP<Dtype>::step_ = -1;
    // template<typename Dtype>  bool  APP<Dtype>::IF_alpf = false; /// if all layer prune finished
    // template<typename Dtype>  bool  APP<Dtype>::IF_speedup_achieved   = false;
    // template<typename Dtype>  bool  APP<Dtype>::IF_compRatio_achieved = false;

    
    // // 2.1 Info shared among layers
    // template<typename Dtype>  map<string, int>  APP<Dtype>::layer_index;
    // template<typename Dtype>  int  APP<Dtype>::fc_layer_cnt   = 0;
    // template<typename Dtype>  int  APP<Dtype>::conv_layer_cnt = 0;
    // template<typename Dtype>  vector<int>  APP<Dtype>::filter_area;
    // template<typename Dtype>  vector<int>  APP<Dtype>::group;
    // template<typename Dtype>  vector<int>  APP<Dtype>::priority;
    
     
    
    // // 2.2 Pruning state (key)
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::num_pruned_col;
    // template<typename Dtype>  vector<int>    APP<Dtype>::num_pruned_row;
    // template<typename Dtype>  vector<int>    APP<Dtype>::num_pruned_weight;
    // template<typename Dtype>  vector<int>    APP<Dtype>::pruned_rows; /// used in UpdateNumCol
    // template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::masks;
    // template<typename Dtype>  vector<vector<bool> >   APP<Dtype>::IF_row_pruned;
    // template<typename Dtype>  vector<vector<vector<bool> > >  APP<Dtype>::IF_col_pruned;
    // template<typename Dtype>  vector<vector<bool> >   APP<Dtype>::IF_weight_pruned;
    // template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::history_prob;
    // template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::history_reg;
    // template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::history_reg_weight; // for each weight, not column
    // template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::hscore;
    // template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::hrank;
    // template<typename Dtype>  vector<vector<Dtype> >  APP<Dtype>::hhrank;
    // template<typename Dtype>  vector<int>    APP<Dtype>::iter_prune_finished;
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::prune_ratio;
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::delta;
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio;
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_col;
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::pruned_ratio_row;
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::GFLOPs;
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::num_param;
    // template<typename Dtype>  vector<Dtype>  APP<Dtype>::reg_to_distribute;
    
    // // 3. Logging
    // //template<typename Dtype>  int     APP<Dtype>::num_log = 0;
    // template<typename Dtype>  vector<vector<vector<Dtype> > >  APP<Dtype>::log_weight;
    // template<typename Dtype>  vector<vector<vector<Dtype> > >  APP<Dtype>::log_diff;
    // template<typename Dtype>  vector<vector<int> >    APP<Dtype>::log_index;
    // template<typename Dtype>  string  APP<Dtype>::snapshot_prefix;
    // /// --------------------------------

    // // use window proposal or score decay ----- legacy
    // template<typename Dtype>  int    APP<Dtype>::window_size = 40;
    // template<typename Dtype>  int    APP<Dtype>::num_log2 = 40;
    // template<typename Dtype>  int    APP<Dtype>::num_log = 0;
    // template<typename Dtype>  bool   APP<Dtype>::use_score_decay = true;
    // template<typename Dtype>  Dtype  APP<Dtype>::score_decay_rate = 0.88;
    
    // // selective reg ----- legacy
    // template<typename Dtype>  bool   APP<Dtype>::use_selective_reg = false; // default is false
    // template<typename Dtype>  Dtype  APP<Dtype>::reg_decay = 0.59;
    
    // // the penalty ratio of column regularization
    // template<typename Dtype>  Dtype  APP<Dtype>::col_reg = 0.05; //0.0075; // 0.0008;  
    // template<typename Dtype>  Dtype  APP<Dtype>::diff_reg = 0.00001; 
        
    // // Decrease-Weight-Decay ----- legacy
    // template<typename Dtype>  int  APP<Dtype>::max_num_column_to_prune = 0; // If "adaptive" used, this must be provided.
    // // When to Prune or Reg etc.
    // template<typename Dtype>  int  APP<Dtype>::when_to_col_reg = 7654321; // when to apply col reg, SSL or SelectiveReg

    // // Adaptive SPP
    // template<typename Dtype>  Dtype  APP<Dtype>::loss = 0;
    // template<typename Dtype>  Dtype  APP<Dtype>::loss_decay = 0.7;
    // template<typename Dtype>  Dtype  APP<Dtype>::Delta_loss_history = 0;
    // template<typename Dtype>  Dtype  APP<Dtype>::learning_speed = 0;

}
