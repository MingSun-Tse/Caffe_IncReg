#ifndef ADAPTIVE_PROBABILISTIC_PRUNING_HPP_
#define ADAPTIVE_PROBABILISTIC_PRUNING_HPP_

#include <string>
#include <vector>
#include <map>
#define NUM_OF_WEIGHT_BUCKET 2
#define RATIO 0.5


namespace caffe {
using namespace std;

class APP {
public:
     APP() {};
    ~APP() {};

    /// --------------------------------
    /// pass params from solver.prototxt to layer
    static string prune_method;
    static string prune_unit;
    static string criteria;
    static int num_once_prune;
    static int prune_interval;
    static float rgamma;
    static float rpower;
    static float cgamma;
    static float cpower;
    static int prune_begin_iter;
    static int iter_size;
    static float score_decay;
    static float AA;
    static float kk;
    static float speedup;
    static float compRatio;
    static bool IF_update_row_col;
    static bool IF_eswpf;
    static float prune_threshold;
    static float target_reg;
    static int num_iter_reg;
    static int reg_cushion_iter; // In the beginning of reg, improve the reg little-by-little to mitigate the side-effect. `reg_cushion_iter` is the iter of this mitigation perioid
    static float hrank_momentum;
    
    
    
    
    static int inner_iter;
    static int step_;
    static bool IF_alpf; 
    static bool IF_speedup_achieved;
    static bool IF_compRatio_achieved;
    
    
    static map<string, int> layer_index;
    static int layer_cnt;
    static vector<int> filter_area;
    static vector<int> group;
    static vector<int> priority;
    
    
    
    static vector<float> num_pruned_col;
    static vector<int>   num_pruned_row;
    static vector<int>   num_pruned_weight;
    static vector<int>   pruned_rows;
    static vector<vector<bool> > masks;
    static vector<vector<bool> > IF_row_pruned;
    static vector<vector<vector<bool> > > IF_col_pruned;
    static vector<vector<bool> > IF_weight_pruned;
    static vector<vector<float> > history_prob;
    static vector<vector<float> > history_reg;
    static vector<vector<float> > history_reg_weight;
    static vector<vector<float> > hscore;
    static vector<vector<float> > hrank;
    static vector<vector<float> > hhrank;
    static vector<int> iter_prune_finished;
    static vector<float> prune_ratio;
    static vector<float> delta;
    static vector<float> pruned_ratio;
    static vector<float> pruned_ratio_col;
    static vector<float> pruned_ratio_row;
    static vector<float> GFLOPs;
    static vector<float> num_param;
    static vector<float> reg_to_distribute;
    
    
    static int num_log;
    static vector<vector<vector<float> > > log_weight;
    static vector<vector<vector<float> > > log_diff;
    static vector<vector<int> > log_index;
    static string snapshot_prefix;
    /// --------------------------------
    
    static int window_size;  
    static float score_decay_rate;
    static bool use_score_decay;
    
    // When to Prune or Reg etc.
    static int when_to_col_reg;
    static float col_reg;
    static float diff_reg;
    
    // Decrease-Weight_Decay
    static int max_num_column_to_prune;

    // Selective Reg
    static float reg_decay;
    static bool use_selective_reg;
    static float selective_reg_cut_threshold;
    
    // Adaptive SPP
    static float loss; 
    static float loss_decay;
    static float Delta_loss_history;
    static float learning_speed;
    
    

    

    
}; 

}

#endif
