#ifndef DEEP_COMPRESSION_HPP_
#define DEEP_COMPRESSION_HPP_

#include <string>
#include <vector>
#define NUM_OF_WEIGHT_BUCKET 2
#define RATIO 0.5

using namespace std;

//static bool if_prune_;
//static bool is_first_step_;
//static float PruneRate_;

namespace caffe {
class DeepCompression {
public:
    DeepCompression() {};
    ~DeepCompression() {};

    static int step_;
    static float PruneRate[100];
    static float prune_ratio[100];
    static int prune_interval[100];
    static int num_pruned_column[100];
    static int num_pruned_row[100];
    static bool IN_TEST;
    static bool IN_RETRAIN;
    static string Method;
    
    static int window_size;
    static string criteria;
        
    static float score_decay_rate;
    static bool use_score_decay;
    static int num_of_col_to_prune_per_time;
    static int num_row_once_prune;
    
    // When to Prune or Reg etc.
    static int period;
    static int when_to_apply_masks;
    static int when_to_col_reg;
    static float col_reg;
    static float diff_reg;
    
    // Decrease-Weight_Decay
    static int when_to_dwd;
    static int dwd_end_iter;
    static float wd_end;
    static string dwd_mode;
    static int dwd_step;
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
    
    // history_prob
    static vector<float> history_prob[100];
    
    static vector<bool> IF_row_pruned[100]; // used in "pfilter" 
    static vector<bool> IF_col_pruned[100]; // used in "pfilter" 
    static int max_layer_index;
    static int filter_area[100];
    static int group[100];
    
    }; 

}

#endif  // CAFFE_NET_HPP_
