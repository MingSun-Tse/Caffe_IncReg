#ifndef DEEP_COMPRESSION_HPP_
#define DEEP_COMPRESSION_HPP_

#include <string>
#include <vector>
#define NUM_OF_WEIGHT_BUCKET 2
#define RATIO 0.5


namespace caffe {
using namespace std;

class DeepCompression {
public:
     DeepCompression() {};
    ~DeepCompression() {};

    /// --------------------------------
    /// pass params from solver.prototxt to layer
    static string prune_method;
    static string criteria;
    static int num_once_prune;
    static int prune_interval;
    static float rgamma;
    static float rpower;
    static float cgamma;
    static float cpower;
    static int prune_begin_iter;
    static int iter_size;
    
    /// share params between solver and layer
    static bool IN_TEST;
    static bool IN_RETRAIN;
    static int inner_iter;
    static int num_pruned_column[100];
    static int num_pruned_row[100];
    static int step_;
    /// --------------------------------
    
    static float PruneRate[100];

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
