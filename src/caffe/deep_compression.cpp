#include "caffe/deep_compression.hpp"

using namespace std;
namespace caffe {
 

    /// --------------------------------
    /// pass params from solver.prototxt to layer, not initialized here.
    string DeepCompression::prune_method = "None"; // initialized for caffe test
    string DeepCompression::criteria;
    int DeepCompression::num_once_prune;
    int DeepCompression::prune_interval;
    float DeepCompression::rgamma;
    float DeepCompression::rpower;
    float DeepCompression::cgamma;
    float DeepCompression::cpower;
    int DeepCompression::prune_begin_iter;
    int DeepCompression::iter_size;
    float DeepCompression::score_decay = 0.9;
    
    /// share params between solver and layer, initailized here.
    int DeepCompression::inner_iter = 0;
    int DeepCompression::step_ = -1;
    int DeepCompression::num_pruned_col[100] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int DeepCompression::num_pruned_row[100] = {0, 0, 0, 0, 0};
    bool DeepCompression::IN_TEST = false;
    bool DeepCompression::IN_RETRAIN = false;
    bool DeepCompression::IF_prune_finished[100] = {0, 0, 0, 0, 0};
    /// --------------------------------
    
    
    // use window proposal or score decay
    int DeepCompression::window_size = 40;
    bool DeepCompression::use_score_decay = true;
    float DeepCompression::score_decay_rate = 0.88;
    
    // selective reg
    bool DeepCompression::use_selective_reg = false; // default is false
    float DeepCompression::reg_decay = 0.59;
    
    // the penalty ratio of column regularization
    float DeepCompression::col_reg = 0.0012; // 0.0008;  
    float DeepCompression::diff_reg = 0.00001; 
        
    // Decrease-Weight-Decay
    int DeepCompression::max_num_column_to_prune = 0; // If "adaptive" used, this must be provided.
    
    // When to Prune or Reg etc.
    int DeepCompression::when_to_col_reg = 7654321; // when to apply col reg, SSL or SelectiveReg
    
    
    // PruneRate etc.
    float DeepCompression::PruneRate[100] = {0.75, 0.75, 0.75, 0.75, 0.75};

    
    // Adaptive SPP
    float DeepCompression::loss = 0;
    float DeepCompression::loss_decay = 0.7;
    float DeepCompression::Delta_loss_history = 0;
    float DeepCompression::learning_speed = 0;
    
    // history_prob
    vector<float> DeepCompression::history_prob[100];
    
    vector<bool> DeepCompression::IF_row_pruned[100];
    vector<bool> DeepCompression::IF_col_pruned[100];
    int DeepCompression::max_layer_index = 0;
    int DeepCompression::filter_area[100] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int DeepCompression::group[100] = {1, 1, 1, 1, 1};

    
    
}
