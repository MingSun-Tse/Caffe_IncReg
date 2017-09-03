#include "caffe/deep_compression.hpp"

using namespace std;
namespace caffe {


    // ----------------
    bool DeepCompression::IF_mask;
    string DeepCompression::prune_method;
    string DeepCompression::criteria;
    int DeepCompression::num_once_prune;
    int DeepCompression::prune_interval;
    float DeepCompression::rgamma;
    float DeepCompression::rpower;
    float DeepCompression::cgamma;
    float DeepCompression::cpower;
    int DeepCompression::prune_begin_iter;
    // ----------------
    
    int DeepCompression::step_ = -1;
    int DeepCompression::num_pruned_column[100] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int DeepCompression::num_pruned_row[100] = {0, 0, 0, 0, 0};
    bool DeepCompression::IN_TEST = false;
    bool DeepCompression::IN_RETRAIN = false;
    string DeepCompression::Method = "PFilter"; // "SPP", "PPruning", "PFilter"
    

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
    const int begin_time = -1;
    int DeepCompression::when_to_dwd = begin_time; // when to decrease weight decay
    int DeepCompression::dwd_end_iter = 150000;
    float DeepCompression::wd_end = 0.175; // decrease weight decay to, say, 20%
    string DeepCompression::dwd_mode = "step_linearly"; // "linealy, step_linearly, adaptive, None". If "None" used, don't decrease weight decay.
    int DeepCompression::dwd_step = 100; // If "step_linearly" used, this must be provided.
    int DeepCompression::max_num_column_to_prune = 0; // If "adaptive" used, this must be provided.
    
    // When to Prune or Reg etc.
    int DeepCompression::period = 7654321; // a hyper-parameters for paper of IHT (Iterative Hard Thresholding), used for iterative training
    int DeepCompression::when_to_col_reg = 7654321; // when to apply col reg, SSL or SelectiveReg
    int DeepCompression::when_to_apply_masks = begin_time; // To determine when masks work. -1: always apply masks
    
    
    // PruneRate etc.
    float DeepCompression::PruneRate[100] = {0.75, 0.75, 0.75, 0.75, 0.75};
    float DeepCompression::prune_ratio[100] = {0.75, 0.75, 0.75, 0.75, 0.75};
    int DeepCompression::num_of_col_to_prune_per_time = 1;  // the number of columns to prune per time
    int DeepCompression::num_row_once_prune = 2; // the "m" parameter in paper "Pruning Filter"
    
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
