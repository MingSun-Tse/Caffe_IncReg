#include "caffe/adaptive_probabilistic_pruning.hpp"

using namespace std;
namespace caffe {
 

    /// --------------------------------
    /// pass params from solver.prototxt to layer, not initialized here.
    string APP::prune_method = "None"; // initialized for caffe test, which has no solver but this info is still needed in layer.
    string APP::criteria;
    int APP::num_once_prune;
    int APP::prune_interval;
    float APP::rgamma;
    float APP::rpower;
    float APP::cgamma;
    float APP::cpower;
    int APP::prune_begin_iter;
    int APP::iter_size;
    float APP::score_decay = 0;
    
    /// info shared between solver and layer, initailized here.
    int APP::inner_iter = 0;
    int APP::step_ = -1;
    
    /// info shared among layers
    map<string, int> APP::layer_index[2];
    int APP::layer_cnt[2] = {0, 0};
    
    vector<int> APP::num_pruned_col;
    vector<int> APP::num_pruned_row;
    vector<vector<bool> > APP::IF_row_pruned;
    vector<vector<bool> > APP::IF_col_pruned;
    vector<vector<float> > APP::history_prob;
    vector<bool> APP::IF_prune_finished;
    vector<float> APP::prune_ratio;
    vector<float> APP::delta;
    vector<float> APP::pruned_ratio;
    
    
    vector<int> APP::filter_area;
    vector<int> APP::group;
    vector<int> APP::priority;
    
    /// --------------------------------
    
    // use window proposal or score decay ----- legacy
    int APP::window_size = 40;
    bool APP::use_score_decay = true;
    float APP::score_decay_rate = 0.88;
    
    // selective reg ----- legacy
    bool APP::use_selective_reg = false; // default is false
    float APP::reg_decay = 0.59;
    
    // the penalty ratio of column regularization
    float APP::col_reg = 0.0012; // 0.0008;  
    float APP::diff_reg = 0.00001; 
        
    // Decrease-Weight-Decay ----- legacy
    int APP::max_num_column_to_prune = 0; // If "adaptive" used, this must be provided.
    // When to Prune or Reg etc.
    int APP::when_to_col_reg = 7654321; // when to apply col reg, SSL or SelectiveReg

    // Adaptive SPP
    float APP::loss = 0;
    float APP::loss_decay = 0.7;
    float APP::Delta_loss_history = 0;
    float APP::learning_speed = 0;

}
