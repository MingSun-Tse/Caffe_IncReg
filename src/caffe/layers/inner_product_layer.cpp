#include <vector>
#include "caffe/filler.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"
#define SHOW_NUM 20

namespace caffe {
using namespace std;

template <typename Dtype>
void InnerProductLayer<Dtype>::PruneSetUp(const PruneParameter& prune_param) {
    // Basic setting
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    
    // Get layer_index
    const string layer_name = this->layer_param_.name();
    if (this->phase_ == TRAIN) {
        if (APP::layer_index.count(layer_name) == 0) {
            APP::layer_index[layer_name] = APP::layer_cnt;
            ++ APP::layer_cnt;
            cout << "A new layer registered: " << layer_name 
                 << "  Total #layer: " << APP::layer_cnt << endl;
        }
    } else { return; }
    const int L = APP::layer_index[layer_name];
    const string mthd = APP::prune_method;
    cout << "PruneSetUp: " << layer_name  
         << "  its layer_index: " << L
         << "  Total #layer: " << APP::layer_cnt << endl;
    
    
    // Note: the varibales below can ONLY be used in training.
    // Note: These varibales will be called for every GPU, whereas since we use `layer_index` to index, so it doesn't matter.
    // Set up prune parameters of layer
    APP::prune_ratio.push_back(prune_param.prune_ratio());
    APP::pruned_ratio.push_back(0);
    APP::pruned_ratio_col.push_back(0);
    APP::pruned_ratio_row.push_back(0);
    APP::num_param.push_back(count);
    APP::GFLOPs.push_back(count);

    // Pruning state
    APP::masks.push_back( vector<bool>(count, 1) );
    APP::num_pruned_weight.push_back(0);
    APP::num_pruned_col.push_back(0);
    APP::num_pruned_row.push_back(0);
    APP::IF_weight_pruned.push_back( vector<bool>(count, false) );
    APP::IF_row_pruned.push_back( vector<bool>(num_row, false) );
    vector<bool> vec_tmp(1, false); // there is no group in fc layers, equivalent to group = 1
    APP::IF_col_pruned.push_back( vector<vector<bool> >(num_col, vec_tmp) );
    APP::hscore.push_back( vector<float>(count, 0) );
    APP::hrank.push_back( vector<float>(count, 0) );
    APP::hhrank.push_back( vector<float>(count, 0) );
    APP::history_reg.push_back( vector<float>(count, 0) );

    // Info shared among layers
    APP::priority.push_back(prune_param.priority());
    APP::iter_prune_finished.push_back(INT_MAX);
    
    cout << "=== Masks etc. Initialized" << endl;
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Print(const int& L, char mode) {
/** print example:
forward:
Index   WeightBeforeMasked   Mask   Prob - conv1
  c 1              0.04044      1      1
  c 2              0.05401      1      1
  c 3              0.06174      1      1

backward:
Index   DiffBeforeMasked   Mask   Prob - conv1
  c 1   0.08216(0.00003)      1      1
  c 2   0.08249(0.00004)      1      1
  c 3   0.08178(0.00007)      1      1
*/
    assert(mode == 'f' || mode = 'b'); /// forward, backward
    const int num_col = this->blobs_[0]->count() / this->blobs_[0]->shape()[0];
    const int num_row = this->blobs_[0]->shape()[0];
    const Dtype* w = this->blobs_[0]->cpu_data();
    const Dtype* d = this->blobs_[0]->cpu_diff();

    // print Index, blob, Mask
    cout.width(5);  cout << "Index" << "   ";
    const string blob = (mode == 'f') ? "WeightBeforeMasked" : "DiffBeforeMasked";
    cout.width(blob.size()); cout << blob << "   ";
    cout.width(4);  cout << "Mask" << "   ";
    
    // print additional info
    char* mthd = new char[strlen(APP::prune_method.c_str()) + 1];
    strcpy(mthd, APP::prune_method.c_str());
    const string mthd_ = strtok(mthd, "_"); // mthd is like "Reg_Col", the first split is `Reg`
    string info;
    vector<float> info_data; 
    if (mthd_ == "Reg") {
        info = "HistoryReg";
        info_data = APP::history_reg[L];
    } else if (mthd_ == "SPP") {
        info = "HistoryProb";
        info_data = APP::history_prob[L];
    }
    cout.width(info.size()); cout << info << " - " << this->layer_param_.name() << endl;
    
    if (APP::prune_unit == "Row") {
        for (int i = 0; i < SHOW_NUM; ++i) {
            // print Index
            cout.width(3); cout << "r"; 
            cout.width(2); cout << i+1 << "   ";
            
            // print blob
            char s[50]; sprintf(s, "%7.5f", d[i * num_col]); // TODO: improve the print for row prune
            if (mode == 'f') { sprintf(s, "%f", fabs(w[i * num_col])); }
            cout.width(blob.size()); cout << s << "   ";
            
            // print Mask
            cout.width(4);  cout << APP::masks[L][i * num_col] << "   ";
            
            // print info
            cout.width(info.size());  cout << info_data[i] << endl;
        }
        
    } else if (APP::prune_unit == "Col") {
        for (int j = 0; j < SHOW_NUM; ++j) {
            // print Index
            cout.width(3); cout << "c"; 
            cout.width(2); cout << j+1 << "   ";
            
            // print blob
            Dtype sum_w = 0, sum_d = 0;
            for (int i = 0; i < num_row; ++i) {
                sum_w += fabs(w[i * num_col + j]);
                sum_d += fabs(d[i * num_col + j]);
            }
            sum_w /= num_row; /// average abs weight
            sum_d /= num_row; /// average abs diff
            const Dtype reg_force = APP::history_reg[L][j] * sum_w;
            char s[50]; sprintf(s, "%7.5f(%7.5f)", sum_d, reg_force);
            if (mode == 'f') { sprintf(s, "%f", sum_w); }
            cout.width(blob.size()); cout << s << "   ";
            
            // print Mask
            cout.width(4);  cout << APP::masks[L][j] << "   ";
            
            // print info
            cout.width(info.size());  cout << info_data[j] << endl;
        }
    } else if (APP::prune_unit == "Weight") {
        for (int i = 0; i < SHOW_NUM; ++i) {
            // print Index
            cout.width(3); cout << "w";
            cout.width(2); cout << i+1 << "   ";
            
            // print blob
            const Dtype reg_force = APP::history_reg[L][i] * fabs(w[i]);
            char s[50]; sprintf(s, "%7.5f(%7.5f)", fabs(d[i]), reg_force);
            if (mode == 'f') { sprintf(s, "%f", fabs(w[i])); }
            cout.width(blob.size()); cout << s << "   ";
            
            // print Mask
            cout.width(4);  cout << APP::masks[L][i] << "   ";
            
            // print info
            cout.width(info.size());  cout << info_data[i] << endl;
        }
    }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::IF_alpf() {
    /** IF_all_layer_prune_finished
    */
    APP::IF_alpf = true;
    for (int i = 0; i < APP::layer_cnt; ++i) {
        if (APP::iter_prune_finished[i] == INT_MAX) {
            APP::IF_alpf = false;
            break;
        }
    }
}

template <typename Dtype> 
void InnerProductLayer<Dtype>::UpdatePrunedRatio() {
    const int L = APP::layer_index[this->layer_param_.name()];
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    
    if (APP::prune_unit == "Weight") {
        for (int i = 0; i < num_row; ++i) {
            if (APP::IF_row_pruned[L][i]) { continue; }
            bool IF_whole_row_pruned = true;
            for (int j = 0; j < num_col; ++j) {
                if (!APP::IF_weight_pruned[L][i * num_col + j]) {
                    IF_whole_row_pruned = false;
                    break;
                }
            }
            if (IF_whole_row_pruned) {
                APP::IF_row_pruned[L][i] = true;
                APP::num_pruned_row[L] += 1;
            }
        }
        for (int j = 0; j < num_col; ++j) {
            if (APP::IF_col_pruned[L][j][0]) { continue; }
            bool IF_whole_col_pruned = true;
            for (int i = 0; i < num_row; ++i) {
                if (!APP::IF_weight_pruned[L][i * num_col + j]) {
                    IF_whole_col_pruned = false;
                    break;
                }
            }
            if (IF_whole_col_pruned) {
                APP::IF_col_pruned[L][j][0] = true;
                APP::num_pruned_col[L] += 1;
            }
        }
    }
    APP::pruned_ratio_col[L] = APP::num_pruned_col[L] / num_col;
    APP::pruned_ratio_row[L] = APP::num_pruned_row[L] * 1.0 / num_row;
    if (APP::prune_unit == "Weight") {
        APP::pruned_ratio[L] = APP::num_pruned_weight[L] * 1.0 / count;
    } else {
        APP::pruned_ratio[L] =  (APP::pruned_ratio_col[L] + APP::pruned_ratio_row[L]) 
                               - APP::pruned_ratio_col[L] * APP::pruned_ratio_row[L];
    }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::ComputeBlobMask() {
    Dtype* muweight   = this->blobs_[0]->mutable_cpu_data();
    const int count   = this->blobs_[0]->count();
    const int L = APP::layer_index[this->layer_param_.name()];
   
    if (APP::prune_unit == "Weight") {
        for (int i = 0; i < count; ++i) {
            if (fabs(muweight[i]) < APP::prune_threshold || APP::history_reg[L][i] >= APP::target_reg) {
                muweight[i] = 0;
                APP::masks[L][i] = 0;
                APP::num_pruned_weight[L] += 1;
                APP::IF_weight_pruned[L][i] = true;
                APP::hrank[L][i] = APP::step_ - 1000000 - (APP::history_reg[L][i] - APP::target_reg);
            }
        }
        APP::pruned_ratio[L] = APP::num_pruned_weight[L] * 1.0 / count;
        if (APP::pruned_ratio[L] >= APP::prune_ratio[L]) {
            APP::iter_prune_finished[L] = -1;
            cout << this->layer_param_.name() << " prune finshed" << endl;
        }
        LOG(INFO) << "    Masks restored"
                  << "  pruned_ratio = "   << APP::pruned_ratio[L]
                  << "  prune_ratio = "    << APP::prune_ratio[L];
    }
    
}


template <typename Dtype>
void InnerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.inner_product_param().num_output();
  bias_term_ = this->layer_param_.inner_product_param().bias_term();
  transpose_ = this->layer_param_.inner_product_param().transpose();
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Initialize the weights
    vector<int> weight_shape(2);
    if (transpose_) {
      weight_shape[0] = K_;
      weight_shape[1] = N_;
    } else {
      weight_shape[0] = N_;
      weight_shape[1] = K_;
    }
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.inner_product_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.inner_product_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
  // Added by WANGHUAN for pruning
  PruneSetUp(this->layer_param_.prune_param());
  
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.inner_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::PruneMinimals() {
    Dtype* muweight   = this->blobs_[0]->mutable_cpu_data();
    const int count   = this->blobs_[0]->count();
    const int L = APP::layer_index[this->layer_param_.name()];
    
    if (APP::prune_unit == "Weight") {
        for (int i = 0; i < count; ++i) {
            if (APP::IF_weight_pruned[L][i]) { continue; }
            if (fabs(muweight[i]) < APP::prune_threshold || APP::history_reg[L][i] >= APP::target_reg) {
                muweight[i] = 0;
                APP::masks[L][i] = 0;
                APP::num_pruned_weight[L] += 1;
                APP::IF_weight_pruned[L][i] = true;
                APP::hrank[L][i]  = APP::step_ - 1000000 - (APP::history_reg[L][i] - APP::target_reg);
                APP::hhrank[L][i] = APP::step_ - 1000000 - (APP::history_reg[L][i] - APP::target_reg);
            }
        }
    }
}


template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, transpose_ ? CblasNoTrans : CblasTrans,
      M_, N_, K_, (Dtype)1.,
      bottom_data, weight, (Dtype)0., top_data);
  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          K_, N_, M_,
          (Dtype)1., bottom_data, top_diff,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans,
          N_, K_, M_,
          (Dtype)1., top_diff, bottom_data,
          (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    }

  }

  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }


  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bottom data
    if (transpose_) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    } else {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
          M_, K_, N_,
          (Dtype)1., top_diff, this->blobs_[0]->cpu_data(),
          (Dtype)0., bottom[0]->mutable_cpu_diff());
    }
  }
}



//template <typename Dtype>
//void InnerProductLayer<Dtype>::ComputeBlobMask(float ratio) {
//    int count = this->blobs_[0]->count();
//
//    this->masks_.resize(count);
//    this->indices_.resize(count);
//    this->centroids_.resize(NUM_OF_WEIGHT_BUCKET);
//
//
//    const Dtype *weight = this->blobs_[0]->cpu_data();
//    vector<Dtype> sort_weight(count);
//    for (int i = 0; i < count; i++)
//        sort_weight[i] = fabs(weight[i]);
//    sort(sort_weight.begin(), sort_weight.end());
//    int index = (int) (count * ratio); // ratio: the ratio to cut off
//    Dtype thr; // threshold
//    Dtype *muweight = this->blobs_[0]->mutable_cpu_data();
//    float rat = 0; // non-zero weight counter, for calculating sparsity
//    if (index > 0) {
//        thr = sort_weight[index - 1];
//        LOG(INFO) << "Inner Threshold: " << thr << " " << ratio << endl;
//        for (int i = 0; i < count; i++) {
//            this->masks_[i] = ((weight[i] >= thr || weight[i] < -thr) ? 1 : 0);
//            muweight[i] *= this->masks_[i];
//            rat += (1 - this->masks_[i]);
//        }
//    } else {
//        std::cout << "wanghuan test: There is no weights in this layer" << endl;
//        for (int i = 0; i < count; i++) {
//            this->masks_[i] = (weight[i] == 0 ? 0 : 1);
//            rat += (1 - this->masks_[i]);
//        }
//    }
//    LOG(INFO) << "sparsity: " << rat / count << endl;
//    Layer<Dtype>::kmeans_cluster(this->indices_, this->centroids_, muweight, count, this->masks_, NUM_OF_WEIGHT_BUCKET, 1000);
//}


#ifdef CPU_ONLY
STUB_GPU(InnerProductLayer);
#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTER_LAYER_CLASS(InnerProduct);

}  // namespace caffe
