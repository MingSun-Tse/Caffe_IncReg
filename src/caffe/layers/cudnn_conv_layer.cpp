#ifdef USE_CUDNN
#include <algorithm>
#include <vector>
#include <math.h>
#include <stdlib.h> // rand() and srand()
#include <time.h>


#include "caffe/layers/cudnn_conv_layer.hpp"
#include "caffe/deep_compression.hpp"

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3

/**
 * TODO(dox) explain cuDNN interface
 */
 
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::UpdateHistoryScores() {
    const int count = this->blobs_[0]->count();
    const int num_filter = this->blobs_[0]->shape()[0];
    const int num_col = count / num_filter;
    const Dtype* weight = this->blobs_[0]->cpu_data();
    const Dtype* weight_diff = this->blobs_[0]->cpu_diff();
    
    Dtype sum = 0;
    for (int j = 0; j < num_col; ++j) {
        if (this->is_pruned[j]) { continue; }
        if (DeepCompression::criteria == "energy") {
            sum = 0;
            for (int i = 0; i < num_filter; ++i) {
                sum += fabs(weight[i * num_col + j]); // * weight[i * num_col + j];
            }
        } else if (DeepCompression::criteria == "diff") {
            sum = 0;
            for (int i = 0; i < num_filter; ++i) {
                sum += fabs(weight[i * num_col + j] * weight_diff[i * num_col + j]);
            }
        }
        this->history_score[j] = DeepCompression::score_decay_rate * this->history_score[j] + sum;
    }        
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::UpdateDiffs() {
    const int count = this->blobs_[0]->count();
    const Dtype* weight = this->blobs_[0]->cpu_data();
    const Dtype* weight_diff = this->blobs_[0]->cpu_diff();
    
    for (int i = 0; i < count; ++i) {
        this->blobs_[0]->mutable_cpu_second_diff[i] = (weight_diff[i] - this->history_diff[i]) / (weight[i] - this->history_weight[i]);
        this->history_diff[i] = weight_diff[i];
        this->history_weight[i] = weight[i];
    }
}

template <typename Dtype>
vector<int> CuDNNConvolutionLayer<Dtype>::GetColToPrune() {
    const int num_ = DeepCompression::num_of_col_to_prune_per_time;
    vector<int> min_index(num_, 0);
    const int count = this->blobs_[0]->count();
    const int num_filter = this->blobs_[0]->shape()[0];
    const int num_col = count / num_filter;
    
    typedef std::pair<Dtype, int> mypair;
    vector<mypair> hist_score(num_col);
    for (int j = 0; j < num_col; ++j) {
        hist_score[j].first = this->history_score[j];
        hist_score[j].second = j;
    }
    sort(hist_score.begin(), hist_score.end()); // in ascending order
    for (int j = 0; j < num_; ++j) {
        min_index[j] = hist_score[this->num_pruned_column + j].second;
        // std:cout << "min_index:" << min_index[j] << std::endl; // check
    }
    
    return min_index;
}

template <typename Dtype>
int CuDNNConvolutionLayer<Dtype>::ProposeCandidate() {
  Dtype min = 1e20;
  int min_index = 0;
  Dtype sum = 0;
  
  const int count = this->blobs_[0]->count();
  const int num_filter = this->blobs_[0]->shape()[0];
  const int num_col = count / num_filter;
  const Dtype* weight = this->blobs_[0]->cpu_data();
  const Dtype* weight_diff = this->blobs_[0]->cpu_diff();
  
  for (int j = 0; j < num_col; ++j) {
      if (this->is_pruned[j]) { continue; }
      
      // compute criteria
      if (DeepCompression::criteria == "energy") {
          sum = 0;
          for (int i = 0; i < num_filter; ++i) {
              sum += fabs(weight[i * num_col + j]); // * weight[i * num_col + j];
          }        
      } else if (DeepCompression::criteria == "diff") {
          sum = 0;
          for (int i = 0; i < num_filter; ++i) {
              sum += fabs(weight[i * num_col + j] * weight_diff[i * num_col + j]);
          }
      }
      
      if (sum < min) {
          min = sum;
          min_index = j;
      }    
  }
  return min_index;
}

template <typename Dtype>
int CuDNNConvolutionLayer<Dtype>::GetMostFrequent(vector<int> candidate, int num_total_column) {
    vector<int>::iterator it = candidate.begin();
    vector<int> hist(num_total_column, 0);
    while (it != candidate.end()) {
        if (*it != -1) { hist[*it]++; }
        it ++;
    }
    return max_element(hist.begin(), hist.end()) - hist.begin();    
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::UpdateMasks() {
    const int count = this->blobs_[0]->count();
    const int num_filter = this->blobs_[0]->shape()[0];
    const int num_col = count / num_filter;
    const Dtype* weight = this->blobs_[0]->cpu_data();
        
    const string layer_name = this->layer_param_.name();
    const int conv_index = layer_name[4] - '1';
    const float PruneRate = DeepCompression::PruneRate[conv_index];    
    const int step = DeepCompression::step_;
    const int prune_interval = DeepCompression::prune_interval[conv_index];
    
    const string prune_what = "prune_col";
    const bool use_score_decay = DeepCompression::use_score_decay;
    if (prune_what == "prune_col") {        
        if (this->num_pruned_column >= PruneRate * num_col) { return; } 
        
        // Selective Reg
        // remove those minimal columns
        if (0) {
            for (int j = 0; j < num_col; ++j) {
                if (this->is_pruned[j]) { continue; }
                Dtype abs_sum = 0;
                for (int i = 0; i < num_filter; ++i) {
                    abs_sum += fabs(weight[i * num_filter + j]); 
                }
                if (abs_sum / num_filter < DeepCompression::selective_reg_cut_threshold) {
                    for (int i = 0; i < num_filter; ++i) {
                        this->masks_[i * num_col + j] = 0;
                    }
                    this->is_pruned[j] = true;
                    this->num_pruned_column ++;
                }
            }
            return;    
        }
        
        
        if (use_score_decay) { 
            UpdateHistoryScores(); 
        } else { // use window proposal
            this->candidate_window[(step - 1) % DeepCompression::window_size] = ProposeCandidate();
            
            // check candidate_window
            vector<int>::iterator it = this->candidate_window.begin();
            while (it != this->candidate_window.end()) {
                std::cout << *it++ << " ";
            }
            std::cout << std::endl;
        }
                        
        if ((step - 1) % prune_interval == 0) {                        
            const vector<int> column_to_prune = GetColToPrune(); //use_score_decay ? GetColToPrune() : GetMostFrequent(this->candidate_window, num_col);    
            std::cout << "column to prune：" << column_to_prune[0] << std::endl;
            for (int j = 0; j < column_to_prune.size(); ++j) {
                for (int i = 0; i < num_filter; ++i) {
                    this->masks_[i * num_col + column_to_prune[j]] =  0;
                }
                this->is_pruned[column_to_prune[j]] = 1;
                this->history_score[column_to_prune[j]] = -1; // set the history_score of the pruned column to -1
            }
            this->num_pruned_column += column_to_prune.size();
            DeepCompression::num_pruned_column[conv_index] = this->num_pruned_column;        
            std::cout << layer_name << " masks updated" << std::endl;
        }
        
    } else if (prune_what == "prune_weight") {    
        if (this->num_pruned_weight >= PruneRate * count) { return; }
        Dtype min = 1e20;
        Dtype min_index = 0;
        const Dtype* weight = this->blobs_[0]->cpu_data();
        for (int i = 0; i < count; ++i) {
            if (this->masks_[i] == 0) { continue; }
            if (fabs(weight[i]) < min) {
                min = fabs(weight[i]);
                min_index = i;
            }
        }
        this->masks_[min_index] = 0;
        ++ (this->num_pruned_weight);
        
        std::cout << layer_name << " masks updated" << std::endl;        
    }
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
    // Initialize CUDA streams and cuDNN.
    stream_ = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
    handle_ = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];

    // Initialize algorithm arrays
    fwd_algo_ = new cudnnConvolutionFwdAlgo_t[bottom.size()];
    bwd_filter_algo_ = new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
    bwd_data_algo_ = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];

    // initialize size arrays
    workspace_fwd_sizes_ = new size_t[bottom.size()];
    workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
    workspace_bwd_data_sizes_ = new size_t[bottom.size()];

    // workspace data
    workspaceSizeInBytes = 0;
    workspaceData = NULL;
    workspace = new void *[this->group_ * CUDNN_STREAMS_PER_GROUP];

    for (size_t i = 0; i < bottom.size(); ++i) {
        // initialize all to default algorithms
        fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t) 0;
        bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t) 0;
        bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t) 0;
        // default algorithms don't require workspace
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
    }

    for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
        CUDA_CHECK(cudaStreamCreate(&stream_[g]));
        CUDNN_CHECK(cudnnCreate(&handle_[g]));
        CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
        workspace[g] = NULL;
    }

    // Set the indexing parameters.
    bias_offset_ = (this->num_output_ / this->group_);

    // Create filter descriptor.
    const int *kernel_shape_data = this->kernel_shape_.cpu_data();
    const int kernel_h = kernel_shape_data[0];
    const int kernel_w = kernel_shape_data[1];
    cudnn::createFilterDesc<Dtype>(&filter_desc_,
                                   this->num_output_ / this->group_, this->channels_ / this->group_,
                                   kernel_h, kernel_w);

    // Create tensor descriptor(s) for data and corresponding convolution(s).
    for (int i = 0; i < bottom.size(); i++) {
        cudnnTensorDescriptor_t bottom_desc;
        cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
        bottom_descs_.push_back(bottom_desc);
        cudnnTensorDescriptor_t top_desc;
        cudnn::createTensor4dDesc<Dtype>(&top_desc);
        top_descs_.push_back(top_desc);
        cudnnConvolutionDescriptor_t conv_desc;
        cudnn::createConvolutionDesc<Dtype>(&conv_desc);
        conv_descs_.push_back(conv_desc);
    }

    // Tensor descriptor for bias.
    if (this->bias_term_) {
        cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
    }

    handles_setup_ = true;

    // Initialize, WANGHUAN
    const int count = this->blobs_[0]->count();
    const int num_filter = this->blobs_[0]->shape()[0];
    const int num_col = count / num_filter;
    this->masks_.resize(count, 1);
    this->is_pruned.resize(num_col, 0);
    this->history_prob.resize(num_col, 0.999);
    this->num_pruned_column = 0;
    this->num_pruned_weight = 0;
    this->candidate_window.resize(DeepCompression::window_size, -1);
    
    this->drop_column_masks.resize(count, 1);
    
    this->history_score.resize(num_col, 0);
    this->history_diff.resize(count, 0);
    this->history_weight.resize(count, 0);

    this->blobs_[0]->mutable_cpu_second_diff = new Dtype[count];
    for (int i = 0; i < count; ++i) {
        this->blobs_[0]->mutable_cpu_second_diff[i] = 0;
    }
    
    string layer_name = this->layer_param_.name();
    const int conv_index = layer_name[4] - '1';
    float PruneRate = DeepCompression::PruneRate[conv_index];  
    if (num_col * PruneRate > DeepCompression::max_num_column_to_prune) {
        DeepCompression::max_num_column_to_prune = num_col * PruneRate;
    }
    cout << "=== Masks etc. have been initialized" << endl;


}
template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  ConvolutionLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(2, this->num_spatial_axes_)
      << "CuDNNConvolution input must have 2 spatial axes "
      << "(e.g., height and width). "
      << "Use 'engine: CAFFE' for general ND convolution.";
  bottom_offset_ = this->bottom_dim_ / this->group_;
  top_offset_ = this->top_dim_ / this->group_;
  const int height = bottom[0]->shape(this->channel_axis_ + 1);
  const int width = bottom[0]->shape(this->channel_axis_ + 2);
  const int height_out = top[0]->shape(this->channel_axis_ + 1);
  const int width_out = top[0]->shape(this->channel_axis_ + 2);
  const int* pad_data = this->pad_.cpu_data();
  const int pad_h = pad_data[0];
  const int pad_w = pad_data[1];
  const int* stride_data = this->stride_.cpu_data();
  const int stride_h = stride_data[0];
  const int stride_w = stride_data[1];

  // Specify workspace limit for kernels directly until we have a
  // planning strategy and a rewrite of Caffe's GPU memory mangagement
  size_t workspace_limit_bytes = 8*1024*1024;

  for (int i = 0; i < bottom.size(); i++) {
    cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
        this->num_,
        this->channels_ / this->group_, height, width,
        this->channels_ * height * width,
        height * width, width, 1);
    cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
        this->num_,
        this->num_output_ / this->group_, height_out, width_out,
        this->num_output_ * this->out_spatial_dim_,
        this->out_spatial_dim_, width_out, 1);
    cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
        filter_desc_, pad_h, pad_w,
        stride_h, stride_w);

    // choose forward and backward algorithms + workspace(s)
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
      workspace_limit_bytes,
      &fwd_algo_[i]));

    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
      bottom_descs_[i],
      filter_desc_,
      conv_descs_[i],
      top_descs_[i],
      fwd_algo_[i],
      &(workspace_fwd_sizes_[i])));

    // choose backward algorithm for filter
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
          workspace_limit_bytes, &bwd_filter_algo_[i]) );

    // get workspace for backwards filter algorithm
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
          bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
          bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));

    // choose backward algo for data
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
        workspace_limit_bytes, &bwd_data_algo_[i]));

    // get workspace size
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
          filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
          bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]) );
  }

  // reduce over all workspace sizes to get a maximum to allocate / reallocate
  size_t total_workspace_fwd = 0;
  size_t total_workspace_bwd_data = 0;
  size_t total_workspace_bwd_filter = 0;

  for (size_t i = 0; i < bottom.size(); i++) {
    total_workspace_fwd        = std::max(total_workspace_fwd,
                                     workspace_fwd_sizes_[i]);
    total_workspace_bwd_data   = std::max(total_workspace_bwd_data,
                                     workspace_bwd_data_sizes_[i]);
    total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
                                     workspace_bwd_filter_sizes_[i]);
  }
  // get max over all operations
  size_t max_workspace = std::max(total_workspace_fwd,
                             total_workspace_bwd_data);
  max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
  // ensure all groups have enough workspace
  size_t total_max_workspace = max_workspace *
                               (this->group_ * CUDNN_STREAMS_PER_GROUP);

  // this is the total amount of storage needed over all groups + streams
  if (total_max_workspace > workspaceSizeInBytes) {
    DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
    workspaceSizeInBytes = total_max_workspace;

    // free the existing workspace and allocate a new (larger) one
    cudaFree(this->workspaceData);

    cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
    if (err != cudaSuccess) {
      // force zero memory path
      for (int i = 0; i < bottom.size(); i++) {
        workspace_fwd_sizes_[i] = 0;
        workspace_bwd_filter_sizes_[i] = 0;
        workspace_bwd_data_sizes_[i] = 0;
        fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
        bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
      }

      // NULL out all workspace pointers
      for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
        workspace[g] = NULL;
      }
      // NULL out underlying data
      workspaceData = NULL;
      workspaceSizeInBytes = 0;
    }

    // if we succeed in the allocation, set pointer aliases for workspaces
    for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
      workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
    }
  }

  // Tensor descriptor for bias.
  if (this->bias_term_) {
    cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
        1, this->num_output_ / this->group_, 1, 1);
  }
}

template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() {
  // Check that handles have been setup before destroying.
  if (!handles_setup_) { return; }

  for (int i = 0; i < bottom_descs_.size(); i++) {
    cudnnDestroyTensorDescriptor(bottom_descs_[i]);
    cudnnDestroyTensorDescriptor(top_descs_[i]);
    cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
  }
  if (this->bias_term_) {
    cudnnDestroyTensorDescriptor(bias_desc_);
  }
  cudnnDestroyFilterDescriptor(filter_desc_);

  for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
    cudaStreamDestroy(stream_[g]);
    cudnnDestroy(handle_[g]);
  }

  cudaFree(workspaceData);
  delete [] stream_;
  delete [] handle_;
  delete [] fwd_algo_;
  delete [] bwd_filter_algo_;
  delete [] bwd_data_algo_;
  delete [] workspace_fwd_sizes_;
  delete [] workspace_bwd_data_sizes_;
  delete [] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);

}   // namespace caffe
#endif
