#include <vector>
#include "caffe/layers/conv_layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"

using namespace std;
namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    this->PruneForward(); /// @mingsuntse, for pruning
    const Dtype* weight = this->blobs_[0]->gpu_data();
    for (int i = 0; i < bottom.size(); ++i) {
        const Dtype* bottom_data    = bottom[i]->gpu_data();
        Dtype* top_data    = top[i]->mutable_gpu_data();
        for (int n = 0; n < this->num_; ++n) {
            this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, weight, top_data + n * this->top_dim_);
            if (this->bias_term_) {
                const Dtype* bias = this->blobs_[1]->gpu_data();
                this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
            }
        }
    }
    /// Print feature map to check --------
    /// If row 3 and 8 are pruned in previous layer, then channel 3 and 8 will be only biases in this layer's feature map.
    /**
    if (!APP<Dtype>::IN_TEST && L == 0) {
        cout << "bottom.size(): " << bottom.size() << endl;
        for (int i = 0; i < bottom.size(); ++i) {
            const Dtype* top_data = top[i]->cpu_data();
            const int channel = top[i]->shape()[1];
            const int width   = top[i]->shape()[2];
            const int height  = top[i]->shape()[3];
            cout << "channel: " << channel << " " << width << " " <<  height << endl;
            
            vector<Dtype> sum(channel, 0);
            for (int c = 0; c < channel; ++c) {
                for (int w = 0 ; w < width; ++w) {
                    for (int h = 0; h < height; ++h) {
                        sum[c] += fabs(top_data[0 + c * width * height + w * height + h]);
                    }
                }
            }
            for (int c = 0; c < channel; ++c) {
                cout << sum[c] << "  ";
            }
            cout << endl;
        }
    }
    */
    
    // Restore weights when using ProbPrune
    if (this->IF_restore) {
        /*
        /// cout << layer_name << ": restore weights! " << endl;
        Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
        /// this->blobs_[0]->gpu_data(); 
        /// Interesting! If the above line is added, something like "control" seems to transfer from cpu to gpu. 
        /// Then modifying cpu weights won't affect their gpu counterparts.
        for (int i = 0; i < count; ++i) {
            muweight[i] = this->weight_backup[i];
        }
        */
        caffe_gpu_memcpy(this->blobs_[0]->count(),
                         this->blobs_backup_[0]->gpu_data(),
                         this->blobs_[0]->mutable_gpu_data());
    }
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  
  caffe_gpu_mul(this->blobs_[0]->count(),
                this->blobs_[0]->gpu_data(),
                this->blobs_[0]->gpu_data(),
                this->blobs_[0]->mutable_gpu_secdata()); // w^2
  
  const Dtype* weight    = this->blobs_[0]->gpu_data();
  const Dtype* secweight = this->blobs_[0]->gpu_secdata();
  Dtype* weight_diff    = this->blobs_[0]->mutable_gpu_diff();
  Dtype* weight_secdiff = this->blobs_[0]->mutable_gpu_secdiff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff    = top[i]->gpu_diff();
    const Dtype* top_secdiff = top[i]->gpu_secdiff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff    = this->blobs_[1]->mutable_gpu_diff();
      Dtype* bias_secdiff = this->blobs_[1]->mutable_gpu_secdiff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff,    top_diff    + n * this->top_dim_);
        this->backward_gpu_bias(bias_secdiff, top_secdiff + n * this->top_dim_); // TODO(mingsuntse): check this, maybe wrong.
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      
      caffe_gpu_mul(bottom[i]->count(),
                    bottom_data,
                    bottom_data,
                    bottom[i]->mutable_gpu_secdata()); // x^2
      const Dtype* bottom_secdata = bottom[i]->gpu_secdata();
      
      Dtype* bottom_diff    = bottom[i]->mutable_gpu_diff();
      Dtype* bottom_secdiff = bottom[i]->mutable_gpu_secdiff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data    + n * this->bottom_dim_, top_diff    + n * this->top_dim_, weight_diff);
          this->weight_gpu_gemm(bottom_secdata + n * this->bottom_dim_, top_secdiff + n * this->top_dim_, weight_secdiff); /// Added by @mingsuntse
        }
        
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff    + n * this->top_dim_, weight,    bottom_diff    + n * this->bottom_dim_);
          this->backward_gpu_gemm(top_secdiff + n * this->top_dim_, secweight, bottom_secdiff + n * this->bottom_dim_); /// Added by @mingsuntse
        } 
      }
    }
  }
  this->PruneBackward(top); /// @mingsuntse, for pruning
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionLayer);

}  // namespace caffe
