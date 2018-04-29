#include <boost/thread.hpp>
#include <boost/thread.hpp>
#include "caffe/layer.hpp"
#include "caffe/adaptive_probabilistic_pruning.hpp"


namespace caffe {

template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}


/*
 * 聚类的目的是得到各个weight属于的中心以及这个中心的值
 */
template<typename Dtype>
void Layer<Dtype>::kmeans_cluster(vector<int> &cLabel, vector<Dtype> &cCentro, Dtype *cWeights, int nWeights, \
  vector<bool> &mask, int nCluster,  int max_iter){

  Dtype maxWeight = std::numeric_limits<Dtype>::min();
  Dtype minWeight = std::numeric_limits<Dtype>::max();

  //find min max
  for(int k = 0; k < nWeights; ++k){
    if(mask[k]){
      if(cWeights[k] > maxWeight)
        maxWeight = cWeights[k];
      if(cWeights[k] < minWeight)
        minWeight = cWeights[k];
    }
    cLabel[k] = -1; //initialize all label to -1
  }

  // generate initial centroids linearly
  for (int k = 0; k < nCluster; k++)
    cCentro[k] = minWeight + (maxWeight - minWeight) * k / (nCluster - 1);


  Dtype *ptrC = new Dtype[nCluster]; // 用来存各个中心的权重的和，和下面的相除得到各个中心新的值
  int *ptrS = new int[nCluster]; // 用来存各个中心的权重的个数
  int iter = 0;
  double mCurDistance = 0.0;
  double mPreDistance = std::numeric_limits<double>::max();
  Dtype distance, mindistance; // 临时变量，用于存每个weight到中心的距离和最短距离

  // clustering
  while (iter < max_iter)
  {
    // check convergence
    if (fabs(mPreDistance - mCurDistance) / mPreDistance < 0.01)  break;
    mPreDistance = mCurDistance; mCurDistance = 0.0;
    for (int n = 0; n < nWeights; n++){
      if (!mask[n])  continue; // 如果mask为0，则跳过这个值
      mindistance = std::numeric_limits<Dtype>::max();
      for (int k = 0; k < nCluster; k++){
        distance = fabs(cWeights[n] - cCentro[k]);
        if (distance < mindistance){
          mindistance = distance;
          cLabel[n] = k;
        }
      }
      mCurDistance = mCurDistance + mindistance;
    }

    // 更新中心，初始化为0
    for (int k = 0; k < nCluster; k++){
      ptrC[k] = 0.f;
      ptrS[k] = 0;
    }

    for (int n = 0; n < nWeights; n++){
      if (mask[n]){
        ptrC[cLabel[n]] += cWeights[n];
        ptrS[cLabel[n]] += 1;
      }
    }

    for (int k = 0; k < nCluster; k++)
      cCentro[k] = ptrC[k]/ptrS[k]; // 更新中心值

    iter++;
  }
  delete[] ptrC; delete[] ptrS;
}

template<typename Dtype>
void Layer<Dtype>::IF_prune_finished() {
    const string layer_name = this->layer_param_.name();
    if (APP<Dtype>::layer_index.count(layer_name) != 0) {
        const int L = APP<Dtype>::layer_index[layer_name];
        if (APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
            Dtype pruned_ratio = 0;
            if (APP<Dtype>::prune_unit == "Weight")   { pruned_ratio = APP<Dtype>::pruned_ratio[L];     }
            else if (APP<Dtype>::prune_unit == "Row") { pruned_ratio = APP<Dtype>::pruned_ratio_row[L]; }
            else if (APP<Dtype>::prune_unit == "Col") { pruned_ratio = APP<Dtype>::pruned_ratio_col[L]; }
            const bool layer_finish     = pruned_ratio >= APP<Dtype>::prune_ratio[L]; /// layer pruning target achieved
            const bool net_finish_speed = APP<Dtype>::IF_speedup_achieved;   /// net pruning target of speed achieved
            const bool net_finish_param = APP<Dtype>::IF_compRatio_achieved; /// net pruning target of compression achieved
            
            if (layer_finish || net_finish_speed || net_finish_param) {
                APP<Dtype>::iter_prune_finished[L] = APP<Dtype>::step_ - 1;
                // if (APP<Dtype>::prune_coremthd.substr(0, 2) == "PP") { CleanWorkForPP(); } // last time, do some clean work
                
                // print when finished
                char rlayer[10], rrow[10], rcol[10];
                sprintf(rlayer, "%6.4f", APP<Dtype>::pruned_ratio[L]);
                sprintf(rrow,   "%6.4f", APP<Dtype>::pruned_ratio_row[L]);
                sprintf(rcol,   "%6.4f", APP<Dtype>::pruned_ratio_col[L]);
                std::cout << layer_name << " prune finished!" 
                     << "  step: " << APP<Dtype>::step_
                     << "  net speedup: " << APP<Dtype>::speedup
                     << "  net compRatio: " << APP<Dtype>::compRatio
                     << "  pruned_ratio: " << rlayer
                     << "  pruned_ratio_row: " << rrow
                     << "  pruned_ratio_col: " << rcol 
                     << "  prune_ratio: " << APP<Dtype>::prune_ratio[L] << std::endl;
                
                // IF_all_layer_prune_finished
                APP<Dtype>::IF_alpf = true;
                for (int i = 0; i < APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt; ++i) {
                    if (APP<Dtype>::iter_prune_finished[i] == INT_MAX) {
                        APP<Dtype>::IF_alpf = false;
                        break;
                    }
                }
            }
        }
    }
}

  INSTANTIATE_CLASS(Layer);

}  // namespace caffe
