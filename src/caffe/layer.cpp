#include <boost/thread.hpp>
#include <boost/thread.hpp>
#include "caffe/layer.hpp"


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


  INSTANTIATE_CLASS(Layer);

}  // namespace caffe
