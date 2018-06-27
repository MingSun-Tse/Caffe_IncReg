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

template<typename Dtype>
void Layer<Dtype>::kmeans_cluster(vector<int> &cLabel, vector<Dtype> &cCentro, Dtype *cWeights, int nWeights, \
                                  vector<bool> &mask, int nCluster,  int max_iter) {

  Dtype maxWeight = std::numeric_limits<Dtype>::min();
  Dtype minWeight = std::numeric_limits<Dtype>::max();

  //find min max
  for(int k = 0; k < nWeights; ++k) {
    if(mask[k]) {
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

  Dtype *ptrC = new Dtype[nCluster];
  int *ptrS = new int[nCluster];
  int iter = 0;
  double mCurDistance = 0.0;
  double mPreDistance = std::numeric_limits<double>::max();
  Dtype distance, mindistance;

  // clustering
  while (iter < max_iter) {
    // check convergence
    if (fabs(mPreDistance - mCurDistance) / mPreDistance < 0.01)  break;
    mPreDistance = mCurDistance;
    mCurDistance = 0.0;
    for (int n = 0; n < nWeights; n++) {
      if (!mask[n])  continue;
      mindistance = std::numeric_limits<Dtype>::max();
      for (int k = 0; k < nCluster; k++) {
        distance = fabs(cWeights[n] - cCentro[k]);
        if (distance < mindistance) {
          mindistance = distance;
          cLabel[n] = k;
        }
      }
      mCurDistance = mCurDistance + mindistance;
    }

    for (int k = 0; k < nCluster; k++) {
      ptrC[k] = 0.f;
      ptrS[k] = 0;
    }

    for (int n = 0; n < nWeights; n++) {
      if (mask[n]) {
        ptrC[cLabel[n]] += cWeights[n];
        ptrS[cLabel[n]] += 1;
      }
    }

    for (int k = 0; k < nCluster; k++)
      cCentro[k] = ptrC[k]/ptrS[k];

    iter++;
  }
  delete[] ptrC;
  delete[] ptrS;
}

template<typename Dtype>
void Layer<Dtype>::IF_layer_prune_finished() {
  const string layer_name = this->layer_param_.name();
  if (APP<Dtype>::layer_index.count(layer_name) != 0) {
    const int L = APP<Dtype>::layer_index[layer_name];
    if (APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
      const bool layer_finish     = APP<Dtype>::pruned_ratio_for_comparison[L] >= APP<Dtype>::current_prune_ratio[L]; // layer pruning target achieved
      const bool net_finish_speed = APP<Dtype>::IF_speedup_achieved;   // net pruning target of speed achieved
      const bool net_finish_param = APP<Dtype>::IF_compRatio_achieved; // net pruning target of compression achieved

      if (layer_finish || net_finish_speed || net_finish_param) {
        APP<Dtype>::iter_prune_finished[L] = APP<Dtype>::step_ - 1;
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
                  << "  current prune_ratio: " << APP<Dtype>::current_prune_ratio[L] << std::endl;

        APP<Dtype>::IF_current_target_achieved = true;
        for (int i = 0; i < APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt; ++i) {
          if (APP<Dtype>::iter_prune_finished[i] == INT_MAX) {
            APP<Dtype>::IF_current_target_achieved = false;
            break;
          }
        }
        if (APP<Dtype>::IF_current_target_achieved) {
          APP<Dtype>::stage_iter_prune_finished = APP<Dtype>::step_ - 1;
        }
      }
    }
  }
}

template <typename Dtype>
bool Layer<Dtype>::IF_hppf() {
  // IF_higher_priority_prune_finished
  bool IF_hppf = true;
  const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
  for (int i = 0; i < APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt; ++i) {
    if (APP<Dtype>::priority[i] < APP<Dtype>::priority[L] && APP<Dtype>::iter_prune_finished[i] == INT_MAX) {
      IF_hppf = false;
      break;
    }
  }
  return IF_hppf;
}

template <typename Dtype>
void Layer<Dtype>::UpdateNumPrunedRow() {
  const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
  const int num_col = this->blobs_[0]->count(1);

  cout << "        " << this->layer_param_.name() << " in UpdateNumPrunedRow" << endl;
  vector<int>::iterator it;
  for (it = APP<Dtype>::rows_to_prune[L].begin(); it != APP<Dtype>::rows_to_prune[L].end(); ++it) {
    caffe_gpu_set(num_col, (Dtype)0, this->blobs_[0]->mutable_gpu_data() + *it * num_col);
    caffe_gpu_set(num_col, (Dtype)0, this->masks_[0]->mutable_gpu_data() + *it * num_col);
    APP<Dtype>::IF_row_pruned[L][*it] = true;
    cout << " " << this->layer_param_.name() << " prune a row successfully: " << (*it) << endl;
  }
  APP<Dtype>::num_pruned_row[L] += APP<Dtype>::rows_to_prune[L].size();
  APP<Dtype>::rows_to_prune[L].clear();
}

template <typename Dtype>
void Layer<Dtype>::UpdateNumPrunedCol() {
  const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
  const int count   = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  const int num_chl = this->blobs_[0]->shape()[1];
  const int num_row_per_g = num_row / APP<Dtype>::group[L];
  const int filter_spatial_size = this->blobs_[0]->count(2);

  cout << "        " << this->layer_param_.name() << " in UpdateNumPrunedCol" << endl;
  vector<int>::iterator it;
  for (it = APP<Dtype>::pruned_rows[L-1].begin(); it != APP<Dtype>::pruned_rows[L-1].end(); ++it) {
    const int chl = *it % num_chl;
    const int g   = *it / num_chl;
    for (int i = g * num_row_per_g; i < (g + 1) * num_row_per_g; ++i) {
      for (int j = chl * filter_spatial_size; j < (chl + 1) * filter_spatial_size; ++j) {
        this->masks_[0]->mutable_cpu_data()[i * num_col + j] = 0;
        APP<Dtype>::IF_col_pruned[L][j][g] = true;
      }
    }
    APP<Dtype>::num_pruned_col[L] += filter_spatial_size * 1.0 / APP<Dtype>::group[L];
    cout << "  " << this->layer_param_.name() << " prune a channel successfully: " << chl << endl;
  }
  APP<Dtype>::pruned_rows[L-1].clear();
}

template <typename Dtype>
void Layer<Dtype>::UpdatePrunedRatio() {
  const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
  const int count   = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  const int group = APP<Dtype>::group[L];
  const Dtype* weight = this->blobs_[0]->cpu_data();

  if (APP<Dtype>::prune_unit == "Weight") {
    for (int i = 0; i < num_row; ++i) { // Row
      if (APP<Dtype>::IF_row_pruned[L][i]) {
        continue;
      }
      bool IF_whole_row_pruned = true;
      for (int j = 0; j < num_col; ++j) {
        if (!weight[i * num_col + j]) {
          IF_whole_row_pruned = false;
          break;
        }
      }
      if (IF_whole_row_pruned) {
        APP<Dtype>::IF_row_pruned[L][i] = true;
        APP<Dtype>::num_pruned_row[L] += 1;
      }
    }
    for (int j = 0; j < num_col; ++j) { // Column
      if (APP<Dtype>::IF_col_pruned[L][j][0]) {
        continue;
      }
      bool IF_whole_col_pruned = true;
      for (int i = 0; i < num_row; ++i) {
        if (!weight[i * num_col + j]) {
          IF_whole_col_pruned = false;
          break;
        }
      }
      if (IF_whole_col_pruned) {
        for (int g = 0; g < group; ++g) {
          APP<Dtype>::IF_col_pruned[L][j][g] = true;
        }
        APP<Dtype>::num_pruned_col[L] += 1;
      }
    }
  }

  APP<Dtype>::pruned_ratio_col[L] = APP<Dtype>::num_pruned_col[L] / num_col;
  APP<Dtype>::pruned_ratio_row[L] = APP<Dtype>::num_pruned_row[L] * 1.0 / num_row;
  APP<Dtype>::pruned_ratio_for_comparison[L] = APP<Dtype>::pruned_ratio_col[L];
  if (APP<Dtype>::prune_unit == "Weight") {
    APP<Dtype>::pruned_ratio[L] = APP<Dtype>::num_pruned_weight[L] * 1.0 / count;
    APP<Dtype>::pruned_ratio_for_comparison[L] = APP<Dtype>::pruned_ratio[L];
  } else {
    APP<Dtype>::pruned_ratio[L] = (APP<Dtype>::pruned_ratio_col[L] + APP<Dtype>::pruned_ratio_row[L])
                                  - APP<Dtype>::pruned_ratio_col[L] * APP<Dtype>::pruned_ratio_row[L];
    if (APP<Dtype>::prune_unit == "Row") {
      APP<Dtype>::pruned_ratio_for_comparison[L] = APP<Dtype>::pruned_ratio_row[L];
    }
  }
}

template <typename Dtype>
void Layer<Dtype>::Print(char mode) {
  assert(mode == 'f' || mode = 'b'); /// forward, backward
  const string layer_name = this->layer_param_.name();
  const int num_col = this->blobs_[0]->count() / this->blobs_[0]->shape()[0];
  const int num_row = this->blobs_[0]->shape()[0];
  const Dtype* w = this->blobs_[0]->cpu_data();
  const Dtype* d = this->blobs_[0]->cpu_diff();
  const Dtype* m = this->masks_[0]->cpu_data();
  // print Index, blob, Mask
  cout.width(5);
  cout << "Index" << "   ";
  const string blob = (mode == 'f') ? "WeightBeforeMasked" : "DiffBeforeMasked";
  cout.width(blob.size());
  cout << blob << "   ";
  cout.width(4);
  cout << "Mask" << "   ";
  // print additional info
  string info = "";
  if (APP<Dtype>::prune_coremthd.substr(0, 2) == "PP") {
    info = "HistoryProb";
  } else if (APP<Dtype>::prune_coremthd.substr(0, 2) == "Reg") {
    info = "HistoryReg";
  } else {
    info = "WeightBeforeMasked";
  }
  Dtype* info_data = NULL;
  if (APP<Dtype>::prune_method.substr(0, 2) == "PP" || APP<Dtype>::prune_method.substr(0, 3) == "Reg") {
    info_data = this->history_punish_[0]->mutable_cpu_data();
  } else {
    info_data = this->blobs_[0]->mutable_cpu_data();
  }
  cout.width(info.size());
  cout << info << " - " << this->layer_param_.name() << endl;

  if (APP<Dtype>::prune_unit == "Row") {
    const int show_num = APP<Dtype>::show_num_weight > num_row ? num_row : APP<Dtype>::show_num_weight;
    for (int i = 0; i < show_num; ++i) {
      // print Index
      cout.width(3);
      cout << "r";
      cout.width(2);
      cout << i+1 << "   ";
      // print blob
      Dtype sum_w = 0, sum_d = 0;
      for (int j = 0; j < num_col; ++j) {
        sum_w += fabs(w[i * num_col + j]);
        sum_d += fabs(d[i * num_col + j]);
      }
      sum_w /= num_col; /// average abs weight
      sum_d /= num_col; /// average abs diff
      const Dtype s = mode == 'f' ? sum_w : sum_d;
      cout.width(blob.size());
      cout << s << "   ";
      // print Mask
      cout.width(4);
      cout << m[i * num_col] << "   ";
      // print info
      cout.width(info.size());
      cout << info_data[i * num_col] << endl;
    }
  } else if (APP<Dtype>::prune_unit == "Col") {
    const int show_num = APP<Dtype>::show_num_weight > num_col ? num_col : APP<Dtype>::show_num_weight;
    for (int j = 0; j < show_num; ++j) {
      // print Index
      cout.width(3);
      cout << "c";
      cout.width(2);
      cout << j+1 << "   ";
      // print blob
      Dtype sum_w = 0, sum_d = 0;
      for (int i = 0; i < num_row; ++i) {
        sum_w += fabs(w[i * num_col + j]);
        sum_d += fabs(d[i * num_col + j]);
      }
      sum_w /= num_row; /// average abs weight
      sum_d /= num_row; /// average abs diff
      const Dtype s = mode == 'f' ? sum_w : sum_d;
      cout.width(blob.size());
      cout << s << "   ";
      // print Mask
      cout.width(4);
      cout << m[j] << "   ";
      // print info
      cout.width(info.size());
      cout << info_data[j] << endl;
    }
  } else if (APP<Dtype>::prune_unit == "Weight") {
    for (int i = 0; i < APP<Dtype>::show_num_weight; ++i) {
      // print Index
      cout.width(3);
      cout << "w";
      cout.width(2);
      cout << i+1 << "   ";
      const Dtype s = mode == 'f' ? fabs(w[i]) : fabs(d[i]);
      cout.width(blob.size());
      cout << s << "   ";
      // print Mask
      cout.width(4);
      cout << m[i] << "   ";
      // print info
      cout.width(info.size());
      cout << info_data[i] << endl;
    }
  }
}

template <typename Dtype>
void Layer<Dtype>::TaylorPrune(const vector<Blob<Dtype>*>& top) {
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_data = top[i]->cpu_data();
    const Dtype* top_diff = top[i]->cpu_diff();
    Dtype* mumasks = this->masks_[0]->mutable_cpu_data();
    const int num_  = top[i]->shape()[0];
    const int num_c = top[i]->shape()[1]; /// channel
    const int num_h = strcmp(this->type(), "InnerProduct") == 0 ? 1 : top[i]->shape()[2]; /// height
    const int num_w = strcmp(this->type(), "InnerProduct") == 0 ? 1 : top[i]->shape()[3]; /// width
    const int count = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];

    typedef std::pair<Dtype, int> mypair;
    vector<mypair> fm_score(num_c); /// feature map score
    for (int c = 0; c < num_c; ++c) {
      fm_score[c].second = c;
      fm_score[c].first  = 0;
    }
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < num_c; ++c) {
        for (int h = 0; h < num_h; ++h) {
          for (int w = 0; w < num_w; ++w) {
            fm_score[c].first += fabs(top_diff[((n * num_c + c) * num_h + h) * num_w + w]
                                    * top_data[((n * num_c + c) * num_h + h) * num_w + w]);
          }
        }
      }
    }
    for (int c = 0; c < num_c; ++c) {
      if (APP<Dtype>::IF_row_pruned[L][c]) {
        fm_score[c].first = INT_MAX;
      }
    }
    sort(fm_score.begin(), fm_score.end());
    const int num_once_prune = ceil(num_row * APP<Dtype>::ratio_once_prune);
    for (int i = 0; i < num_once_prune; ++i) {
      const int c = fm_score[i].second;
      for (int j = 0; j < num_col; ++j) {
        mumasks[c * num_col + j] = 0;
      }
      APP<Dtype>::IF_row_pruned[L][c] = true;
      ++ APP<Dtype>::num_pruned_row[L];
      APP<Dtype>::pruned_rows[L].push_back(c);
    }
  }
}

template <typename Dtype>
void Layer<Dtype>::FilterPrune() {
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  Dtype* mumasks = this->masks_[0]->mutable_cpu_data();
  const int count = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  const int L = APP<Dtype>::layer_index[this->layer_param_.name()];

  typedef pair<Dtype, int> mypair;
  vector<mypair> row_score(num_row);
  for (int i = 0; i < num_row; ++i) {
    row_score[i].second = i; // index
    if (APP<Dtype>::IF_row_pruned[L][i]) {
      row_score[i].first = INT_MAX; // make those pruned row "float" up
      continue;
    }
    row_score[i].first  = 0;
    for (int j = 0; j < num_col; ++j) {
      row_score[i].first += fabs(muweight[i * num_col +j]);
    }
  }
  sort(row_score.begin(), row_score.end());
  const int num_once_prune = ceil(num_row * APP<Dtype>::ratio_once_prune);
  for (int i = 0; i < num_once_prune; ++i) {
    const int r = row_score[i].second;
    for (int j = 0; j < num_col; ++j) {
      mumasks[r * num_col + j] = 0;
    }
    APP<Dtype>::IF_row_pruned[L][r] = true;
    ++ APP<Dtype>::num_pruned_row[L];
    APP<Dtype>::pruned_rows[L].push_back(r);
  }
}

template <typename Dtype>
void Layer<Dtype>::ProbPruneCol(const int& prune_interval) {
  Dtype* muhistory_score = this->history_score_[0]->mutable_cpu_data();
  Dtype* muhistory_punish = this->history_punish_[0]->mutable_cpu_data();
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  const int count = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  const string layer_name = this->layer_param_.name();
  const int L = APP<Dtype>::layer_index[layer_name];
  const int num_col_to_prune_ = ceil(APP<Dtype>::prune_ratio[L] * num_col); // a little bit higher goal
  const int group = APP<Dtype>::group[L];

  /// Calculate history score
  typedef std::pair<Dtype, int> mypair;
  vector<mypair> col_score(num_col);
  for (int j = 0; j < num_col; ++j) {
    col_score[j].second = j; /// index
    Dtype score = 0;
    for (int i = 0; i < num_row; ++i) {
      score += fabs(muweight[i * num_col +j]);
    }
    muhistory_score[j] = APP<Dtype>::score_decay * muhistory_score[j] + score;
    col_score[j].first = muhistory_score[j];
    if (APP<Dtype>::IF_col_pruned[L][j][0]) {
      col_score[j].first = INT_MAX;  /// make the pruned columns "float" up
    }
  }
  sort(col_score.begin(), col_score.end());

  /// Update history_prob
  if ((APP<Dtype>::step_ - 1) % prune_interval == 0 && APP<Dtype>::inner_iter == 0) {
    /// Calculate existence probability of each weight
    const Dtype AA = APP<Dtype>::AA;
    const Dtype kk = APP<Dtype>::kk;
    const Dtype alpha = log(2/kk) / (num_col_to_prune_ - APP<Dtype>::num_pruned_col[L]);
    const Dtype N1 = -log(kk)/alpha;
    const Dtype k_ = AA / (num_col_to_prune_ - APP<Dtype>::num_pruned_col[L]); /// linear punishment

    for (int j = 0; j < num_col - APP<Dtype>::num_pruned_col[L]; ++j) {
      const int col_of_rank_j = col_score[j].second;
      Dtype delta = j < N1 ? AA * exp(-alpha * j) : -AA * exp(-alpha * (2*N1-j)) + 2*kk*AA;
      if (APP<Dtype>::prune_method == "PP-linear_Col") {
        delta = AA - k_ * j; /// linear punishment
      }
      const Dtype old_prob = 1 - muhistory_punish[col_of_rank_j];
      const Dtype new_prob = std::min(std::max(old_prob - delta, Dtype(0)), Dtype(1));
      muhistory_punish[col_of_rank_j] = 1 - new_prob;
      if (new_prob == 0) {
        APP<Dtype>::num_pruned_col[L] += 1;
        for (int g = 0; g < group; ++g) {
          APP<Dtype>::IF_col_pruned[L][col_of_rank_j][g] = true;
        }
        for (int i = 0; i < num_row; ++i) {
          muweight[i * num_col + col_of_rank_j] = 0;
        } /// once pruned, zero out weights
      }
      // Print
      if (new_prob > old_prob) {
        cout << "recover prob: " << layer_name << "-" << col_of_rank_j
             << "   old prob: " << old_prob
             << "   new prob: " << new_prob << endl;
      }
    }
  }
  this->GenerateMasks(); // With probability updated, generate masks and do pruning
}

/*
TODO(mingsuntse): check, why does ProbPruneCol_chl work so poor?
*/
template <typename Dtype>
void Layer<Dtype>::ProbPruneCol_chl(const int& prune_interval) {
  Dtype* muhistory_score = this->history_score_[0]->mutable_cpu_data();
  Dtype* muhistory_punish = this->history_punish_[0]->mutable_cpu_data();
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  const int count = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  const int num_chl = this->blobs_[0]->shape()[1];
  const int kernel_spatial_size = this->blobs_[0]->count(2);
  const string layer_name = this->layer_param_.name();
  const int L = APP<Dtype>::layer_index[layer_name];
  const int num_chl_to_prune = ceil(APP<Dtype>::prune_ratio[L] * num_chl);
  const int group = APP<Dtype>::group[L];

  /// Calculate history score
  typedef std::pair<Dtype, int> mypair;
  vector<mypair> chl_score(num_chl); // channel score
  for (int c = 0; c < num_chl; ++c) {
    chl_score[c].second = c;
    Dtype sum = 0;
    for (int i = 0; i < num_row; ++i) {
      for (int j = c * kernel_spatial_size; j < (c+1) * kernel_spatial_size; ++j) {
        sum += fabs(muweight[i * num_col +j]);
      }
    }
    muhistory_score[c] = APP<Dtype>::score_decay * muhistory_score[c] + sum;
    chl_score[c].first = muhistory_score[c];
    if (APP<Dtype>::IF_col_pruned[L][c * kernel_spatial_size][0]) {
      chl_score[c].first = INT_MAX;  /// make the pruned columns "float" up
    }
  }
  sort(chl_score.begin(), chl_score.end());

  /// Update history_prob
  if ((APP<Dtype>::step_ - 1) % prune_interval == 0 && APP<Dtype>::inner_iter == 0) {
    /// Calculate functioning probability of each weight
    const Dtype AA = APP<Dtype>::AA;
    const Dtype kk = APP<Dtype>::kk;
    const Dtype alpha = log(2/kk) / (num_chl_to_prune - APP<Dtype>::num_pruned_col[L] / kernel_spatial_size);
    const Dtype N1 = -log(kk)/alpha;
    const Dtype k_ = AA / (num_chl_to_prune - APP<Dtype>::num_pruned_col[L] / kernel_spatial_size); /// linear punishment
    for (int rk = 0; rk < (num_col - APP<Dtype>::num_pruned_col[L])/kernel_spatial_size; ++rk) {
      const int chl_of_rank_rk = chl_score[rk].second;
      Dtype delta = rk < N1 ? AA * exp(-alpha * rk) : -AA * exp(-alpha * (2*N1-rk)) + 2*kk*AA;
      if (APP<Dtype>::prune_method == "PP-chl-linear_Col") {
        delta = AA - k_ * rk; /// linear punishment
      }
      const Dtype old_prob = 1 - muhistory_punish[chl_of_rank_rk * kernel_spatial_size];
      const Dtype new_prob = std::min(std::max(old_prob - delta, Dtype(0)), Dtype(1));
      for (int j = chl_of_rank_rk * kernel_spatial_size; j < (chl_of_rank_rk + 1) * kernel_spatial_size; ++j) {
        muhistory_punish[j] = 1 - new_prob;
      }
      if (new_prob == 0) {
        APP<Dtype>::num_pruned_col[L] += kernel_spatial_size;
        for (int j = chl_of_rank_rk * kernel_spatial_size; j < (chl_of_rank_rk+1) * kernel_spatial_size; ++j) {
          for (int g = 0; g < group; ++g) {
            APP<Dtype>::IF_col_pruned[L][j][g] = true;
          }
        }
        for (int i = 0; i < num_row; ++i) {
          for (int j = chl_of_rank_rk * kernel_spatial_size; j < (chl_of_rank_rk+1) * kernel_spatial_size; ++j) {
            muweight[i * num_col + j] = 0;
          }
        } // once pruned, zero out weights
      }
      // Print
      if (new_prob > old_prob) {
        cout << "recover prob: " << layer_name << "-" << chl_of_rank_rk
             << "   old prob: " << old_prob
             << "   new prob: " << new_prob << endl;
      }
    }
  }
  this->GenerateMasks(); // With probability updated, generate masks and do pruning
}

template <typename Dtype>
void Layer<Dtype>::ProbPruneRow(const int& prune_interval) {
  Dtype* muhistory_score = this->history_score_[0]->mutable_cpu_data();
  Dtype* muhistory_punish = this->history_punish_[0]->mutable_cpu_data();
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  const int count = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  const string layer_name = this->layer_param_.name();
  const int L = APP<Dtype>::layer_index[layer_name];

  /// Calculate history score
  typedef std::pair<Dtype, int> mypair;
  vector<mypair> row_score(num_row);
  for (int i = 0; i < num_row; ++i) {
    row_score[i].second = i;
    if (APP<Dtype>::IF_row_pruned[L][i]) {
      row_score[i].first = INT_MAX; // make the pruned columns "float" up
      continue;
    }
    Dtype score = 0;
    for (int j = 0; j < num_col; ++j) {
      score += fabs(muweight[i * num_col +j]);
    }
    muhistory_score[i * num_col] = APP<Dtype>::score_decay * muhistory_score[i * num_col] + score;
    row_score[i].first = muhistory_score[i * num_col];
  }
  sort(row_score.begin(), row_score.end());

  /// Update history_prob
  if (APP<Dtype>::step_ % prune_interval == 0 && APP<Dtype>::inner_iter == 0) {
    /// Calculate functioning probability of each weight
    /// use two linear functions
    const Dtype AA = APP<Dtype>::AA;
    const int num_row_to_prune_ = ceil(APP<Dtype>::prune_ratio[L] * num_row) - APP<Dtype>::num_pruned_row[L];
    const int num_row_ = num_row - APP<Dtype>::num_pruned_row[L];
    const Dtype k1 = AA / (row_score[num_row_to_prune_].first - row_score[0].first);
    const Dtype k2 = AA / (row_score[num_row_ - 1].first - row_score[num_row_to_prune_].first);
    cout << k1 << "  " << k2 << endl;
    for (int rk = 0; rk < num_row_; ++rk) {
      const int row_of_rank_rk = row_score[rk].second;
      const Dtype Delta = (rk <= num_row_to_prune_) ? AA - k1*(row_score[rk].first - row_score[0].first)
                          : -(k2*(row_score[rk].first - row_score[num_row_to_prune_].first));
      const Dtype old_prob = 1 - muhistory_punish[row_of_rank_rk * num_col];
      const Dtype new_prob = std::min(std::max(old_prob - Delta, Dtype(0)), Dtype(1));
      muhistory_punish[row_of_rank_rk * num_col] = 1 - new_prob;
      if (new_prob == 0) {
        ++ APP<Dtype>::num_pruned_row[L];
        APP<Dtype>::IF_row_pruned[L][row_of_rank_rk] = true;
        for (int j = 0; j < num_col; ++j) {
          muweight[row_of_rank_rk * num_col + j] = 0; // once pruned, zero out weights
        }
        APP<Dtype>::pruned_rows[L].push_back(row_of_rank_rk);
      }
      // Print
      if (new_prob > old_prob) {
        cout << "recover prob: " << layer_name << "-" << row_of_rank_rk
             << "  old prob: " << old_prob
             << "  new prob: " << new_prob << endl;
      }
    }
  }
  this->GenerateMasks(); // With probability updated, generate masks and do pruning
}

template <typename Dtype>
void Layer<Dtype>::GenerateMasks() {
  Dtype* muhistory_punish = this->history_punish_[0]->mutable_cpu_data();
  Dtype* mumasks = this->masks_[0]->mutable_cpu_data();
  Dtype* mublobs_backup = this->blobs_backup_[0]->mutable_cpu_data();
  Dtype* muweight = this->blobs_[0]->mutable_cpu_data();
  const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
  const int count = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  assert(APP<Dtype>::prune_unit != "Weight"); // PP not used for element-wise pruning for now.

  if (APP<Dtype>::mask_generate_mechanism == "group-wise") {
    if (APP<Dtype>::prune_unit == "Row") {
      Dtype rands[num_row];
      caffe_rng_uniform(num_row, (Dtype)0, (Dtype)1, rands);
      for (int i = 0; i < count; ++i) {
        const int row_index = i / num_col;
        const int col_index = i % num_col;
        const bool cond1 = rands[row_index] < muhistory_punish[row_index * num_col];
        const bool cond2 = !APP<Dtype>::IF_col_pruned[L][col_index][0];
        mumasks[i] = (cond1 && cond2) ? 1 : 0;
        mublobs_backup[i] = muweight[i];
        muweight[i] *= mumasks[i];
      }
    } else {
      Dtype rands[num_col];
      caffe_rng_uniform(num_col, (Dtype)0, (Dtype)1, rands);
      for (int i = 0; i < count; ++i) {
        const int row_index = i / num_col;
        const int col_index = i % num_col;
        const bool cond1 = rands[col_index] < muhistory_punish[col_index];
        const bool cond2 = !APP<Dtype>::IF_row_pruned[L][row_index];
        mumasks[i] = (cond1 && cond2) ? 1 : 0;
        mublobs_backup[i] = muweight[i];
        muweight[i] *= mumasks[i];
      }
    }
  } else if (APP<Dtype>::mask_generate_mechanism == "element-wise") {
    Dtype rands[count/100];
    if (APP<Dtype>::prune_unit == "Row") {
      for (int i = 0; i < count; ++i) {
        if (i % (count/100) == 0) {
          caffe_rng_uniform(count/100, (Dtype)0, (Dtype)1, rands);
        }
        const int row_index = i / num_col;
        const int col_index = i % num_col;
        const bool cond1 = rands[i % (count/100)] < muhistory_punish[row_index * num_col];
        const bool cond2 = !APP<Dtype>::IF_col_pruned[L][col_index][0];
        mumasks[i] = (cond1 && cond2) ? 1 : 0;
        mublobs_backup[i] = muweight[i];
        muweight[i] *= mumasks[i];
      }
    } else {
      for (int i = 0; i < count; ++i) {
        if (i % (count/100) == 0) {
          caffe_rng_uniform(count/100, (Dtype)0, (Dtype)1, rands);
        }
        const int row_index = i / num_col;
        const int col_index = i % num_col;
        const bool cond1 = rands[i % (count/100)] < muhistory_punish[col_index];
        const bool cond2 = !APP<Dtype>::IF_row_pruned[L][row_index];
        mumasks[i] = (cond1 && cond2) ? 1 : 0;
        mublobs_backup[i] = muweight[i];
        muweight[i] *= mumasks[i];
      }
    }
  } else if (APP<Dtype>::mask_generate_mechanism == "channel-wise") {
    assert(APP<Dtype>::prune_unit != "Col"); // "channel-wise" only used for row pruning
    const int num = this->blobs_[0]->count(0, 2); // number of channels
    const int kernel_spatial_size = this->blobs_[0]->count(2);
    Dtype rands[num];
    caffe_rng_uniform(num, (Dtype)0, (Dtype)1, rands);
    for (int i = 0; i < count; ++i) {
      const int row_index = i / num_col;
      const int col_index = i % num_col;
      const int chl_index = i / kernel_spatial_size; // channel index
      const bool cond1 = rands[chl_index] < muhistory_punish[row_index * num_col];
      const bool cond2 = !APP<Dtype>::IF_col_pruned[L][col_index][0];
      mumasks[i] = (cond1 && cond2) ? 1 : 0;
      mublobs_backup[i] = muweight[i];
      muweight[i] *= mumasks[i];
    }
  } else {
    LOG(INFO) << "Wrong, unknown mask_generate_mechanism";
    exit(1);
  }
  this->IF_restore = true;
}

/*
template <typename Dtype>
void Layer<Dtype>::PruneMinimals() {
    Dtype* muweight   = this->blobs_[0]->mutable_cpu_data();
    const int count   = this->blobs_[0]->count();
    const int num_row = this->blobs_[0]->shape()[0];
    const int num_col = count / num_row;
    const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
    const int group = APP<Dtype>::group[L];

    if (APP<Dtype>::prune_unit == "Weight") {
        for (int i = 0; i < count; ++i) {
            if (APP<Dtype>::IF_weight_pruned[L][i]) { continue; }
            if (fabs(muweight[i]) < APP<Dtype>::prune_threshold || APP<Dtype>::history_reg[L][i] >= APP<Dtype>::target_reg) {
                this->masks_[0]->mutable_cpu_data()[i] = 0;

                APP<Dtype>::num_pruned_weight[L] += 1;
                APP<Dtype>::IF_weight_pruned[L][i] = true;
                APP<Dtype>::hrank[L][i]  = APP<Dtype>::step_ - 1000000 - (APP<Dtype>::history_reg[L][i] - APP<Dtype>::target_reg);
                APP<Dtype>::hhrank[L][i] = APP<Dtype>::step_ - 1000000 - (APP<Dtype>::history_reg[L][i] - APP<Dtype>::target_reg);
            }
        }
    } else if (APP<Dtype>::prune_unit == "Col") {
        for (int j = 0; j < num_col; ++j) {
            if (APP<Dtype>::IF_col_pruned[L][j][0]) { continue; }
            // bool IF_all_weights_are_small = true;
            Dtype sum = 0;
            for (int i = 0; i < num_row; ++i) {
                sum += fabs(muweight[i * num_col + j]);
            }
            sum /= num_row;
            if (sum < APP<Dtype>::prune_threshold ||  APP<Dtype>::history_reg[L][j] >= APP<Dtype>::target_reg) {
                for (int i = 0; i < num_row; ++i) {
                    this->masks_[0]->mutable_cpu_data()[i * num_col + j] = 0;
                }
                APP<Dtype>::num_pruned_col[L] += 1;
                for (int g = 0; g < group; ++g) {
                    APP<Dtype>::IF_col_pruned[L][j][g] = true;
                }
                APP<Dtype>::hrank[L][j] = APP<Dtype>::step_ - 1000000 - (APP<Dtype>::history_reg[L][j] - APP<Dtype>::target_reg);  // the worser column, the earlier pruned column will be ranked in fronter
            }
        }
    } else if (APP<Dtype>::prune_unit == "Row") {
        for (int i = 0; i < num_row; ++i) {
            if (APP<Dtype>::IF_row_pruned[L][i]) { continue; }
            Dtype sum = 0;
            for (int j = 0; j < num_col; ++j) {
                sum += fabs(muweight[i * num_col + j]);
            }
            sum /= num_col;
            if (sum < APP<Dtype>::prune_threshold ||  APP<Dtype>::history_reg[L][i] >= APP<Dtype>::target_reg) {
                for (int j = 0; j < num_col; ++j) {
                    this->masks_[0]->mutable_cpu_data()[i * num_col + j] = 0;
                }
                ++ APP<Dtype>::num_pruned_row[L];
                APP<Dtype>::IF_row_pruned[L][i] = true;
                APP<Dtype>::pruned_rows.push_back(i);
                APP<Dtype>::hrank[L][i] = APP<Dtype>::step_ - 1000000 - (APP<Dtype>::history_reg[L][i] - APP<Dtype>::target_reg);
            }
        }
    }
}
*/
template <typename Dtype>
void Layer<Dtype>::RestoreMasks() {
  const int count   = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  const Dtype *weight = this->blobs_[0]->cpu_data();
  const string layer_name = this->layer_param_.name();
  const int L = APP<Dtype>::layer_index[layer_name];
  const int group = APP<Dtype>::group[L];
  const int num_row_per_g = num_row / group;
  const string mthd = APP<Dtype>::prune_method;
  Dtype num_pruned_col = 0;
  int   num_pruned_row = 0;

  // Clear existing pruning state
  caffe_gpu_set(this->masks_[0]->count(),
                Dtype(1),
                this->masks_[0]->mutable_gpu_data());
  APP<Dtype>::num_pruned_weight[L] = 0;
  APP<Dtype>::num_pruned_col[L]    = 0;
  APP<Dtype>::num_pruned_row[L]    = 0;
  vector<bool> vec_tmp(APP<Dtype>::group[L], false);
  APP<Dtype>::IF_col_pruned[L] = vector<vector<bool> >(num_col, vec_tmp);
  APP<Dtype>::IF_row_pruned[L] = vector<bool>(num_row, false);

  if (APP<Dtype>::prune_unit == "Weight") {
    for (int i = 0; i < count; ++i) {
      if (!weight[i]) {
        this->masks_[0]->mutable_cpu_data()[i] = 0;
        ++ APP<Dtype>::num_pruned_weight[L];
      }
    }
  } else {
    for (int j = 0; j < num_col; ++j) { // Column
      for (int g = 0; g < group; ++g) {
        Dtype sum = 0;
        for (int i = g * num_row_per_g; i < (g+1) * num_row_per_g; ++i) {
          sum += fabs(weight[i * num_col + j]);
        }
        if (sum == 0) {
          num_pruned_col += 1.0 / group; /// note that num_pruned_row is always integer while num_pruned_col can be non-integer.
          APP<Dtype>::IF_col_pruned[L][j][g] = true;
          for (int i = g * num_row_per_g; i < (g+1) * num_row_per_g; ++i) {
            this->masks_[0]->mutable_cpu_data()[i * num_col + j] = 0;
          }
        }
      }
    }
    for (int i = 0; i < num_row; ++i) { // Row
      Dtype sum = 0;
      for (int j = 0; j < num_col; ++j) {
        sum += fabs(weight[i * num_col + j]);
      }
      if (sum == 0) {
        ++ num_pruned_row;
        APP<Dtype>::IF_row_pruned[L][i] = true;
        for (int j = 0; j < num_col; ++j) {
          this->masks_[0]->mutable_cpu_data()[i * num_col + j] = 0;
        }
      }
    }
    APP<Dtype>::num_pruned_col[L] = num_pruned_col;
    APP<Dtype>::num_pruned_row[L] = num_pruned_row;
  }
  this->UpdatePrunedRatio();
  if (APP<Dtype>::pruned_ratio_for_comparison[L] >= APP<Dtype>::prune_ratio[L]) {
    APP<Dtype>::iter_prune_finished[L] = -1; // TODO(mingsuntse): check multi-GPU
    cout << "layer " << L << ": " << layer_name << " prune finished." << endl;
  }

  LOG(INFO) << "  Masks restored,"
            << "  num_pruned_col=" << APP<Dtype>::num_pruned_col[L] << "(" << APP<Dtype>::num_pruned_col[L] * 1.0 / num_col << ")"
            << "  num_pruned_row=" << APP<Dtype>::num_pruned_row[L] << "(" << APP<Dtype>::num_pruned_row[L] * 1.0 / num_row << ")"
            << "  pruned_ratio="   << APP<Dtype>::pruned_ratio[L]
            << "  prune_ratio="    << APP<Dtype>::prune_ratio[L];
}

template <typename Dtype>
void Layer<Dtype>::PruneSetUp(const PruneParameter& prune_param) {
  const int count   = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  APP<Dtype>::prune_ratio.push_back(prune_param.prune_ratio());
  APP<Dtype>::prune_ratio_step.push_back(0.1); // TODO(mingsuntse-newprune)
  APP<Dtype>::current_prune_ratio.push_back(prune_param.prune_ratio_begin()); // APP<Dtype>::prune_ratio_step.back()); // Start from 20%
  APP<Dtype>::pruned_ratio.push_back(0); // used in TEST
  this->IF_masks_updated = true;
  if (this->phase_ == TEST) {
    return;
  }

  // Get layer_index
  const string layer_name = this->layer_param_.name();
  if (APP<Dtype>::layer_index.count(layer_name) == 0) {
    APP<Dtype>::layer_index[layer_name] = APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt;
    if (!strcmp(this->type(), "Convolution") or !strcmp(this->type(), "NdConvolution")) {
      ++ APP<Dtype>::conv_layer_cnt;
    } else if (!strcmp(this->type(), "InnerProduct")) {
      ++ APP<Dtype>::fc_layer_cnt;
    } else {
      LOG(FATAL) << "Seems wrong, pruning setup can ONLY be put in the layers with learnable parameters (Conv and FC), please check.";
    }
    LOG(INFO) << "New learnable layer registered: " << layer_name
              << ". Its layer index: " << APP<Dtype>::layer_index[layer_name] << endl;
  }
  const int L = APP<Dtype>::layer_index[layer_name];

  // Note: the varibales below can ONLY be used in training.
  // Note: These varibales will be called for every GPU, whereas since we use `layer_index` to index, so it doesn't matter.
  // Set up prune parameters of layer
  APP<Dtype>::IF_update_row_col_layer.push_back(prune_param.if_update_row_col());
  APP<Dtype>::IF_layer_far_from_borderline.push_back(true);
  APP<Dtype>::rows_to_prune.push_back(vector<int>());
  APP<Dtype>::pruned_rows.push_back(vector<int>());
  APP<Dtype>::pruned_ratio_col.push_back(0);
  APP<Dtype>::pruned_ratio_row.push_back(0);
  APP<Dtype>::pruned_ratio_for_comparison.push_back(0);
  APP<Dtype>::last_feasible_prune_ratio.push_back(0);
  APP<Dtype>::last_infeasible_prune_ratio.push_back(0);
  APP<Dtype>::GFLOPs.push_back(this->blobs_[0]->count()); // further calculated in `net.cpp`, after layer SetUp
  APP<Dtype>::num_param.push_back(count);
  // Pruning state
  APP<Dtype>::num_pruned_col.push_back(0);
  APP<Dtype>::num_pruned_row.push_back(0);
  APP<Dtype>::num_pruned_weight.push_back(0);
  APP<Dtype>::IF_row_pruned.push_back(vector<bool>(num_row, false));
  vector<bool> vec_tmp(APP<Dtype>::group[L], false);
  APP<Dtype>::IF_col_pruned.push_back(vector<vector<bool> >(num_col, vec_tmp));
  APP<Dtype>::IF_weight_pruned.push_back(vector<bool>(count, false));
  // Info shared among layers
  APP<Dtype>::filter_spatial_size.push_back(this->blobs_[0]->shape()[2] * this->blobs_[0]->shape()[3]); // TODO(mingsuntse): check 3D CNN, this is okay?
  APP<Dtype>::priority.push_back(prune_param.priority());
  APP<Dtype>::iter_prune_finished.push_back(INT_MAX);
  LOG(INFO) << "Pruning setup done: " << layer_name;
}

template <typename Dtype>
void Layer<Dtype>::PruneForward() {
#ifdef ShowTimingLog
  clock_t t1 = clock();
  cout << this->layer_param_.name() << ": forward GPU begins timing" << endl;
#endif
  const int count = this->blobs_[0]->count();
  const int num_row = this->blobs_[0]->shape()[0];
  const int num_col = count / num_row;
  const string layer_name = this->layer_param_.name();
  const string mthd = APP<Dtype>::prune_method;
  const int L = APP<Dtype>::layer_index[layer_name];
  this->IF_restore = false;

  /// IF_prune
  const bool IF_want_prune  = mthd != "None" && APP<Dtype>::prune_ratio[L] > 0; // if you want to prune, you must specify a meaningful prune_method and give a positive prune_ratio
  const bool IF_been_pruned = APP<Dtype>::pruned_ratio[L] > 0; // for a pruned layer, continue to prune
  const bool IF_enough_iter = APP<Dtype>::step_ >= APP<Dtype>::prune_begin_iter+1; // for a raw layer, if iter is enough, then prune
  const bool IF_prune = IF_want_prune && (IF_been_pruned || IF_enough_iter);

  if (this->phase_ == TRAIN && APP<Dtype>::inner_iter == 0) {
    // For a layer which doesn't want to prune, it still should UpdateNumPrunedCol/Row because of neighbour layer
    if (mthd != "None" && (IF_been_pruned || IF_enough_iter)) {
      if (APP<Dtype>::IF_update_row_col && APP<Dtype>::IF_update_row_col_layer[L]) {
        // Note that, UpdateNumPrunedRow/Col before pruning, so that when calculating score, the zombie weights will not be counted.
        // The last conv and last fc layer need not updating num of pruned row.
        // In fact, the last conv should be updated row and the first fc should be updated col, but for simplicity, which are ignored for now.
        if (APP<Dtype>::prune_unit == "Col"
            && L != APP<Dtype>::conv_layer_cnt - 1
            && L != APP<Dtype>::conv_layer_cnt + APP<Dtype>::fc_layer_cnt - 1 // The last conv layer and last fc layer need not update row.
            && APP<Dtype>::rows_to_prune[L].size()) {
          this->UpdateNumPrunedRow();
        } else if (APP<Dtype>::prune_unit == "Row"
                   && L != 0
                   && L != APP<Dtype>::conv_layer_cnt // The first conv layer and first fc layer need not update column.
                   && mthd != "TP_Row" // Don't update column for TP, because their method doesn't mention this.
                   && APP<Dtype>::pruned_rows[L-1].size()) {
          this->UpdateNumPrunedCol();
        }
        this->UpdatePrunedRatio();
        this->IF_layer_prune_finished();
      }
    }

    // Print and check, before update probs
    // put this outside, to print even when we do not prune
    if (APP<Dtype>::show_layer.size() >= L+1 && APP<Dtype>::show_layer[L] == '1'
        && APP<Dtype>::step_ % APP<Dtype>::show_interval == 0) {
      this->Print('f');
    }

    // Update masks
    if (IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
      if (APP<Dtype>::prune_coremthd.substr(0, 2) == "FP" && APP<Dtype>::prune_unit == "Row" && (APP<Dtype>::step_ - 1) % APP<Dtype>::prune_interval == 0) {
        this->FilterPrune();
      } else if (mthd == "PP_Col" && this->IF_hppf()) {
        this->ProbPruneCol(APP<Dtype>::prune_interval);
      } else if (mthd == "PP_Row" && this->IF_hppf()) {
        this->ProbPruneRow(APP<Dtype>::prune_interval);
      } else if ((mthd == "PP-chl_Col" || mthd == "PP-chl-linear_Col") && this->IF_hppf()) { // TODO(mingsuntse): improve prune method name
        this->ProbPruneCol_chl(APP<Dtype>::prune_interval);
      } else {
        //LOG(FATAL) << "Wrong: unknown prune_method";
        //exit(1);
      }
      this->UpdatePrunedRatio();
      this->IF_layer_prune_finished();
    }
#ifdef ShowTimingLog
    cout << "  after updating masks: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
#endif

    // Print weight magnitude
    const Dtype* weight = this->blobs_[0]->cpu_data();
    if (APP<Dtype>::num_log > 0) {
      if (APP<Dtype>::prune_unit == "Col") {
        cout << "ave-magnitude_col " << APP<Dtype>::step_ << " " << layer_name << ":";
        for (int j = 0; j < num_col; ++j) {
          Dtype sum = 0;
          for (int i = 0; i < num_row; ++i) {
            sum += fabs(weight[i*num_col + j]);
          }
          cout << " " << sum;
        }
        cout << endl;
      } else if (APP<Dtype>::prune_unit == "Row") {
        cout << "ave-magnitude_row " << APP<Dtype>::step_ << " " << layer_name << ":";
        for (int i = 0; i < num_row; ++i) {
          Dtype sum = 0;
          for (int j = 0; j < num_col; ++j) {
            sum += fabs(weight[i*num_col + j]);
          }
          cout << " " << sum;
        }
        cout << endl;
      }
    }
    // Summary print
    if (mthd != "None" && L < APP<Dtype>::show_num_layer) {
      cout << layer_name << "  IF_prune: " << IF_prune
           << "  pruned_ratio: " << APP<Dtype>::pruned_ratio[L];
      cout << "  pruned_ratio_row: " << APP<Dtype>::num_pruned_row[L] * 1.0 / num_row << "(" << APP<Dtype>::num_pruned_row[L] << ")"
           << "  pruned_ratio_col: " << APP<Dtype>::num_pruned_col[L] * 1.0 / num_col << "(" << APP<Dtype>::num_pruned_col[L] << ")";
      cout << "  current prune_ratio: "  << APP<Dtype>::current_prune_ratio[L];
      cout << "  prune_ratio: "  << APP<Dtype>::prune_ratio[L] << endl;
      cout << "  iter_prune_finished: " << APP<Dtype>::iter_prune_finished[L];
      cout << "  IF_acc_recovered: " << APP<Dtype>::IF_acc_recovered;
      cout << "  lr: " << APP<Dtype>::learning_rate;
      cout << "  iter_size: " << APP<Dtype>::iter_size;
      cout << endl;
    }

    // Apply masks
    if (1) {
      caffe_gpu_mul(this->blobs_[0]->count(),
                    this->blobs_[0]->gpu_data(),
                    this->masks_[0]->gpu_data(),
                    this->blobs_[0]->mutable_gpu_data());
      this->IF_masks_updated = false;
    }
#ifdef ShowTimingLog
    cout << "  after updating masks: " << (double)(clock() - t1) / CLOCKS_PER_SEC << endl;
#endif

  } else if (this->phase_ == TEST
             && IF_prune
             && APP<Dtype>::iter_prune_finished[L] == INT_MAX
             && APP<Dtype>::prune_coremthd.substr(0, 2) == "PP") {
    this->GenerateMasks();
  }
}

template <typename Dtype>
void Layer<Dtype>::PruneBackward(const vector<Blob<Dtype>*>& top) {
  const int L = APP<Dtype>::layer_index[this->layer_param_.name()];
  // TaylorPrune
  if (APP<Dtype>::prune_method == "TP_Row" && (APP<Dtype>::step_ - 1) % APP<Dtype>::prune_interval == 0 && APP<Dtype>::inner_iter == 0) {
    const bool IF_want_prune  = APP<Dtype>::prune_method != "None" && APP<Dtype>::prune_ratio[L] > 0;
    const bool IF_been_pruned = APP<Dtype>::pruned_ratio[L] > 0;
    const bool IF_enough_iter = APP<Dtype>::step_ >= APP<Dtype>::prune_begin_iter+1;
    const bool IF_prune = IF_want_prune && (IF_been_pruned || IF_enough_iter);
    if (IF_prune && APP<Dtype>::iter_prune_finished[L] == INT_MAX) {
      this->TaylorPrune(top);
    }
  }

  // Print and check
  if (APP<Dtype>::show_layer.size() >= L+1 && APP<Dtype>::show_layer[L] == '1'
      && APP<Dtype>::step_ % APP<Dtype>::show_interval == 0 && APP<Dtype>::inner_iter == 0) {
    this->Print('b');
  }

  if (APP<Dtype>::pruned_ratio[L] > 0) {
    caffe_gpu_mul(this->blobs_[0]->count(),
                  this->blobs_[0]->gpu_diff(),
                  this->masks_[0]->gpu_data(),
                  this->blobs_[0]->mutable_gpu_diff());
  }
}


INSTANTIATE_CLASS(Layer);

}  // namespace caffe
