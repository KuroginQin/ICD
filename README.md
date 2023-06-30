# Towards a Better Trade-Off between Quality and Efficiency of Community Detection: An Inductive Embedding Method across Graphs

This repository provides a reference implementation of the *ICD* method introduced in the paper "[**Towards a Better Trade-Off between Quality and Efficiency of Community Detection: An Inductive Embedding Method across Graphs**](https://dl.acm.org/doi/abs/10.1145/3596605)", which has been accepcted by **ACM Transactions on Knowledge Discovery from Data** (**TKDD**).

### Abstract
Many network applications can be formulated as NP-hard combinatorial optimization problems of community detection (CD) that partitions nodes of a graph into several groups with dense linkage. Most existing CD methods are *transductive*, which independently optimized their models for each single graph, and can only ensure either high quality or efficiency of CD by respectively using advanced machine learning techniques or fast heuristic approximation. In this study, we consider the CD task and aim to alleviate its NP-hard challenge. Motivated by the efficient *inductive* inference of graph neural networks (GNNs), we explore the possibility to achieve a better trade-off between the quality and efficiency of CD via an *inductive* embedding scheme across multiple graphs of a system and propose a novel *inductive* community detection (ICD) method. Concretely, ICD first conducts the *offline* training of an adversarial dual GNN structure on historical graphs to capture key properties of a system. The trained model is then directly generalized to new graphs of the same system for *online* CD without additional optimization, where a better trade-off between quality and efficiency can be achieved. Compared with existing *inductive* approaches, we develop a novel feature extraction module based on graph coarsening, which can efficiently extract informative feature inputs for GNNs. Moreover, our original designs of adversarial dual GNN and clustering regularization loss further enable ICD to capture permutation-invariant community labels in the *offline* training and help derive community-preserved embedding to support the high-quality *online* CD. Experiments on a set of benchmarks demonstrate that ICD can achieve a significant trade-off between quality and efficiency over various baselines.

### Citing
If you find this project useful for your research, please cite the following paper.

```
@article{qin2023towards,
  title={Towards a better trade-off between quality and efficiency of community detection: An inductive embedding method across graphs},
  author={Qin, Meng and Zhang, Chaorui and Bai, Bo and Zhang, Gong and Yeung, Dit-Yan},
  journal={ACM Transactions on Knowledge Discovery from Data (TKDD)},
  year={2023},
  publisher={ACM New York, NY}
}
```

If you have questions, you can contact the author via [mengqin_az@foxmail.com].

### Requirements
pytorch

scipy

munkres

sklearn

### Usage

Please download the example [datasets](https://hkustconnect-my.sharepoint.com/:u:/g/personal/mqinae_connect_ust_hk/ETxx52cWPVtGqokbKMd97RcB6fii0IxdD554U0liAaqYTg?e=XFR42L) (~1.41GB), including the edge list and 'ground-truth' for each graph. Unzip it and put all the data files under the directory ./data.

One can also generate syntetic graphs (e.g., *GN-Net* and *LFR-Net*) from scratch using the script under the directory ./data. Before generating graphs of *LFR-Net*, cd to ./data/LFR and use 'make' to build the corresponidng C++ project.

For the demo of ICD-M and ICD-C on different datasets, please run the script *ICD_[M,C]_[dataset name]_demo.py*. The evaluation result (on validation set) w.r.t. each epoch will be saved under the directory ./res. If the flag variable *save_flag* (in *ICD_[M,C]_[dataset name]_demo.py*) is set to **True**, the checkpoint w.r.t. each epoch will be saved under the directory ./chpt. If the flag variable *test_eva_flag* (in *ICD_[M,C]_[dataset name]_demo.py*) is set to **True**, the script will also conduct the evaluation on test set and save correponding results w.r.t. each epoch under the directory ./res.

To evaluate the quality and efficiency of a saved checkppoint (on test set), set *epoch_idx* to the epoch to be checked in *ICD_[M,C]_[dataset name]_chpt.py* and run it.

### Notes
When testing the runtime, please make sure there are no other processes with heavy resource reqirements (e.g., GPUs and memory) running on the same server. Otherwise, the evaluated runtime may not be stable.

For some large snapshots (e.g., with several thousands nodes), the compution of the modularity and NCut metrics may be time-consuming.

