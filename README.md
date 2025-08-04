# ScaleBigNet: Multi-scale GNN for large graphs

## Overview

ScaleNet leverage multi-scale features to learn from graph to get better performance than single-scale models. 
ScaleBigNet avoids multiplication between adjacency matrices (i.e., avoids A^2). 
Instead of computing (AA)X as in ScaleNet,  it uses A(AX) to obtain aggregated features.


## Getting Started

To get up and running with the project, you need to first set up your environment and install the necessary dependencies. This guide will walk you through the process step by step.

### Setting Up the Environment

Tested Python version: 3.9-3.12. We recommend using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to set up the environment as follows:

```bash
conda create -n scalebignet python=3.10
conda activate scalebignet
```

### Installing Dependencies

Once the environment is activated, install the required packages:


1. Follow the official instructions to install [PyTorch](https://pytorch.org/get-started/previous-versions/).
2. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
3. For `pytorch-scatter` and `pytorch-sparse`, download the packages from [this link](https://pytorch-geometric.com/whl/torch-2.3.0%2Bcu121.html) according to your PyTorch, Python, and OS version. Then, use `pip` to install them.
4. Other packages to install:
```bash
pip install ogb
pip install pytorch_lightning
pip install gdown
```

For M1/M2/M3 Mac users, `pyg` (PyTorch Geometric) needs to be installed from source. Detailed instructions for this process can be found [here](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#installation-from-source).

### Code Structure

* `run.py`: This script is used to run the models.

* `model.py`: Contains the definition of the directed convolutional network called HoloNet in the code and FaberNet in the Paper.


## Running Experiments

This section provides instructions on how to reproduce the experiments outlined in the paper. Note that some of the results may not be reproduced *exactly*, given that some of the operations used are intrinsically non-deterministic on the GPU, as explained [here](https://github.com/pyg-team/pytorch_geometric/issues/92). However, you should obtain results very close to those in the paper.

### FaberNet-Experiments

To reproduce the results of Table 1 in the paper, use the following command:

```bash
python run.py --dataset arxiv-year --use_best_hyperparams=1 --num_runs 10
```

The `--dataset` parameter specifies the dataset to be used. Replace `arxiv-year` with the name of the dataset you want to use. 

## Command Line Arguments

The following command line arguments can be used with the code:

### Dataset Arguments

| Argument               | Type | Default Value | Description                   |
| ---------------------- | ---- | ------------- | ----------------------------- |
| --dataset              | str  | "arxiv-year"   | Name of the dataset           |
| --dataset_directory    | str  | "dataset"     | Directory to save datasets    |
| --checkpoint_directory | str  | "checkpoint"  | Directory to save checkpoints |

### Preprocessing Arguments

| Argument     | Action     | Description                     |
| ------------ | ---------- | ------------------------------- |
| --undirected | store_true | Use undirected version of graph |
| --self_loops | store_true | Add self-loops to the graph     |
| --transpose  | store_true | Use transpose of the graph      |

### Model Arguments

| Argument         | Type   | Default Value | Description                                             |
|------------------| ------ |--------------|---------------------------------------------------------|
| --model          | str    | "gnn"        | Model type                                              |
| --hidden_dim     | int    | 64           | Hidden dimension of model                               |
| --num_layers     | int    | 3            | Number of GNN layers                                    |
| --dropout        | float  | 0.0          | Feature dropout                                         |
| --alpha          | float  | 0.5          | Direction convex combination params between A and At    |
| --beta           | float  | 0.5          | Direction convex combination params between AAt and AtA |
| --gamma          | float  | 0.5          | Direction convex combination params between AA and AtAt |
| --learn_alpha    | action | -            | If set, learn alpha                                     |
| --conv_type      | str    | "scalenet"   | Model                                                   |
| --normalize      | action | -            | If set, normalize                                       |
| --jk             | str    | "max"        | Either "max", "cat" or None                             |
| --weight_penalty | str    | "exp"        | Either "exp", "line" or None                            |
| --k_plus         | int    | 3            | Polynomial Order                                        |
| --exponent       | float  | -0.25        | normalization in adj-matrix                             |
| --lrelu_slope    | float  | -1           | LeakyRelu slope                                         |
| --zero_order     | bool   | False        | Whether to use zero-order term                          |


### Training Args

| Argument            | Type  | Default Value | Description                                        |
| ------------------- | ----- | ------------- | -------------------------------------------------- |
| --lr                | float | 0.001         | Learning Rate                                      |
| --weight_decay      | float | 0.0           | Weight decay (if only real weights are used)       |
| --num_epochs        | int   | 10000         | Max number of epochs                               |
| --patience          | int   | 10            | Patience for early stopping                        |
| --num_runs          | int   | 1             | Max number of runs                                 |



## Credit
This repository builds on top of Emanuele Rossi's [dir-gnn repository](https://github.com/emalgorithm/directed-graph-neural-network) and Christian Koke's [HoloNet](https://github.com/ChristianKoke/HoloNets/tree/6bcd8b92177f0b075ae0664a1288efb3b589ee3b). 


