# ScaleBigNet: Multi-scale GNN for large graphs

 - [Overview](#overview)
  - [Getting Started](#getting-started)
    - [Setting Up the Environment](#setting-up-the-environment)
    - [Installing Dependencies](#installing-dependencies)
    - [Code Structure](#code-structure)
  - [Running Experiments](#running-experiments)
    - [Node Classification Experiments](#node-classification-experiments)
  - [Command Line Arguments](#command-line-arguments)
    - [Dataset Arguments](#dataset-arguments)
    - [Preprocessing Arguments](#preprocessing-arguments)
    - [Model Arguments](#model-arguments)
    - [Training Args](#training-args)

## Overview

ScaleNet leverage multi-scale features to learn from graph to get better performance than single-scale models. 
However, higher-scaled feature learning in ScaleNet involves matrix multiplication of adjacency matrix, whose dimensions are node number of graph.
ScaleBigNet avoids AA, by A(AX) instead of (AA)X to get aggregated feature.

## Getting Started

To get up and running with the project, you need to first set up your environment and install the necessary dependencies. This guide will walk you through the process step by step.

### Setting Up the Environment

The project is designed to run on Python 3.10. We recommend using [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) to set up the environment as follows:

```bash
conda create -n holonet python=3.10
conda activate holonet
```

### Installing Dependencies

Once the environment is activated, install the required packages:

```bash
conda install pytorch==2.0.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg pytorch-sparse -c pyg
pip install ogb==1.3.6
pip install pytorch_lightning==2.0.2
pip install gdown==4.7.1
```

Please ensure that the version of `pytorch-cuda` matches your CUDA version. If your system does not have a GPU, use the following command to install PyTorch:

```bash
conda install pytorch==2.0.1 -c pytorch
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
python -m src.run --dataset chameleon --use_best_hyperparams --num_runs 10
```

The `--dataset` parameter specifies the dataset to be used. Replace `chameleon` with the name of the dataset you want to use. 

## Command Line Arguments

The following command line arguments can be used with the code:

### Dataset Arguments

| Argument               | Type | Default Value | Description                   |
| ---------------------- | ---- | ------------- | ----------------------------- |
| --dataset              | str  | "chameleon"   | Name of the dataset           |
| --dataset_directory    | str  | "dataset"     | Directory to save datasets    |
| --checkpoint_directory | str  | "checkpoint"  | Directory to save checkpoints |

### Preprocessing Arguments

| Argument     | Action     | Description                     |
| ------------ | ---------- | ------------------------------- |
| --undirected | store_true | Use undirected version of graph |
| --self_loops | store_true | Add self-loops to the graph     |
| --transpose  | store_true | Use transpose of the graph      |

### Model Arguments

| Argument         | Type   | Default Value | Description                         |
| ---------------- | ------ | ------------- | ----------------------------------- |
| --model          | str    | "gnn"         | Model type                          |
| --hidden_dim     | int    | 64            | Hidden dimension of model           |
| --num_layers     | int    | 3             | Number of GNN layers                |
| --dropout        | float  | 0.0           | Feature dropout                     |
| --alpha          | float  | 0.5           | Direction convex combination params |
| --learn_alpha    | action | -             | If set, learn alpha                 |
| --conv_type      | str    | "fabernet"    | Model                               |
| --normalize      | action | -             | If set, normalize                   |
| --jk             | str    | "max"         | Either "max", "cat" or None         |
| --weight_penalty | str    | "exp"         | Either "exp", "line" or None        |
| --k_plus         | int    | 3             | Polynomial Order                    |
| --exponent       | float  | -0.25         | normalization in adj-matrix         |
| --lrelu_slope    | float  | -1            | LeakyRelu slope                     |
| --zero_order     | bool   | False         | Whether to use zero-order term      |




### Training Args

| Argument            | Type  | Default Value | Description                                        |
| ------------------- | ----- | ------------- | -------------------------------------------------- |
| --lr                | float | 0.001         | Learning Rate                                      |
| --weight_decay      | float | 0.0           | Weight decay (if only real weights are used)       |
| --imag_weight_decay | float | 0.0           | Weight decay for imaginary part of weight matrices |
| --real_weight_decay | float | 0.0           | Weight decay for real      part of weight matrices |
| --num_epochs        | int   | 10000         | Max number of epochs                               |
| --patience          | int   | 10            | Patience for early stopping                        |
| --num_runs          | int   | 1             | Max number of runs                                 |





## Credit
This repository builds on top of Emanuele Rossi's [dir-gnn repository](https://github.com/emalgorithm/directed-graph-neural-network) and Christian Koke's [HoloNet](https://github.com/ChristianKoke/HoloNets/tree/6bcd8b92177f0b075ae0664a1288efb3b589ee3b). 


