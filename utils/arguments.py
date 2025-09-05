import argparse
import logging

parser = argparse.ArgumentParser("Directed Graph Neural Network")
parser.add_argument("--seed", type=int, help="manual seed", default=0)

### Dataset Args
parser.add_argument("--dataset", type=str, default="chameleon",
                    help="Options: genius, 'pokec', 'penn94', 'chameleon'." )
parser.add_argument("--dataset_directory", type=str, help="Directory to save datasets", default="dataset")
parser.add_argument("--checkpoint_directory", type=str, help="Directory to save checkpoints", default="checkpoint")

### Preprocessing Args
parser.add_argument("--undirected", action="store_true", help="Whether to use undirected version of graph")
parser.add_argument("--self_loops", action="store_true", help="Whether to add self-loops to the graph")
parser.add_argument("--transpose", action="store_true", help="Whether to use transpose of the graph")

### Model Args
parser.add_argument("--model", type=str, help="gnn for scale, faber, linkx, link", default="scale")
parser.add_argument("--monitor", type=str, help="train_loss, train_acc, val_loss, val_acc, ", default="val_acc")
parser.add_argument("--hid_dim", type=int, help="Hidden dimension of model", default=32)
parser.add_argument("--num_layers", type=int, help="Number of GNN layers", default=2)
parser.add_argument("--dropout", type=float, help="Feature dropout", default=0.5)
parser.add_argument("--alpha", type=float, help="Direction convex combination params: A, At", default=1)
parser.add_argument("--beta", type=float, help="Direction convex combination params: AAt, AtA", default=-1)
parser.add_argument("--gamma", type=float, help="Direction convex combination params: AA, AtAt", default=-1)
parser.add_argument("--learn_alpha", action="store_true")
parser.add_argument("--conv_type", type=str, help="Selects Convolutional Layer", default="scale")
parser.add_argument("--normalize", type=int, default=0)
parser.add_argument("--jk", type=str, choices=["max", "cat", "None"], default="max")
parser.add_argument("--weight_penalty", type=str, choices=["exp", "lin", "None"], default="None")
parser.add_argument("--k_plus", type=int, help="Polynomial order", default=2)
parser.add_argument("--exponent", type=float, help="exponent in norm", default= -0.25)
parser.add_argument("--lrelu_slope", type=float, help="negative slope of Leaky Relu", default= -1.0)

parser.add_argument("--zero_order", type=int, help="If include zero order", default=0)
parser.add_argument("--cat_A_X", type=int, help="If include concatenate A and X", default=0)
parser.add_argument("--structure", type=float, default=0, help="1 pure structure, 0 pure feature, 0.5 structure is feature too")


### Training Args
parser.add_argument("--lr", type=float, help="Learning Rate", default=0.01)
parser.add_argument("--weight_decay", type=float, help="Weight decay", default=1e-3)
parser.add_argument("--num_epochs", type=int, help="Max number of epochs", default=10000)
parser.add_argument("--patience", type=int, help="Patience for early stopping", default=500)
parser.add_argument("--num_runs", type=int, help="Max number of runs", default=1)

### System Args
parser.add_argument("--use_best_hyperparams", type=int, default=1, help="whether use parameters in best_hyperparameters.yml")
parser.add_argument("--gpu_idx", type=int, help="Indexes of gpu to run program on", default=0)
parser.add_argument("--num_workers", type=int, help="Num of workers for the dataloader", default=0)
parser.add_argument("--profiler", action="store_true")

### for linkx
parser.add_argument('--link_init_layers_A', type=int, default=1)
parser.add_argument('--link_init_layers_X', type=int, default=1)
parser.add_argument('--inner_activation', action='store_true', help='Whether linkV3 uses inner activation')
parser.add_argument('--inner_dropout', action='store_true', help='Whether linkV3 uses inner dropout')

args = parser.parse_args()
if args.weight_penalty == "None":
    args.weight_penalty = None
if args.jk == "None":
    args.jk = None