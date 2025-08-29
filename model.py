import torch
from torch import nn, optim
import pytorch_lightning as pl
from torch_sparse import SparseTensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear
from torch_geometric.nn import (GCNConv,
    JumpingKnowledge)

from datasets.data_utils import get_norm_adj
from link_model import LINKX, LINK_Concat, H2GCN, LINK


def get_conv(input_dim, output_dim, args):
    if args.conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=args.self_loops)
    elif args.conv_type == "fabernet":
        return FaberConv(input_dim, output_dim, args)
    elif args.conv_type == "scale":
        return ScaleConv(input_dim, output_dim, args)
    else:
        raise ValueError(f"Convolution type {args.conv_type} not supported")


class ScaleConv(torch.nn.Module):
    '''
    multi-scale
    '''
    def __init__(self, input_dim, output_dim, args):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_plus = args.k_plus
        self.exponent = args.exponent
        self.weight_penalty = args.weight_penalty
        self.zero_order = args.zero_order
        self.structure = args.structure
        self.cat_A_X = args.cat_A_X

        if self.structure != 0:
            self.mlp_struct = Linear(args.num_nodes, output_dim)

        if self.cat_A_X  !=0:
            self.mlp_cat = Linear(2*output_dim, output_dim)

        if self.zero_order:
            self.lin_zero = Linear(input_dim, output_dim)
            # self.lin_dst_to_src_zero = Linear(input_dim, output_dim)

        # Lins for positive powers:
        self.lins_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(2* args.k_plus)
        ])

        self.lins_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(2*args.k_plus)
        ])

        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir", exponent=self.exponent)

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir", exponent=self.exponent)

        if self.structure != 1 or self.cat_A_X:
            y = self.adj_norm @ x
            y_t = self.adj_t_norm @ x
            sum_src_to_dst = self.lins_src_to_dst[0](y)
            sum_dst_to_src = self.lins_dst_to_src[0](y_t)

            totalB = 0
            totalC = 0
            if self.k_plus > 1:
                def get_weight(i):
                    if self.weight_penalty == 'exp':
                        return 1 / (2 ** i)
                    elif self.weight_penalty == 'lin':
                        return 1 / i
                    elif self.weight_penalty == 'None' or self.weight_penalty is None:
                        return 1
                    else:
                        raise ValueError(f"Weight penalty type {self.weight_penalty} not supported")
                yy = y
                ytyt = y_t
                yty = y
                yyt = y_t
                for i in range(1, self.k_plus):
                    yy = self.adj_norm @ yy
                    yty = self.adj_t_norm @ yty

                    yyt = self.adj_norm @ yyt
                    ytyt = self.adj_t_norm @ ytyt

                    w = get_weight(i)

                    if self.beta != -1:
                        b_term = (
                                self.beta * self.lins_src_to_dst[2 * i - 1](yyt)
                                + (1 - self.beta) * self.lins_src_to_dst[2 * i - 1](yty)
                        )
                        totalB += b_term * w

                    if self.gamma != -1:
                        c_term = (
                                self.gamma * self.lins_src_to_dst[2 * i](yy)
                                + (1 - self.gamma) * self.lins_dst_to_src[2 * i](ytyt)
                        )
                        totalC += c_term * w
            if self.alpha == -1:
                totalA = 0
            else:
                totalA = self.alpha * sum_src_to_dst + (1 - self.alpha) * sum_dst_to_src

            gnn_total = totalA + totalB + totalC
            if self.structure != 0:
                struct_value = self.adj_norm @ self.mlp_struct.weight.T
                total = self.structure* struct_value  + (1- self.structure)* gnn_total
            else:
                total = gnn_total

            if self.cat_A_X:
                struct_value = self.adj_norm @ self.mlp_struct.weight.T
                concat_feat = torch.cat([struct_value, gnn_total], dim=1)
                concat_output = self.mlp_cat(concat_feat)
                total += concat_output
        else:  # structure=1 and cat!=1
            total = self.adj_norm @ self.mlp_struct.weight.T

        if self.zero_order:
            total = total + self.lin_zero(x)

        return total


class FaberConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, args):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.k_plus = args.k_plus
        self.exponent = args.exponent
        self.weight_penalty = args.weight_penalty
        self.zero_order = args.zero_order

        if self.zero_order:
            self.lin_src_to_dst_zero = Linear(input_dim, output_dim)
            self.lin_dst_to_src_zero = Linear(input_dim, output_dim)

        self.lins_src_to_dst = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(args.k_plus)
        ])

        self.lins_dst_to_src = torch.nn.ModuleList([
            Linear(input_dim, output_dim) for _ in range(args.k_plus)
        ])

        self.alpha = args.alpha
        self.adj_norm, self.adj_t_norm = None, None

    def forward(self, x, edge_index):
        if self.adj_norm is None:
            row, col = edge_index
            num_nodes = x.shape[0]

            adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
            self.adj_norm = get_norm_adj(adj, norm="dir", exponent = self.exponent)

            adj_t = SparseTensor(row=col, col=row, sparse_sizes=(num_nodes, num_nodes))
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir", exponent = self.exponent)

        y   = self.adj_norm   @ x
        y_t = self.adj_t_norm @ x
        sum_src_to_dst = self.lins_src_to_dst[0](y) 
        sum_dst_to_src = self.lins_dst_to_src[0](y_t) 
        if self.zero_order:
            sum_src_to_dst =  sum_src_to_dst + self.lin_src_to_dst_zero(x)
            sum_dst_to_src =  sum_dst_to_src + self.lin_dst_to_src_zero(x)

        if self.k_plus > 1:
            if self.weight_penalty == 'exp':
                for i in range(1,self.k_plus):
                    y   = self.adj_norm   @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y)/(2**i)
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t)/(2**i)

            elif self.weight_penalty == 'lin':
                for i in range(1,self.k_plus):
                    y   = self.adj_norm   @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y)/i
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t)/i
            elif self.weight_penalty == None:
                for i in range(1,self.k_plus):
                    y   = self.adj_norm   @ y
                    y_t = self.adj_t_norm @ y

                    sum_src_to_dst = sum_src_to_dst + self.lins_src_to_dst[i](y)
                    sum_dst_to_src = sum_dst_to_src + self.lins_dst_to_src[i](y_t)
            else:
                raise ValueError(f"Weight penalty type {self.weight_penalty} not supported")
       
        total = self.alpha * sum_src_to_dst + (1 - self.alpha) * sum_dst_to_src

        return total


class GNN(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.conv_type = args.conv_type
        self.alpha = nn.Parameter(torch.ones(1) * args.alpha, requires_grad=args.learn_alpha)
        self.lrelu_slope = args.lrelu_slope

        output_dim = args.hid_dim if args.jk else args.num_classes
        if args.num_layers == 1:
            self.convs = ModuleList([get_conv(args.num_features, output_dim, args)])
        else:
            self.convs = ModuleList([get_conv(args.num_features, args.hid_dim, args)])
            for _ in range(args.num_layers - 2):
                self.convs.append(get_conv(args.hid_dim, args.hid_dim, args))
            self.convs.append(get_conv(args.hid_dim, output_dim, args))

        if args.jk is not None:
            input_dim = args.hid_dim * args.num_layers if args.jk == "cat" else args.hid_dim
            self.lin = Linear(input_dim, args.num_classes)
            self.jump = JumpingKnowledge(mode=args.jk, channels=args.hid_dim, num_layers=args.num_layers)

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.jk = args.jk
        self.normalize = args.normalize
    def forward(self, x, edge_index):
        xs = []
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1 or self.jk:
                x = F.leaky_relu(x,negative_slope= self.lrelu_slope)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize:
                    x = F.normalize(x, p=2, dim=1)
            xs += [x]

        if self.jk is not None:
            x = self.jump(xs)
            x = self.lin(x)

        return torch.nn.functional.log_softmax(x, dim=1)


class LightingFullBatchModelWrapper(pl.LightningModule):
    def __init__(self, model, lr, weight_decay, train_mask, val_mask, test_mask, evaluator=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.evaluator = evaluator
        self.train_mask, self.val_mask, self.test_mask = train_mask, val_mask, test_mask

    # def training_step(self, batch, batch_idx):
    #     x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
    #     out = self.model(x, edge_index)
    #
    #     # Training loss and accuracy
    #     train_loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
    #     y_pred = out.max(1)[1]
    #     train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
    #
    #     self.log("train_loss", train_loss, prog_bar=True)
    #     self.log("train_acc", train_acc, prog_bar=True)
    #
    #     return train_loss
    #
    # def validation_step(self, batch, batch_idx):
    #     x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
    #     out = self.model(x, edge_index)
    #
    #     # Validation loss and accuracy
    #     val_loss = nn.functional.nll_loss(out[self.val_mask], y[self.val_mask].squeeze())
    #     y_pred = out.max(1)[1]
    #     val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
    #
    #     self.log("val_loss", val_loss, prog_bar=True)
    #     self.log("val_acc", val_acc, prog_bar=True)
    #
    #     return val_loss
    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        train_loss = nn.functional.nll_loss(out[self.train_mask], y[self.train_mask].squeeze())
        self.log("train_loss", train_loss)

        val_loss = nn.functional.nll_loss(out[self.val_mask], y[self.val_mask].squeeze())
        self.log("val_loss", val_loss)

        y_pred = out.max(1)[1]
        train_acc = self.evaluate(y_pred=y_pred[self.train_mask], y_true=y[self.train_mask])
        self.log("train_acc", train_acc)
        val_acc = self.evaluate(y_pred=y_pred[self.val_mask], y_true=y[self.val_mask])
        self.log("val_acc", val_acc)

        return train_loss

    def evaluate(self, y_pred, y_true):
        if self.evaluator:
            acc = self.evaluator.eval({"y_true": y_true, "y_pred": y_pred.unsqueeze(1)})["acc"]
        else:
            acc = y_pred.eq(y_true.squeeze()).sum().item() / y_pred.shape[0]
        return acc

    def test_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y.long(), batch.edge_index
        out = self.model(x, edge_index)

        y_pred = out.max(1)[1]
        test_acc = self.evaluate(y_pred=y_pred[self.test_mask], y_true=y[self.test_mask])
        self.log("test_acc", test_acc, prog_bar=True)

    def configure_optimizers(self):
        other_params, imag_weights, real_weights = [], [], []

        for name, param in self.model.named_parameters():
            if "imag" in name:
                imag_weights.append(param)
  
            elif "real" in name:
                real_weights.append(param)

            else:
                other_params.append(param)

        optimizer = optim.AdamW([{'params': other_params, 'weight_decay': self.weight_decay}], lr = self.lr)
        print(optimizer)
        return optimizer


def get_model(args):
    if args.model == 'h2gcn':
        return H2GCN(args)
    elif args.model == 'link_concat':
        return LINK_Concat(args)
    elif args.model == 'linkx':
        return LINKX(args)
    elif args.model == 'link':
        return LINK(args)
    else:
        return GNN(args=args)
