import gc
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # supress: oneDNN custom operations are on
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 supress warning:Unable to register cuFFT factory...
import numpy as np
import uuid
import socket
import sys, os
print('CWD:', os.getcwd())
print('sys.path:', sys.path)

import torch
# torch.cuda.set_device(0)
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint

from utils.utils import use_best_hyperparams, get_available_accelerator, log_file, rename_log, free_space
from datasets.data_loading import get_dataset, get_dataset_split
from datasets.dataset import FullBatchGraphDataset
from model import get_model, LightingFullBatchModelWrapper
from utils.arguments import args
import warnings   # ScaleNet2
warnings.filterwarnings("ignore")
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)   #
import time

original_load = torch.load

import torch

def print_memory(tag=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"[{tag}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB")
    else:
        print(f"[{tag}] CUDA not available, skipping memory check")

def custom_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return original_load(*args, **kwargs)

torch.load = custom_load

def run(args):
    torch.manual_seed(args.seed)

    # Get dataset and dataloader
    dataset, evaluator = get_dataset(
        name=args.dataset,
        root_dir=args.dataset_directory,
        undirected=args.undirected,
        self_loops=args.self_loops,
        transpose=args.transpose,
    )
    data = dataset._data
    data_loader = DataLoader(FullBatchGraphDataset(data), batch_size=1, collate_fn=lambda batch: batch[0])
    num_node = data.x.shape[0]
    print(data)
    print("nodes:", data.x.shape, "edges:", data.edge_index.shape, )
    if hasattr(data, "train_mask"):
        print("train:", data.train_mask.sum().item(),
              "val:", data.val_mask.sum().item() if hasattr(data, "val_mask") else "N/A",
              "test:", data.test_mask.sum().item() if hasattr(data, "test_mask") else "N/A")
    else:
        print("No split masks available.")


    start_time = time.time()
    with open(log_directory + log_file_name_with_timestamp, 'w') as log_file:
        print(args, file=log_file)
        print(f"Machine ID: {socket.gethostname()}-{':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8 * 6, 8)][::-1])}", file=log_file)
        # sys.stdout = log_file

        print_memory("Start")
        val_accs, test_accs = [], []
        for num_run in range(args.num_runs):
            print("\nstart run: ", num_run)
            train_mask, val_mask, test_mask = get_dataset_split(args.dataset, data, args.dataset_directory, num_run)
            if num_run == 0:
                print("train:", train_mask.sum().item(),
                      "val:", val_mask.sum().item(),
                      "test:", test_mask.sum().item(),
                      file=sys.__stdout__)

            print_memory("Before model load")
            # Get model
            args.num_features, args.num_classes, args.edge_index, args.num_nodes = data.num_features, dataset.num_classes, data.edge_index, num_node
            model = get_model(args)
            print(model)
            print_memory("After model load")


            lit_model = LightingFullBatchModelWrapper(
                model=model,
                args=args,
                evaluator=evaluator,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
            )

            # Setup Pytorch Lighting Callbacks
            monitor_metric = args.monitor  # "val_loss"   "val_acc"   "train_loss"
            if "loss" in monitor_metric:
                mode = "min"
            else:
                mode = "max"

            early_stopping_callback = EarlyStopping(monitor=monitor_metric,mode=mode,patience=args.patience)

            model_summary_callback = ModelSummary(max_depth=-1)
            if not os.path.exists(f"{args.checkpoint_directory}/"):
                os.mkdir(f"{args.checkpoint_directory}/")
            model_checkpoint_callback = ModelCheckpoint(
                monitor=monitor_metric,
                mode=mode,
                dirpath=f"{args.checkpoint_directory}/{str(uuid.uuid4())}/",
            )

            # Setup Pytorch Lighting Trainer
            trainer = pl.Trainer(
                log_every_n_steps=1,
                enable_progress_bar = False,
                enable_model_summary=False,  # suppresses the model table  # ScaleNet2
                max_epochs=args.num_epochs,
                callbacks=[
                    early_stopping_callback,  # comment out will be much slower!
                    # model_summary_callback,
                    model_checkpoint_callback,
                ],
                profiler="simple" if args.profiler else None,
                accelerator=get_available_accelerator(),
                devices=[args.gpu_idx],
            )
            print(lit_model)

            print_memory("Before training")
            trainer.fit(model=lit_model, train_dataloaders=data_loader)


            # Compute validation and test accuracy
            val_acc = model_checkpoint_callback.best_model_score.item()
            test_acc = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]["test_acc"]
            test_accs.append(test_acc)
            val_accs.append(val_acc)
            print(f"Test Acc: {test_acc* 100:.2f}", file=sys.__stdout__)

            del model
            del lit_model
            del trainer
            del early_stopping_callback
            del model_summary_callback
            del model_checkpoint_callback
            torch.cuda.empty_cache()
            gc.collect()

            print('Used time: ', time.time() - start_time)

        print(f"Test Acc: {np.mean(test_accs) *100:.2f}±{np.std(test_accs) * 100:.2f}")
        print(f"Test Acc: {np.mean(test_accs) *100:.2f}±{np.std(test_accs) * 100:.2f}", file=sys.__stdout__)
        result_str = f"{np.mean(test_accs) * 100:.2f}±{np.std(test_accs) * 100:.2f}"

    rename_log(log_directory, log_file_name_with_timestamp, result_str)
    free_space()


if __name__ == "__main__":
    args = use_best_hyperparams(args, args.dataset) if args.use_best_hyperparams else args
    log_directory, log_file_name_with_timestamp = log_file(args.model, args.dataset, args)
    print(args)
    run(args)
