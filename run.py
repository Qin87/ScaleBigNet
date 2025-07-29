import gc
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # supress: oneDNN custom operations are on
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 supress warning:Unable to register cuFFT factory...
import numpy as np
import uuid
import socket
import sys
import sys, os
print('CWD:', os.getcwd())
print('sys.path:', sys.path)

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary, ModelCheckpoint

from utils.utils import use_best_hyperparams, get_available_accelerator, log_file
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

    start_time = time.time()
    with open(log_directory + log_file_name_with_timestamp, 'w') as log_file:
        print(args, file=log_file)
        print(f"Machine ID: {socket.gethostname()}-{':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) for elements in range(0, 8 * 6, 8)][::-1])}", file=log_file)
        sys.stdout = log_file

        val_accs, test_accs = [], []
        for num_run in range(args.num_runs):
            print("\nstart run: ", num_run)
            train_mask, val_mask, test_mask = get_dataset_split(args.dataset, data, args.dataset_directory, num_run)

            # Get model
            args.num_features, args.num_classes = data.num_features, dataset.num_classes
            model = get_model(args)

            lit_model = LightingFullBatchModelWrapper(
                model=model,
                lr=args.lr,
                weight_decay=args.weight_decay,
                evaluator=evaluator,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask,
            )

            # Setup Pytorch Lighting Callbacks
            early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=args.patience)
            model_summary_callback = ModelSummary(max_depth=-1)
            if not os.path.exists(f"{args.checkpoint_directory}/"):
                os.mkdir(f"{args.checkpoint_directory}/")
            model_checkpoint_callback = ModelCheckpoint(
                monitor="val_acc",
                mode="max",
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

            # Fit the model
            trainer.fit(model=lit_model, train_dataloaders=data_loader)

            # Compute validation and test accuracy
            val_acc = model_checkpoint_callback.best_model_score.item()
            test_acc = trainer.test(ckpt_path="best", dataloaders=data_loader)[0]["test_acc"]
            test_accs.append(test_acc)
            val_accs.append(val_acc)

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


if __name__ == "__main__":
    args = use_best_hyperparams(args, args.dataset) if args.use_best_hyperparams else args
    log_directory, log_file_name_with_timestamp = log_file(args.model, args.dataset, args)
    print(args)
    run(args)
