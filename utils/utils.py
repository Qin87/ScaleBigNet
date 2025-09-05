import os
import yaml
import torch
from datetime import datetime
import shutil
import psutil
import sys

def use_best_hyperparams(args, dataset_name):
    best_params_file_path = "best_hyperparams.yml"
    with open(best_params_file_path, "r") as file:
        hyperparams = yaml.safe_load(file)

    for name, value in hyperparams[dataset_name].items():
        print(f' {name}: {value}')
        if hasattr(args, name):
            setattr(args, name, value)
        else:
            raise ValueError(f"Trying to set non existing parameter: {name}")

    return args


def get_available_accelerator():
    if torch.cuda.is_available():
        return "gpu"
    # Keep the following commented out as some of the operations
    # we use are currently not supported by mps
    # elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    #     return "mps"
    else:
        return "cpu"

def log_file(net_to_print, dataset_to_print, args):
    log_file_name = dataset_to_print+'_'+args.conv_type+'_'+net_to_print

    log_file_name += '_struct'+str(args.structure)
    log_file_name += '_zero'+str(args.zero_order)
    log_file_name += '_cat'+str(args.cat_A_X)

    if args.self_loops:
        log_file_name += '_Sloop'
    if args.undirected:
        log_file_name += '_Undirect'
    if args.transpose:
        log_file_name += '_Reverse'

    if args.conv_type == 'scale' and args.k_plus > 1:
        log_file_name += '_a'+str(args.alpha)+'_b'+str(args.beta)+'_c'+str(args.gamma)
    log_file_name += ('_k'+ str(args.k_plus)+'_lay'+str(args.num_layers)+'_lr'+str(
        args.lr)+'_split'+str(args.num_runs)+'_hid'+str(args.hid_dim)+'_'+str(args.weight_penalty)
                      +'_dp'+str(args.dropout)+'_n'+str(args.normalize)
                    +'_P'+str(args.patience)+'_jk'+args.jk+str(args.exponent)+'_s'+str(args.seed))


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_name_with_timestamp = f"{log_file_name}_{timestamp}.log"

    log_directory = "~/Documents/Benlogs/"  # Change this to your desired directory
    log_directory = os.path.expanduser(log_directory)

    return log_directory, log_file_name_with_timestamp


def clear_directory(path):
    if os.path.exists(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}", file=sys.__stdout__)
    else:
        print("Path:", path, "not existed!", file=sys.__stdout__)


def free_space():
    current_pid = os.getpid()
    python_procs = [
        p.info for p in psutil.process_iter(attrs=["pid", "name"])
        if "python" in p.info["name"].lower() and p.info["pid"] != current_pid
    ]

    if not python_procs:
        print("No other Python processes detected. Clearing logs...", file=sys.__stdout__)

        cwd = os.getcwd()
        print(f"Current working directory: {cwd}", file=sys.__stdout__)

        checkpoint_path = os.path.join(cwd, "checkpoint")
        logs_path = os.path.join(cwd, "lightning_logs")

        clear_directory(checkpoint_path)
        clear_directory(logs_path)
    else:
        print("Other Python processes are still running. Skipping cleanup.")


def rename_log(log_directory, log_file_name_with_timestamp, result_str):
    old_path = os.path.join(log_directory, log_file_name_with_timestamp)
    new_file_name = f"{result_str}_{log_file_name_with_timestamp}"
    new_path = os.path.join(log_directory, new_file_name)

    os.rename(old_path, new_path)
    print(f"Log file renamed to: {new_path}", file=sys.__stdout__)


