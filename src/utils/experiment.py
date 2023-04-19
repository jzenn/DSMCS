import os
import time
from datetime import datetime

import numpy as np
import torch
import wandb

from src.utils.io import dump_json, load_json, mkdir


def get_device(args):
    try:
        mps = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except AttributeError:
        mps = False
    if args.no_cuda:
        return torch.device("cpu")
    elif mps:
        return torch.device("mps")
    elif args.cuda:
        return torch.device("cuda")
    else:
        # fallback to cuda or cpu, whichever is available
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


def save_model(args, model, additional_info=None):
    if additional_info is None:
        additional_info = {}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "additional_information": additional_info,
        },
        os.path.join(args.save_dir, "models", "model.pt"),
    )


def init_experiment(args, folder_name="default"):
    current_date_time = get_datetime_str()
    experiment_run_name = current_date_time
    if args.slurm_job_id is not None:
        experiment_run_name = f"{args.slurm_job_id}-" + experiment_run_name
    save_dir = os.path.join(
        args.path, "results", folder_name, args.experiment_name, experiment_run_name
    )
    args.save_dir = save_dir
    # set up directories
    if not os.path.isdir(save_dir):
        mkdir(save_dir)
    mkdir(os.path.join(save_dir, "logging"))
    mkdir(os.path.join(save_dir, "models"))
    # setup wandb
    if args.wandb:
        mkdir(os.path.join(save_dir, "wandb"))
        setup_wandb(args, run_name=experiment_run_name)
    # save arguments to path
    save_arguments_to_path(args, os.path.join(save_dir, "logging", "args.txt"))
    return save_dir


def get_datetime_str():
    current_date_time = datetime.today().strftime("%Y_%m_%d_%H-%M-%S-%s")
    return current_date_time


def save_arguments_to_path(args, path):
    # adapted from https://stackoverflow.com/a/55114771
    dump_json(args.__dict__, path)


def watch_model(args, model):
    # WandB
    if args.wandb:
        # watch gradients and parameters of the current model with WandB
        if not args.wandb_do_not_watch:
            wandb.watch(
                model,
                log="all",
                log_freq=500,  # every 500 iterations (doing ~900) approx. 2 x / epoch
                log_graph=False,
            )


def setup_wandb(args, run_name):
    if args.wandb_file is None:
        raise RuntimeError("Cannot use WandB without configuration file.")
    load_wandb_file_to_environment(args.wandb_file, args.wandb_offline)

    wandb_initialized = False
    while not wandb_initialized:
        try:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.experiment_name,
                dir=os.path.join(args.save_dir, "wandb"),
            )
            wandb_initialized = True
        except wandb.errors.UsageError:
            time.sleep(5)
        except wandb.errors.CommError:
            time.sleep(5)
        except wandb.errors.WaitTimeoutError:
            time.sleep(5)

    # save experiment config to WandB
    wandb.config.update(args)
    # set the run name of WandB
    wandb.run.name = run_name


def load_wandb_file_to_environment(wandb_file, wandb_offline=False):
    # fix ServiceStartTimeoutError
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    # load wandb properties
    wandb_props = load_json(wandb_file)
    for k, v in wandb_props.items():
        os.environ[k] = v
    if wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
