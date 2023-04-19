import argparse


def get_arguments():
    parser = argparse.ArgumentParser(description="VAE MNIST")
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--path", type=str, help="absolute path to experiment")
    parser.add_argument(
        "--experiment_name", type=str, default="local-mbp", help="name of experiment"
    )
    parser.add_argument("--slurm_job_id", type=str, help="job id on slurm cluster")
    # device
    parser.add_argument("--no_cuda", action="store_true", help="do not use cuda")
    parser.add_argument("--cuda", action="store_true", help="use cuda")
    # wandb
    parser.add_argument("--wandb", action="store_true", help="Use WandB for logging.")
    parser.add_argument(
        "--wandb_offline", action="store_true", help="Do not sync WandB."
    )
    parser.add_argument(
        "--wandb_do_not_watch",
        action="store_true",
        help="Turn off watching the model.",
    )
    parser.add_argument("--wandb_file", type=str, help="WandB file.")
    parser.add_argument(
        "--wandb_project", type=str, default="ldvi", help="WandB project."
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="jzenn", help="WandB entity."
    )
    # data
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="input batch size for training (default: 100)",
    )
    parser.add_argument(
        "--drop_last_batch",
        action="store_true",
        help="drop last batch in training and testing that is not of batch size",
    )
    # training
    parser.add_argument("--only_test", action="store_true", help="only test the model")
    parser.add_argument(
        "--lrate", type=float, default=1e-3, help="learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--lrate_scheduler_step_size",
        type=int,
        default=250,
        help="interval of epochs doing a step in the learning rate scheduler",
    )
    parser.add_argument(
        "--lrate_scheduler_gamma",
        type=float,
        default=0.75,
        help="factor of reducing the learning rate every step size epochs",
    )
    parser.add_argument(
        "--lrate_scheduler_up_to",
        type=int,
        default=2001,
        help="epoch up to which the leraning rate is scheduled",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="number of epochs to train (default: 1500)",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--hdim", type=int, default=200, help="number of hidden units (default: 200)"
    )
    parser.add_argument(
        "--zdim",
        type=int,
        default=50,
        help="dimension of latent variables (default: 20)",
    )
    parser.add_argument(
        "--max_iterations_per_epoch",
        type=int,
        default=-1,
        help="maximum number of iterations within one epoch",
    )
    # ais
    parser.add_argument(
        "--dsmcs", action="store_true", help="use the DSMCS bound instead of DAIS bound"
    )
    parser.add_argument(
        "--variational_augmentation",
        choices=["RW", "ULA", "Score-ULA", "UHA", "Score-UHA"],
    )
    parser.add_argument(
        "--n_particles",
        type=int,
        default=1,
        help="number of particles for iwae (default: 1)",
    )
    parser.add_argument(
        "--n_transitions", type=int, default=0, help="number of transitions"
    )
    parser.add_argument(
        "--deltas_nn_output_scale",
        type=float,
        default=1.0,
        help="use a NN to predict learning rates",
    )
    parser.add_argument(
        "--resampling",
        type=str,
        default=None,
        help="apply resampling to particles (bound changes: DAIS -> DSMCS)",
    )
    parser.add_argument(
        "--gumbel_tau",
        type=float,
        default=0.1,
        help="tau value for the Gapped Staight Through Estimator (Fan et al. 2022)",
    )
    args = parser.parse_args()
    return args
