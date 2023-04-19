import json

import torch.optim as optim

from src.arguments import get_arguments
from src.sampling.betas import get_betas
from src.sampling.deltas import get_deltas
from src.toy.distributions import GaussianMixtureToyJoint, SimpleGaussianToyVariational
from src.toy.static_target import StaticTarget
from src.trainer.static_target_trainer import StaticTargetTrainer
from src.utils.experiment import get_device, init_experiment, watch_model


def main():
    # init
    args = get_arguments()
    device = get_device(args)
    init_experiment(args, "static_target")
    print(("ARGUMENTS-" * 20)[:80])
    print(json.dumps(vars(args), indent=4))
    print(("DEVICE-" * 20)[:80])
    print(device.type)
    # model
    betas = get_betas(args).to(device)
    deltas = get_deltas(args, device).to(device)
    joint = GaussianMixtureToyJoint(
        args, mean=3.0, diagonal_covariance=1.0, num_components=8
    ).to(device)
    variational = SimpleGaussianToyVariational(args, diagonal_covariance=3.0**2).to(
        device
    )
    model = StaticTarget(args, joint, variational, betas, deltas).to(device)
    watch_model(args, model)
    print(("MODEL-" * 20)[:80])
    print(model)
    # optimizer
    optimizer = optim.Adam(params=model.parameters(), lr=args.lrate)
    # scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lrate_scheduler_step_size,
        gamma=args.lrate_scheduler_gamma,
    )
    # trainer
    trainer = StaticTargetTrainer(
        args,
        model,
        optimizer,
        scheduler,
        {
            "train": [None] * args.max_iterations_per_epoch,
            "test": [None] * args.max_iterations_per_epoch,
        },
        device,
    )
    # train
    print(("TRAIN-" * 20)[:80])
    trainer.train()
    # test
    print(("TEST-" * 20)[:80])
    trainer.test()


if __name__ == "__main__":
    main()
