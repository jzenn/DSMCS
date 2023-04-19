import os

import wandb

from src.utils.experiment import save_model
from src.utils.io import dump_json


def list_mean(l):
    if len(l) == 0:
        return float("nan")
    return sum(l) / len(l)


class Trainer:
    def __init__(self, args, model, optimizer, scheduler, data_loaders, device):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_data_loader = data_loaders["train"]
        self.test_data_loader = data_loaders["test"] if "test" in data_loaders else None
        self.val_data_loader = data_loaders["val"] if "val" in data_loaders else None
        self.device = device
        #
        self.B = self.args.batch_size
        self.N = self.args.n_particles
        self.K = self.args.n_transitions
        self.D = self.args.zdim
        #
        self.total_iterations = {"train": 0, "test": 0, "val": 0}
        self.max_iterations_per_epoch = args.max_iterations_per_epoch
        #
        self.transition_log_dict = dict()

    def prepare_data(self, data_batch):
        raise NotImplementedError

    def forward_model(self, data, **kwargs):
        raise NotImplementedError

    def loss(self, return_dict, data):
        raise NotImplementedError

    def train(self):
        self.model.train()
        for epoch in range(self.args.epochs):
            # run epoch
            self.run_epoch(epoch, self.train_data_loader, mode="train")
            # schedule learning rate
            if self.scheduler is not None:
                # learning rate scheduling
                if (self.args.lrate_scheduler_up_to > -1) and (
                    epoch < self.args.lrate_scheduler_up_to
                ):
                    self.scheduler.step()
                elif self.args.lrate_scheduler_up_to == -1:
                    self.scheduler.step()

    def val(self):
        self.model.eval()
        if self.val_data_loader is not None:
            self.run_epoch(self.args.epochs, self.val_data_loader, mode="val")

    def test(self):
        self.model.eval()
        if self.test_data_loader is not None:
            self.run_epoch(self.args.epochs, self.test_data_loader, mode="test")

    def test_with_dataloader(self, data_loader, mode):
        if mode not in self.total_iterations:
            self.total_iterations.update({mode: 0})
        self.model.eval()
        self.run_epoch(self.args.epochs, data_loader, mode)

    def run_epoch(self, epoch, data_loader, mode="train"):
        epoch_loss_stats = dict()
        for iteration, data_batch in enumerate(data_loader):
            # increase total iteration counter
            self.total_iterations[mode] += 1
            # check whether to stop the current epoch
            if self.break_epoch(epoch, iteration, mode):
                print("breaking epoch early ...")
                break
            # get data
            data = self.prepare_data(data_batch)
            # run model
            return_dict = self.forward_model(
                data,
                # provide more information to the model
                # (accesible via kwargs in model)
                mode=mode,
                epoch=epoch,
                total_iteration=self.total_iterations[mode],
            )
            # compute loss
            loss = self.loss(return_dict, data)
            return_dict.update({"loss": loss})
            self.step(loss, mode=mode)
            # logging
            self.update_epoch_loss_stats(epoch_loss_stats, return_dict)
            self._log(data, return_dict, epoch, self.total_iterations[mode], mode)
        # log epoch loss stats
        self._log_epoch_loss_stats(
            epoch_loss_stats, epoch, self.total_iterations[mode], mode
        )

    def break_epoch(self, epoch, iteration, mode):
        if mode == "train":
            return (self.max_iterations_per_epoch > -1) and (
                iteration > self.max_iterations_per_epoch
            )
        return False

    def step(self, loss, mode):
        if mode == "train":
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def log(self, x, return_dict, epoch, iteration, mode):
        pass

    def update_epoch_loss_stats(self, epoch_loss_stats, return_dict):
        pass

    def log_epoch_loss_stats(self, epoch_loss_stats, epoch, iteration, mode):
        # log every log_interval in training, else log every test epoch
        if (iteration % self.args.log_interval == 0) or ("test" in mode):
            # take mean of lists in epoch_loss_stats
            epoch_loss_stats = {k: list_mean(l) for k, l in epoch_loss_stats.items()}
            dump_json(
                epoch_loss_stats,
                os.path.join(self.args.save_dir, "models", f"{mode}_loss_stats.txt"),
            )
        # log to WandB accumulated test stats for fast comparison
        if self.args.wandb and ("test" in mode):
            wandb.log({f"acc/{mode}/{k}": v for k, v in epoch_loss_stats.items()})

    def _log_epoch_loss_stats(self, epoch_loss_stats, epoch, iteration, mode):
        self.log_epoch_loss_stats(epoch_loss_stats, epoch, iteration, mode)

    def _log(self, x, return_dict, epoch, iteration, mode):
        if iteration % self.args.log_interval == 0:
            print(
                f"Epoch: {epoch} | "
                f"Iteration: {iteration} | "
                f"Loss: {return_dict['loss'].mean():.3f} | "
            )
            self.log(x, return_dict, epoch, iteration, mode)
            # checkpoint model every log interval
            self.checkpoint()

    def checkpoint(self):
        save_model(
            self.args,
            self.model,
            {
                "total_iterations": self.total_iterations,
                "args": vars(self.args),
            },
        )
