import wandb

from src.trainer.trainer import Trainer


class StaticTargetTrainer(Trainer):
    def __init__(self, args, model, optimizer, scheduler, data_loaders, device):
        super(StaticTargetTrainer, self).__init__(
            args, model, optimizer, scheduler, data_loaders, device
        )

    def prepare_data(self, data_batch):
        return None

    def forward_model(self, data, **kwargs):
        return_dict = self.model(data, **kwargs)
        return return_dict

    def loss(self, return_dict, data):
        loss = -return_dict["elbo"].mean()
        return loss

    def update_epoch_loss_stats(self, epoch_loss_stats, return_dict):
        # initialize
        if len(epoch_loss_stats) == 0:
            epoch_loss_stats.update(
                {
                    "elbo": list(),
                    "last_log_joint": list(),
                    "initial_log_variational_post": list(),
                }
            )
        # update by mean of batch
        epoch_loss_stats["elbo"].append(return_dict["elbo"].mean(-1).item())
        epoch_loss_stats["last_log_joint"].append(
            return_dict["last_log_joint"].mean(-1).item()
        )
        epoch_loss_stats["initial_log_variational_post"].append(
            return_dict["initial_log_variational_post"].mean(-1).item()
        )
        return epoch_loss_stats

    def log(self, x, return_dict, epoch, iteration, mode):
        if self.args.wandb:
            # metrics and other stats
            log_dict = {
                f"{mode}/main/epoch": epoch,
                f"{mode}/main/elbo": return_dict["elbo"].mean(-1).item(),
                f"{mode}/main/last_log_joint": return_dict["last_log_joint"].mean(-1),
                f"{mode}/main/initial_log_variational_post": return_dict[
                    "initial_log_variational_post"
                ].mean(-1),
            }
            # transition alpha
            for key in [
                "transition_alpha_mean",
                "transition_alpha_median",
                "ess_mean",
            ]:
                if key in return_dict:
                    log_dict.update({f"{mode}/sub/{key}": return_dict[key].mean(-1)})
            # log to WandB
            wandb.log(log_dict)
