import os
from abc import abstractmethod

import torch

from src.sampling.ais import AnnealedImportanceSampling
from src.sampling.resampling import get_log_ess_from_log_particle_weights
from src.toy.distributions import normal_log_prob
from src.utils.io import dump_json


def log_mean_exp(x):
    max_, _ = torch.max(x, 1, keepdim=True)
    return torch.log(torch.mean(torch.exp(x - max_), 1)) + torch.squeeze(max_)


class DifferentiableAnnealedImportanceSampling(AnnealedImportanceSampling):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(DifferentiableAnnealedImportanceSampling, self).__init__(
            args,
            log_joint=log_joint,
            log_variational=log_variational,
            deltas=deltas,
            betas=betas,
        )
        self.transition_log_dict = dict()

    def reset_transition_log_dict(self):
        self.transition_log_dict = dict()
        self.transition_log_dict.update({"ess_mean": list()})

    def average_transition_log_dict(self):
        for key in self.transition_log_dict.keys():
            self.transition_log_dict[key] = torch.stack(
                self.transition_log_dict[key]
            ).T.mean(-1)

    def save_transition_log_dict(self, iteration):
        if iteration is not None:
            # save transition log dict to disk
            if (iteration % self.args.log_interval) == 0:
                dump_json(
                    {
                        k: torch.stack(v).detach().cpu().numpy().tolist()
                        for k, v in self.transition_log_dict.items()
                    },
                    os.path.join(
                        self.args.save_dir,
                        "logging",
                        f"transition_log_dict_{iteration}.json",
                    ),
                )

    @abstractmethod
    def transition(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward_transition(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward_transition(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_initial_momentum(self, z):
        raise NotImplementedError

    @abstractmethod
    def initialize_elbo(self, z, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_last_elbo_increment(self, z, **kwargs):
        raise NotImplementedError

    def get_weight(
        self, z, z_new, k, z_new_forward_log_prob, z_backward_log_prob, **kwargs
    ):
        numerator = z_backward_log_prob
        denominator = z_new_forward_log_prob
        return numerator, denominator

    def get_elbo_increment(self, log_numerator, log_denominator, **kwargs):
        log_normalization_increment = log_numerator - log_denominator
        return log_normalization_increment

    def forward(self, z, **kwargs):
        # grab data from kwargs
        # epoch = kwargs.get("epoch")
        total_iteration = kwargs.get("total_iteration")
        # reset additional logging parameters
        self.reset_transition_log_dict()
        with torch.set_grad_enabled(self.training):
            v = self.get_initial_momentum(z)
            # initialize elbo
            elbo, initial_log_variational_post = self.initialize_elbo(z, v=v, **kwargs)
            # transitions
            for k in range(1, self.K + 1):
                # log ess in previous log gamma_k distribution
                log_gamma_k = self.log_gamma(self.betas[k - 1], z, **kwargs)
                if v is not None:
                    log_momentum_k = normal_log_prob(
                        v,
                        torch.zeros_like(v),
                        self.mass_matrix.precision(),
                        self.mass_matrix.logdet(),
                    )
                else:
                    log_momentum_k = 0.0
                log_increment_k = log_gamma_k + log_momentum_k
                # log ESS
                self.transition_log_dict["ess_mean"].append(
                    get_log_ess_from_log_particle_weights(elbo + log_increment_k)
                    .detach()
                    .exp()
                )
                # transition
                (
                    z_new,
                    v_new,
                    z_new_forward_log_prob,
                    z_backward_log_prob,
                ) = self.transition(z, k, v=v, **kwargs)
                # compute weight
                numerator, denominator = self.get_weight(
                    z,
                    z_new,
                    k,
                    z_new_forward_log_prob,
                    z_backward_log_prob,
                    v=v,
                    v_new=v_new,
                    **kwargs,
                )
                # update elbo
                log_normalization_increment = self.get_elbo_increment(
                    numerator, denominator
                )
                elbo = elbo + log_normalization_increment
                # update z's and momentum variables
                z = z_new
                v = v_new

            # last step
            last_log_joint = self.target_dist.log_prob(z, **kwargs)
            # update elbo with terminal distribution
            last_log_normalization_increment = self.get_last_elbo_increment(
                z, last_log_joint=last_log_joint, v=v
            )
            elbo = elbo + last_log_normalization_increment
            # average and save transition log dict
            self.save_transition_log_dict(total_iteration)
            self.average_transition_log_dict()

        return {
            "elbo": log_mean_exp(elbo),
            #
            "last_z": z,
            "last_log_joint": last_log_joint.mean(-1),
            "initial_log_variational_post": initial_log_variational_post.mean(-1),
        } | self.transition_log_dict
