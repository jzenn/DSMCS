import math
from abc import ABC

import torch

from src.sampling.dais import DifferentiableAnnealedImportanceSampling
from src.sampling.resampling import (
    get_log_ess_from_log_particle_weights,
    get_resampling_callable,
)
from src.toy.distributions import normal_log_prob


class DifferentiableSequentialMonteCarloSampler(
    DifferentiableAnnealedImportanceSampling, ABC
):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(DifferentiableSequentialMonteCarloSampler, self).__init__(
            args,
            log_joint=log_joint,
            log_variational=log_variational,
            deltas=deltas,
            betas=betas,
        )
        self.transition_log_dict = dict()
        self.resampling_callable = get_resampling_callable(args)

    def reset_transition_log_dict(self):
        self.transition_log_dict = dict()
        self.transition_log_dict.update({"ess_mean": list()})

    def get_weight(
        self, z, z_new, k, z_new_forward_log_prob, z_backward_log_prob, **kwargs
    ):
        # no momentum variable (Langevin)
        numerator = z_backward_log_prob + self.log_gamma(self.betas[k], z_new, **kwargs)
        denominator = z_new_forward_log_prob + self.log_gamma(
            self.betas[k - 1], z, **kwargs
        )
        # momentum variable (Langevin + Momentum)
        v = kwargs["v"]
        v_new = kwargs["v_new"]
        if v is not None:
            momentum_precision, momentum_log_det = (
                self.mass_matrix.precision(),
                self.mass_matrix.logdet(),
            )
            numerator = numerator + normal_log_prob(
                v_new, torch.zeros_like(v_new), momentum_precision, momentum_log_det
            )
            denominator = denominator + normal_log_prob(
                v, torch.zeros_like(v), momentum_precision, momentum_log_det
            )
        return numerator, denominator

    def resample(self, z, log_particle_weights, **kwargs):
        v = kwargs["v"]
        log_ess = kwargs["log_ess"]
        # resample
        resampling_dict = self.resampling_callable(
            z, v, log_particle_weights, log_ess, self.args
        )
        z_resampled = resampling_dict["z_resampled"]
        v_resampled = resampling_dict["v_resampled"]
        updated_log_particle_weights = resampling_dict["updated_log_particle_weights"]
        return z_resampled, v_resampled, updated_log_particle_weights, resampling_dict

    def get_elbo_increment(self, log_numerator, log_denominator, **kwargs):
        log_particle_weights = kwargs["log_particle_weights"]
        log_unnormalized_incremental_particle_weights = log_numerator - log_denominator
        log_unnormalized_particle_weights = (
            log_particle_weights + log_unnormalized_incremental_particle_weights
        )
        new_log_particle_weights = log_unnormalized_particle_weights - torch.logsumexp(
            log_unnormalized_particle_weights, -1, keepdim=True
        )
        log_normalization_increment = torch.logsumexp(
            log_unnormalized_particle_weights, -1
        )
        return log_normalization_increment, new_log_particle_weights

    def forward(self, z, **kwargs):
        # grab data from kwargs
        # epoch = kwargs.get("epoch")
        total_iteration = kwargs.get("total_iteration")
        B, N = z.shape[0], z.shape[1]
        # reset additional logging parameters
        self.reset_transition_log_dict()
        with torch.set_grad_enabled(self.training):
            v = self.get_initial_momentum(z)
            # initialize elbo and log particle weights
            elbo, initial_log_variational_post = self.initialize_elbo(z, v=v, **kwargs)
            log_particle_weights = torch.full((B, N), -math.log(N), device=z.device)
            # transitions
            for k in range(1, self.K + 1):
                # resample
                if k > 1:
                    log_ess = get_log_ess_from_log_particle_weights(
                        log_particle_weights
                    )
                    z, v, log_particle_weights, resampling_dict = self.resample(
                        z, log_particle_weights, v=v, log_ess=log_ess, **kwargs
                    )
                    # log ESS
                    self.transition_log_dict["ess_mean"].append(log_ess.detach().exp())
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
                    **kwargs
                )
                # update elbo
                (
                    log_normalization_increment,
                    log_particle_weights,
                ) = self.get_elbo_increment(
                    numerator, denominator, log_particle_weights=log_particle_weights
                )
                elbo = elbo + log_normalization_increment
                # update z's and momentum variables
                z = z_new
                v = v_new

            # log last step
            with torch.no_grad():
                last_log_joint = self.target_dist.log_prob(z, **kwargs)

            # average and save transition log dict
            self.save_transition_log_dict(total_iteration)
            self.average_transition_log_dict()

        return {
            # elbo over batch (particles are already averaged by SMCS)
            "elbo": elbo,
            #
            "last_z": z,
            # *.mean(-1) is to average over particles
            "last_log_joint": last_log_joint.mean(-1),
            "initial_log_variational_post": initial_log_variational_post.mean(-1),
        } | self.transition_log_dict
