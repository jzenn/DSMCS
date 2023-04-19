import torch

from src.sampling.dais import DifferentiableAnnealedImportanceSampling
from src.sampling.dsmcs import DifferentiableSequentialMonteCarloSampler
from src.toy.distributions import normal_log_prob, sample_normal


class LangevinDiffusionDAIS(DifferentiableAnnealedImportanceSampling):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(LangevinDiffusionDAIS, self).__init__(
            args,
            log_joint=log_joint,
            log_variational=log_variational,
            deltas=deltas,
            betas=betas,
        )

    def get_initial_momentum(self, z):
        # no momentum for Langevin
        return None

    def initialize_elbo(self, z, **kwargs):
        # log prior
        initial_log_variational_post = self.initial_dist.log_prob(z, **kwargs)
        # initialize elbo
        elbo = -initial_log_variational_post
        return elbo, initial_log_variational_post

    def get_last_elbo_increment(self, z, **kwargs):
        last_log_joint = kwargs["last_log_joint"]
        elbo_increment = last_log_joint
        return elbo_increment

    def get_normal_params(self, delta):
        covariance_factor = 2 * delta
        covariance_diagonal = covariance_factor * torch.ones(
            self.D, device=delta.device
        )
        cholesky_factor = torch.diag(torch.sqrt(covariance_diagonal))
        precision = torch.diag(1.0 / covariance_diagonal)
        logdet = torch.sum(torch.log(covariance_diagonal))
        return {
            "cholesky_factor": cholesky_factor,
            "precision": precision,
            "logdet": logdet,
        }

    def forward_transition(self, z, k, normal_params, **kwargs):
        forward_mean = z + self.deltas[k] * self.grad_log_gamma(
            self.betas[k], z, **kwargs
        )
        z_new = sample_normal(
            forward_mean, normal_params["cholesky_factor"], (z.shape[0], z.shape[1])
        )
        z_new_forward_log_prob = normal_log_prob(
            z_new, forward_mean, normal_params["precision"], normal_params["logdet"]
        )
        return z_new, z_new_forward_log_prob

    def backward_transition(self, z, z_new, k, normal_params, **kwargs):
        z_new_grad_log_annealed_prob = self.grad_log_gamma(
            self.betas[k], z_new, **kwargs
        )
        backward_mean = z_new + self.deltas[k] * z_new_grad_log_annealed_prob
        z_backward_log_prob = normal_log_prob(
            z, backward_mean, normal_params["precision"], normal_params["logdet"]
        )
        return z_backward_log_prob

    def transition(self, z, k, **kwargs):
        normal_params = self.get_normal_params(self.deltas[k])
        # forward transition
        z_new, z_new_forward_log_prob = self.forward_transition(
            z, k, normal_params, **kwargs
        )
        # backward transition
        z_backward_log_prob = self.backward_transition(
            z, z_new, k, normal_params, **kwargs
        )
        return z_new, None, z_new_forward_log_prob, z_backward_log_prob


class LangevinDiffusionDSMCS(
    LangevinDiffusionDAIS, DifferentiableSequentialMonteCarloSampler
):
    def __init__(self, args, log_joint, log_variational, deltas, betas):
        super(LangevinDiffusionDAIS, self).__init__(
            args,
            log_joint=log_joint,
            log_variational=log_variational,
            deltas=deltas,
            betas=betas,
        )

    def initialize_elbo(self, z, **kwargs):
        _, initial_log_variational_post = super().initialize_elbo(z, **kwargs)
        elbo = torch.tensor(0.0, device=z.device)
        return elbo, initial_log_variational_post
