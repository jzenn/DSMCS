import math
from contextlib import nullcontext

import torch
from torch import nn


def get_log_ess_from_log_particle_weights(log_particle_weights):
    normalized_log_particle_weights = log_particle_weights - torch.logsumexp(
        log_particle_weights, dim=-1, keepdim=True
    )
    log_ess = 2.0 * torch.logsumexp(normalized_log_particle_weights, dim=-1) - (
        torch.logsumexp(2 * normalized_log_particle_weights, dim=-1)
    )
    return log_ess


def get_resampling_callable(args):
    if args.resampling == "no":
        resampling_callable = lambda *arguments: no_resampling(*arguments, grad=None)
    elif args.resampling == "gst":
        resampling_callable = lambda *arguments: gst_resampling(*arguments, grad=True)
    elif args.resampling == "cat":
        resampling_callable = lambda *arguments: gst_resampling(*arguments, grad=False)
    elif args.resampling == "bern-gst":
        resampling_callable = lambda *arguments: gst_resampling_bernoulli(
            *arguments, grad=True
        )
    elif args.resampling == "bern-cat":
        resampling_callable = lambda *arguments: gst_resampling_bernoulli(
            *arguments, grad=False
        )
    else:
        raise NotImplementedError
    return resampling_callable


def no_resampling(z, v, log_particle_weights, log_ess, args, grad):
    return {
        "z_resampled": z,
        "v_resampled": v,
        "updated_log_particle_weights": log_particle_weights.clone(),
    }


def masked_softmax(logits, mask=None, dim=-1):
    # taken from https://github.com/chijames/GST/blob/master/model/basic.py
    eps = 1e-20
    probs = nn.functional.softmax(logits, dim=dim)
    if mask is not None:
        mask = mask.float()
        ddif = len(probs.shape) - len(mask.shape)
        mask = mask.view(mask.shape + (1,) * ddif) if ddif > 0 else mask
        probs = probs * mask + eps
        probs = probs / probs.sum(dim, keepdim=True)
    return probs


def weighted_logsumexp(t, w, dim):
    maxes = torch.max(t, dim=dim, keepdim=True)[0]
    sum_exp = torch.sum(w * torch.exp(t - maxes), dim=dim, keepdim=True)
    log_sum_exp = maxes + torch.log(sum_exp + 1e-8)
    return log_sum_exp


def gst_mover(logits, temperature=1.0, mask=None, hard=True, gap=1.0, detach=True):
    # taken from https://github.com/chijames/GST/blob/master/model/basic.py
    logits_cpy = logits.detach() if detach else logits
    probs = masked_softmax(logits_cpy, mask)
    m = torch.distributions.one_hot_categorical.OneHotCategorical(probs=probs)
    action = m.sample()
    argmax = probs.argmax(dim=-1, keepdim=True)

    action_bool = action.bool()
    max_logits = torch.gather(logits_cpy, -1, argmax)
    move = (max_logits - logits_cpy) * action

    if type(gap) != float:
        pi = probs[action_bool] * (1 - 1e-5)  # for numerical stability
        gap = (-(1 - pi).log() / pi).view(logits.shape[:-1] + (1,))
    move2 = (logits_cpy + (-max_logits + gap)).clamp(min=0.0)
    move2[action_bool] = 0.0
    logits = logits + (move - move2)

    logits = logits - logits.mean(dim=-1, keepdim=True)
    prob = masked_softmax(logits=logits / temperature, mask=mask)
    action = action - prob.detach() + prob if hard else prob
    return action.reshape(logits.shape)


def gst_resampling(z, v, log_particle_weights, log_ess, args, grad):
    B, N = log_particle_weights.shape[0], log_particle_weights.shape[1]
    # resample
    with nullcontext() if grad else torch.no_grad():
        one_hot_index = gst_mover(
            logits=torch.repeat_interleave(log_particle_weights.unsqueeze(-2), N, -2),
            temperature=args.gumbel_tau,
            hard=True,
            gap=1.0,
        )
    # create resampled z and v
    z_resampled = one_hot_index @ z
    if v is not None:
        v_resampled = one_hot_index @ v
    else:
        v_resampled = None
    # update log particle weights
    updated_log_particle_weights = torch.full(
        (B, N), -math.log(N), device=log_particle_weights.device
    )
    return {
        "z_resampled": z_resampled,
        "v_resampled": v_resampled,
        "updated_log_particle_weights": updated_log_particle_weights,
    }


def gst_resampling_bernoulli(z, v, log_particle_weights, log_ess, args, grad):
    B, N = z.shape[0], z.shape[1]
    # create logits for reparametrized Bernoulli decision
    logsumexp_weights = torch.ones((B, 2), device=z.device)
    logsumexp_weights[:, 1] = -1.0
    log_ess_unit_interval = weighted_logsumexp(
        torch.stack([log_ess, torch.zeros((B,), device=z.device)], -1),
        logsumexp_weights,
        -1,
    ).squeeze(-1) - math.log(N - 1)
    log_ess_extended = torch.stack(
        [
            weighted_logsumexp(
                torch.stack(
                    [torch.zeros((B,), device=z.device), log_ess_unit_interval], -1
                ),
                logsumexp_weights,
                -1,
            ).squeeze(-1),
            log_ess_unit_interval,
        ],
        -1,
    )
    # sample Bernoulli decision and resample
    with nullcontext() if grad else torch.no_grad():
        # Bernoulli decision
        bernoulli_one_hot_index = gst_mover(
            logits=log_ess_extended[:, 0],
            temperature=args.gumbel_tau,
            hard=True,
            gap=1.0,
        )
        # resampling
        one_hot_index = gst_mover(
            logits=torch.repeat_interleave(
                log_particle_weights.unsqueeze(-2), args.n_particles, -2
            ),
            temperature=args.gumbel_tau,
            hard=True,
            gap=1.0,
        )
    # create resampled z and v
    z_resampled = (1 - bernoulli_one_hot_index.unsqueeze(-1).unsqueeze(-1)) * (
        one_hot_index @ z
    ) + bernoulli_one_hot_index.unsqueeze(-1).unsqueeze(-1) * z
    if v is not None:
        v_resampled = (1 - bernoulli_one_hot_index.unsqueeze(-1).unsqueeze(-1)) * (
            one_hot_index @ v
        ) + bernoulli_one_hot_index.unsqueeze(-1).unsqueeze(-1) * v
    else:
        v_resampled = None
    # update log particle weights
    updated_log_particle_weights = log_particle_weights.clone()
    updated_log_particle_weights[bernoulli_one_hot_index == 0] = torch.full(
        (N,), -math.log(N), device=z.device
    )
    return {
        "z_resampled": z_resampled,
        "v_resampled": v_resampled,
        "updated_log_particle_weights": updated_log_particle_weights,
    }
