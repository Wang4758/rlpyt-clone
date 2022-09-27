from rlpyt.utils.misc import zeros
from scipy.stats import norm
import torch

def cvar_Delta(var, alpha):
    """Compute the variance based offset from the mean such that mean + cvar_Delta = cvar"""
    pdf_cdf = alpha ** (-1) * norm.pdf(norm.ppf(alpha))
    return pdf_cdf * torch.sqrt(var)

def distributional_gae(reward, value, var_value, done, bootstrap_value, var_bootstrap_value,
        discount, gae_lambda, alpha, reward_scale, advantage_dest=None, return_dest=None):
    """Special distributional version of GAE. Time-major inputs, optional other dimensions: [T], [T,B], etc.  Similar
    to `discount_return()` but using Generalized Advantage Estimation to
    compute advantages and returns. (See: rlpyt/algos/utils -> discount_return,
    generalized_advantage_estimation)"""

    advantage = advantage_dest if advantage_dest is not None else zeros(
        reward.shape, dtype=reward.dtype)
    old_advantage = zeros(advantage.shape, dtype=advantage.dtype)
    return_ = return_dest if return_dest is not None else zeros(
        reward.shape, dtype=reward.dtype)
    nd = 1 - done
    nd = nd.type(reward.dtype) if isinstance(nd, torch.Tensor) else nd

    value_n = torch.stack([*value[1:], bootstrap_value[0]])
    var_value_n = torch.stack([*var_value[1:], var_bootstrap_value[0]])
    var_return = reward**2 - value**2 + (
        2 * discount * reward * value_n
        + discount**2 * var_value_n
        + discount**2 * value_n**2
    )
    # if we are done, variance should be 0
    var_return[done.type(torch.bool)] *= 0.0
    var_value[done.type(torch.bool)] *= 0.0

    # sometimes the variances are noisy, to prevent bad effects, clamp to sensible value
    # can check in the logs that this does not actually cap the variance overall
    # if it does --> raise this limit
    var_return = torch.clamp(var_return, 1e-8, 2.5 * reward_scale).detach()
    var_value = torch.clamp(var_value, 1e-8, 2.5 * reward_scale).detach()

    Delta_bar = cvar_Delta(var_return[-1], alpha)
    Delta = cvar_Delta(var_value[-1], alpha)
    old_advantage[-1] = reward[-1] + discount * bootstrap_value * nd[-1] - value[-1]
    advantage[-1] = old_advantage[-1] + Delta_bar - Delta

    for t in reversed(range(len(reward) - 1)):
        old_delta = reward[t] + discount * value[t + 1] * nd[t] - value[t]
        Delta_bar = cvar_Delta(var_return[t], alpha) * nd[t]
        Delta = cvar_Delta(var_value[t], alpha) * nd[t]
        delta = old_delta + Delta_bar - Delta
        advantage[t] = delta + discount * gae_lambda * nd[t] * advantage[t + 1]
        old_advantage[t] = old_delta + discount * gae_lambda * nd[t] * old_advantage[t + 1]
        a_t = advantage[t]
        ao_t = old_advantage[t]
    
    return_[:] = old_advantage + value # need normal cost return for value losses

    return advantage, return_, var_return, Delta, Delta_bar