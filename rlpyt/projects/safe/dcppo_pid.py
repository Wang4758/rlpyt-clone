"""
    Modified version of cppo_pid to support Worst-Case Actor.
    Instead of pid lagrange it can also use Adam and primal (crpo by Xu 2021) based optimization
"""

######################################################################
# Algorithm file.
######################################################################


from rlpyt.projects.safe.running_mean_std import RunningMeanStd
from rlpyt.projects.safe.util import distributional_gae
import torch
from collections import namedtuple, deque

from scipy.stats import norm as sp_norm

from rlpyt.algos.pg.base import PolicyGradientAlgo, OptInfo
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.tensor import valid_mean
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.misc import iterate_mb_idxs
from rlpyt.algos.utils import (discount_return,
    generalized_advantage_estimation, valid_from_done)
from rlpyt.models.utils import strip_ddp_state_dict
from rlpyt.utils.logging import logger
import numpy as np
import torch.nn.functional as F

# whatever is in here has to be set in the algo below and then gets logged
OptInfoCost = namedtuple("OptInfoCost", OptInfo._fields + ("costPenalty",
    "costLimit", "valueError", "cvalueError", "valueAbsError", "cvalueAbsError", "c_var_value", "cvar_q_delta", 
    "cvar_v_delta", "c_return", "pid_i", "pid_p", "pid_d", "pid_o", "c_loss", "pi_loss", "stopIter", "kl", "cumCost", 
    "delta", "c_var_value_loss", "ep_cost_var", "ep_cost_cvar", "c_advantage", "advantage", "exp_var", "c_exp_var"))

LossInputs = namedarraytuple("LossInputs",
    ["agent_inputs", "action", "return_", "advantage", "valid", "old_dist_info",
    "c_return", "c_advantage", "c_return_var", "value_old", "c_value_old"])


def compute_cvar(mean, var, alpha):
    """Compute the CVaR as the expected value of the UPPER quantile (1-alpha)"""
    pdf_cdf_alpha = alpha**(-1)*sp_norm.pdf(sp_norm.ppf(alpha))
    offset = pdf_cdf_alpha * torch.sqrt(var)
    return (mean + offset, offset)

def explained_variance(pred, v):
    """Computes 1 - Var(v-pred) / Var(pred). 0 => bad, 1 => perfect, <0 => worse than not predicting anything"""
    var_v = torch.var(v)
    return 1 - torch.var(v-pred) / (var_v + 1e-7)

class DCppoPID(PolicyGradientAlgo):
    """
        This is the algo implementation. It is adapted and extended from the `CppoPID` one. The class has 2 main methods that the framework calls. 
        1) `initialize`: self explanatory...
        2) `optimize_agent`: firstly processes the samples in `process_returns`, 
        then updates lambda in `update_cost_penalty` 
        and finally uses `loss` to compute the loss per minibatch.

        **All arguments provided to `__init__` are available on self**. Even if they are shown as unused they get saved in the constructor.
        This is a bit quirky but it was like that before. (see `save__init__args(locals())`)
    """

    opt_info_fields = OptInfoCost._fields

    def __init__(
            self,
            discount=0.99,
            learning_rate=0.001,
            value_loss_coeff=1.,
            entropy_loss_coeff=0.,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            clip_grad_norm=1.,
            initial_optim_state_dict=None,
            gae_lambda=1.0,  # was 0.97 before...
            minibatches=1,
            epochs=8,
            ratio_clip=0.1,
            linear_lr_schedule=False,
            normalize_advantage=False,
            cost_discount=None,  # if None, defaults to discount.
            cost_gae_lambda=None,
            cost_value_loss_coeff=None,
            ep_cost_ema_alpha=0,  # 0 for hard update, 1 for no update.
            use_ep_cost_rms=False, # if this is True, ep_cost_ema_alpha is not used!
            ep_cost_rms_max_count=25_000,
            objective_penalized=True,  # False for reward-only learning
            learn_c_value=True,  # Also False for reward-only learning
            penalty_init=1.,
            cost_limit=25,
            cost_scale=1.,  # divides; applied to raw cost and cost_limit
            normalize_cost_advantage=False,
            pid_Kp=0,
            pid_Ki=1,
            pid_Kd=0,
            pid_d_delay=10,
            pid_delta_p_ema_alpha=0.95,  # 0 for hard update, 1 for no update
            pid_delta_d_ema_alpha=0.95,
            pid_i_decay = 1.0,
            pid_i_max = 100000.0,
            sum_norm=True,  # L = (J_r - lam * J_c) / (1 + lam); lam <= 0
            diff_norm=False,  # L = (1 - lam) * J_r - lam * J_c; 0 <= lam <= 1
            penalty_max=100,  # only used if sum_norm=diff_norm=False
            step_cost_limit_steps=None,  # Change the cost limit partway through
            step_cost_limit_value=None,  # New value.
            use_beta_kl=False,
            use_beta_grad=False,
            record_beta_kl=False,
            record_beta_grad=False,
            beta_max=10,
            beta_ema_alpha=0.9,
            beta_kl_epochs=1,
            reward_scale=1,  # multiplicative (unlike cost_scale)
            lagrange_quadratic_penalty=False,
            quadratic_penalty_coeff=1,
            wc_risk_lvl: float = 1.,
            wc_update_advantage: bool = True,
            use_primal: bool = False,
            use_clipped_cost_obj: bool = True,
            use_kl_limit = False,
            target_kl=0.01, # from safety starter agents ppo-lag
            kl_margin=1.2, # from safety starter agents ppo-lag,
            learn_penalty=False, # this disables PID and instead learns the cost_penalty
            penalty_lr = 5e-2, # used if learn_penalty
            ep_cost_buffer_sz=400,
            use_clipped_vf_loss=False,
            ):
        assert not (learn_penalty and use_primal), "Learning a penalty with primal optimization does not make sense :)"
        assert learn_c_value or not objective_penalized
        assert (step_cost_limit_steps is None) == (step_cost_limit_value is None)
        assert not (sum_norm and diff_norm)
        assert not (use_beta_kl or use_beta_grad), "not supported anymore"
        cost_discount = discount if cost_discount is None else cost_discount
        cost_gae_lambda = (gae_lambda if cost_gae_lambda is None else
            cost_gae_lambda)
        cost_value_loss_coeff = (value_loss_coeff if cost_value_loss_coeff is
            None else cost_value_loss_coeff)
        if optim_kwargs is None:
            optim_kwargs = dict()
        save__init__args(locals())
        self.cost_limit /= self.cost_scale
        if step_cost_limit_value is not None:
            self.step_cost_limit_value /= self.cost_scale
        self._beta_kl = 1.
        self._beta_grad = 1.
        self.beta_min = 1. / self.beta_max
        self.wc_risk_lvl = wc_risk_lvl
        self.use_primal = use_primal
        self.is_costlimit_violated = False # used if is_primal == True
        self.cum_cost = 0
        if use_ep_cost_rms:
            self.ep_cost_rms = RunningMeanStd(shape=(1,), max_count=ep_cost_rms_max_count)

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self._batch_size = self.batch_spec.size // self.minibatches  # For logging.
        if self.linear_lr_schedule:
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.optimizer,
                lr_lambda=lambda itr: (self.n_itr - itr) / self.n_itr)  # Step once per itr.
            self._ratio_clip = self.ratio_clip  # Save base value.
        if self.step_cost_limit_steps is None:
            self.step_cost_limit_itr = None
        else:
            self.step_cost_limit_itr = int(self.step_cost_limit_steps //
                (self.batch_spec.size * self.world_size))
            # print("\n\n step cost itr: ", self.step_cost_limit_itr, "\n\n")
        self._ep_cost_ema = self.cost_limit  # No derivative at start.
        self._ep_cost_var_ema = torch.sqrt(torch.tensor(1.0))
        self._ddp = self.agent._ddp
        self.pid_i = self.cost_penalty = self.penalty_init
        self.cost_ds = deque(maxlen=self.pid_d_delay)
        self.cost_ds.append(0)
        self._delta_p = 0
        self._cost_d = 0
        # if learning penalty with adam, setup optimizer and underlying parameter
        if self.learn_penalty:
            param_init = np.log(max(np.exp(self.penalty_init)-1, 1e-8))
            self.cost_penalty_param = torch.tensor(param_init, requires_grad=True)
            self.penalty_optimizer = torch.optim.Adam([self.cost_penalty_param], lr=self.penalty_lr)
        # estimate the episodic mean and std of cost using a running means and std model.
        # Might be more accurate and less noisy but takes time to converge
        if self.use_ep_cost_rms:
            logger.log(f'Using EpCostRMS with max_count = {self.ep_cost_rms_max_count}')

    def optimize_agent(self, itr, samples):
        """Called by MinibatchRl with the new samples that this function uses to perform gradient descent updates"""
        opt_info = OptInfoCost(*([] for _ in range(len(OptInfoCost._fields))))
        recurrent = self.agent.recurrent
        agent_inputs = AgentInputs(  # Move inputs to device once, index there.
            observation=samples.env.observation,
            prev_action=samples.agent.prev_action,
            prev_reward=samples.env.prev_reward,
        )
        agent_inputs = buffer_to(agent_inputs, device=self.agent.device)
        (return_, advantage, valid, c_return, c_advantage, c_return_var, c_var_value, cvar, cvar_return_offset,
            cvar_value_offset, ep_cost_avg, value_old, c_value_old) = self.process_returns(itr, samples, opt_info)
        loss_inputs = LossInputs(  # So can slice all.
            agent_inputs=agent_inputs,
            action=samples.agent.action,
            return_=return_,
            advantage=advantage,
            valid=valid,
            old_dist_info=samples.agent.agent_info.dist_info,
            c_return=c_return,  # Can be None.
            c_advantage=c_advantage,
            c_return_var=c_return_var,
            value_old = value_old,
            c_value_old = c_value_old,
        )
        opt_info.c_var_value.append(torch.mean(c_var_value).cpu())
        opt_info.c_return.append(torch.mean(c_return).cpu())
        opt_info.c_advantage.append(torch.mean(c_advantage).cpu())
        opt_info.advantage.append(torch.mean(advantage).cpu())
        opt_info.cvar_q_delta.append(torch.mean(cvar_return_offset).cpu())
        opt_info.cvar_v_delta.append(torch.mean(cvar_value_offset).cpu())
        if (self.step_cost_limit_itr is not None and
                self.step_cost_limit_itr == itr):
            self.cost_limit = self.step_cost_limit_value
        opt_info.costLimit.append(self.cost_limit)
        opt_info.cumCost.append(self.cum_cost)

        # PID update here:
        self.update_cost_penalty(ep_cost_avg, opt_info)

        if hasattr(self.agent, "update_obs_rms"):
            self.agent.update_obs_rms(agent_inputs.observation)
            if itr == 0:
                opt_info.loss.append(0)
                opt_info.pi_loss.append(0)
                opt_info.c_loss.append(0)
                return opt_info  # Sacrifice the first batch to get obs stats.

        if recurrent:
            # Leave in [B,N,H] for slicing to minibatches.
            init_rnn_state = samples.agent.agent_info.prev_rnn_state[0]  # T=0.
        T, B = samples.env.reward.shape[:2]
        # If recurrent, use whole trajectories, only shuffle B; else shuffle all.
        batch_size = B if self.agent.recurrent else T * B
        mb_size = batch_size // self.minibatches

        for epoch in range(self.epochs):
            for idxs in iterate_mb_idxs(batch_size, mb_size, shuffle=True):
                T_idxs = slice(None) if recurrent else idxs % T
                B_idxs = idxs if recurrent else idxs // T
                self.optimizer.zero_grad()
                rnn_state = init_rnn_state[B_idxs] if recurrent else None
                # NOTE: if not recurrent, will lose leading T dim, should be OK.
                loss, entropy, perplexity, value_errors, abs_value_errors, pi_loss, c_loss, approx_kl, c_var_value_loss = self.loss(
                    *loss_inputs[T_idxs, B_idxs], rnn_state)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(), self.clip_grad_norm)
                self.optimizer.step()

                opt_info.loss.append(loss.item())
                opt_info.pi_loss.append(pi_loss.mean().item())
                opt_info.c_loss.append(c_loss.mean().item())
                opt_info.gradNorm.append(grad_norm.item())
                opt_info.entropy.append(entropy.item())
                opt_info.perplexity.append(perplexity.item())
                opt_info.valueError.extend(value_errors[0][::10].numpy())
                opt_info.cvalueError.extend(value_errors[1][::10].numpy())
                opt_info.valueAbsError.extend(abs_value_errors[0][::10].numpy())
                opt_info.cvalueAbsError.extend(abs_value_errors[1][::10].numpy())
                opt_info.c_var_value_loss.append(c_var_value_loss.mean().item())

                self.update_counter += 1
            
            approx_kl_mean = approx_kl.mean().item()
            if self.use_kl_limit and approx_kl_mean > self.kl_margin * self.target_kl:
                print(f'Early stopping at epoch {epoch} because max kl div reached')
                break
        
        opt_info.stopIter.append(epoch)
        opt_info.kl.append(approx_kl_mean)

        if self.linear_lr_schedule:
            self.lr_scheduler.step()
            self.ratio_clip = self._ratio_clip * (self.n_itr - itr) / self.n_itr

        return opt_info

    def update_cost_penalty(self, ep_cost_avg, opt_info):
        """ Update the current cost penalty based on the episode cost (also when using primal.) """

        opt_info.ep_cost_var.append(self._ep_cost_var_ema.item())

        if self.wc_risk_lvl < 1.0:
            ep_cost_cvar, _ = compute_cvar(ep_cost_avg, self._ep_cost_var_ema, self.wc_risk_lvl)
            delta = float(ep_cost_cvar - self.cost_limit)  # ep_cost_avg: tensor
            opt_info.ep_cost_cvar.append(ep_cost_cvar.item())
        else:
            delta = float(ep_cost_avg - self.cost_limit)  # ep_cost_avg: tensor
            opt_info.ep_cost_cvar.append(ep_cost_avg)
        # set boolean flag for primal optimization to know which objective to optimize
        self.is_costlimit_violated = delta > (0.0 / self.cost_scale) # crpo paper uses .5 tolerance --> do the same here, maybe hyperparam later?
        if not self.learn_penalty:
            self.pid_i = max(0., self.pid_i + delta * self.pid_Ki) * self.pid_i_decay
            if self.diff_norm:
                self.pid_i = max(0., min(1., self.pid_i))
            self.pid_i = max(-.5, min(self.pid_i_max, self.pid_i))
            a_p = self.pid_delta_p_ema_alpha
            self._delta_p *= a_p
            self._delta_p += (1 - a_p) * delta
            a_d = self.pid_delta_d_ema_alpha
            self._cost_d *= a_d
            self._cost_d += (1 - a_d) * float(ep_cost_avg)
            pid_d = max(0., self._cost_d - self.cost_ds[0])
            pid_o = (self.pid_Kp * self._delta_p + self.pid_i +
                self.pid_Kd * pid_d)
            self.cost_penalty = max(0., min(pid_o, 100.))
            if self.diff_norm:
                self.cost_penalty = min(1., self.cost_penalty)
            else:
                self.cost_penalty = min(self.cost_penalty, self.penalty_max)
            self.cost_ds.append(self._cost_d)
            opt_info.pid_i.append(self.pid_i)
            opt_info.pid_p.append(self._delta_p)
            opt_info.pid_d.append(pid_d)
            opt_info.pid_o.append(pid_o)
            opt_info.costPenalty.append(self.cost_penalty)
        else:
            # update first
            self.penalty_optimizer.zero_grad()
            penalty_loss = -self.cost_penalty_param * delta
            penalty_loss.backward(retain_graph=True)
            self.penalty_optimizer.step()

            # compute penalty after
            self.cost_penalty = F.softplus(self.cost_penalty_param).item()

            opt_info.costPenalty.append(self.cost_penalty)
        
        opt_info.delta.append(delta)
        

    def loss(self, agent_inputs, action, return_, advantage, valid, old_dist_info,
            c_return, c_advantage, c_return_var, value_old, c_value_old, init_rnn_state=None):
        if init_rnn_state is not None:
            # [B,N,H] --> [N,B,H] (for cudnn).
            init_rnn_state = buffer_method(init_rnn_state, "transpose", 0, 1)
            init_rnn_state = buffer_method(init_rnn_state, "contiguous")
            dist_info, value, _rnn_state = self.agent(*agent_inputs, init_rnn_state)
        else:
            dist_info, value = self.agent(*agent_inputs)
        dist = self.agent.distribution

        approx_kl = dist.kl(old_dist_info=old_dist_info, new_dist_info=dist_info)

        ratio = dist.likelihood_ratio(action, old_dist_info=old_dist_info,
            new_dist_info=dist_info)
        surr_1 = ratio * advantage
        clipped_ratio = torch.clamp(ratio, 1. - self.ratio_clip,
            1. + self.ratio_clip)
        surr_2 = clipped_ratio * advantage
        surrogate = torch.min(surr_1, surr_2)
        pi_loss = - valid_mean(surrogate, valid)

        vpredclipped = value_old + torch.clip(value.value - value_old, - self.ratio_clip, self.ratio_clip)
        if self.reward_scale == 1.:
            value_error = value.value - return_
            value_error_clip = vpredclipped - return_
        else:
            value_error = value.value - (return_ / self.reward_scale)  # Undo the scaling
            value_error_clip = vpredclipped - (return_ / self.reward_scale)  # Undo the scaling
        value_se = 0.5 * value_error ** 2
        value_se_clip = 0.5 * value_error_clip ** 2
        vf_loss_tmp = torch.max(value_se, value_se_clip) if self.use_clipped_vf_loss else value_se
        value_loss = self.value_loss_coeff * valid_mean(vf_loss_tmp, valid)

        entropy = dist.mean_entropy(dist_info, valid)
        entropy_loss = - self.entropy_loss_coeff * entropy

        # calc c loss
        c_surr_1 = ratio * c_advantage
        c_surr_2 = clipped_ratio * c_advantage
        c_surrogate = torch.max(c_surr_1, c_surr_2) if self.use_clipped_cost_obj else c_surr_1 # starter agents does not use clipping for cost obj
        c_loss = valid_mean(c_surrogate, valid)

        # combine losses depending on whether primal is used or not
        if self.objective_penalized and self.use_primal:
            # switch out loss if required
            pi_loss = c_loss if self.is_costlimit_violated else pi_loss
        elif self.objective_penalized and not self.use_primal:
            # weigh the c_loss
            c_loss *= self.cost_penalty
            # and combine it with pi_loss
            if self.use_beta_kl:
                c_loss *= self._beta_kl
            elif self.use_beta_grad:
                c_loss *= self._beta_grad
            if self.diff_norm:  # (1 - lam) * R + lam * C
                pi_loss *= (1 - self.cost_penalty)
                pi_loss += c_loss
            elif self.sum_norm:  # 1 / (1 + lam) * (R + lam * C)
                pi_loss += c_loss
                pi_loss /= (1 + self.cost_penalty)
            else:
                pi_loss += c_loss

            # this we never use but it is from the original code -> leave it so not to break the settings...
            if self.lagrange_quadratic_penalty:
                quad_loss = (self.quadratic_penalty_coeff
                    * valid_mean(c_surrogate, valid)
                    * torch.max(torch.tensor(0.), self._ep_cost_ema - self.cost_limit))
                pi_loss += quad_loss

        loss = pi_loss + value_loss + entropy_loss

        if self.learn_c_value:  # Then separate cost value estimate.
            assert value.c_value is not None
            assert c_return is not None
            assert c_return_var is not None
            c_value_error = value.c_value - c_return
            c_value_se = 0.5 * c_value_error ** 2
            c_value_loss = self.cost_value_loss_coeff * valid_mean(c_value_se, valid)

            # wasserstein loss for the variance
            c_var_value_loss_before_mean = value.c_var_value + c_return_var - 2 * ((value.c_var_value * c_return_var) ** 0.5)
            c_var_value_loss = self.cost_value_loss_coeff * 0.5 * valid_mean(c_var_value_loss_before_mean ,valid)

            loss += c_value_loss
            if self.wc_risk_lvl < 1.0:
                loss += c_var_value_loss

        value_errors = (value_error.detach(), c_value_error.detach())
        if valid is not None:
            valid_mask = valid > 0
            value_errors = tuple(v[valid_mask] for v in value_errors)
        else:
            value_errors = tuple(v.view(-1) for v in value_errors)
        abs_value_errors = tuple(abs(v) for v in value_errors)
        perplexity = dist.mean_perplexity(dist_info, valid)
        return loss, entropy, perplexity, value_errors, abs_value_errors, pi_loss, c_loss, approx_kl, c_var_value_loss

    def process_returns(self, itr, samples, opt_info):
        """Preprocess samples to get returns and episodic cost from done trajectories."""
        reward, cost = samples.env.reward, samples.env.env_info.cost
        
        if hasattr(self.agent.model, "rew_rms"):
            reward_transformed: torch.Tensor = reward.unsqueeze(-1)
            self.agent.update_rew_rms(reward_transformed)

            rew_var = self.agent.model.rew_rms.var
            rew_mean = self.agent.model.rew_rms.mean
            if self.agent.model.var_clip is not None:
                rew_var = torch.clamp(rew_var, min=self.agent.model.var_clip)
            reward = (reward - rew_mean.item()) / rew_var.sqrt().item()

        cost = cost.float()
        cost /= self.cost_scale
        done = samples.env.done
        value, c_value, c_var_value = samples.agent.agent_info.value  # A named 3-tuple.
        bv, c_bv, c_var_bv = samples.agent.bootstrap_value  # A named 3-tuple.

        # prob not negative anyways, just to be safe :)
        c_var_value = torch.clamp(c_var_value, 1e-8, 1e8)

        if self.reward_scale != 1:
            reward *= self.reward_scale
            value *= self.reward_scale  # Keep the value learning the same.
            bv *= self.reward_scale

        done = done.type(reward.dtype)  # rlpyt does this in discount_returns?

        if c_value is not None:  # Learning c_value, even if reward penalized.
            if self.cost_gae_lambda == 1:  # GAE reduces to empirical discount.
                c_return = discount_return(cost, done, c_bv, self.cost_discount)
                c_advantage = c_return - c_value
            else:
                c_advantage, c_return, c_return_var, cvar_value_offset, cvar_return_offset = distributional_gae(
                    cost, c_value, c_var_value, done, c_bv, c_var_bv, self.cost_discount,
                    self.cost_gae_lambda, self.wc_risk_lvl, 1. / self.cost_scale)
                # c_advantage, c_return = generalized_advantage_estimation(
                #     cost, c_value, done, c_bv, self.cost_discount,
                #     self.cost_gae_lambda)

            cvar_return = c_return + cvar_return_offset
        else:
            # c_advantage = c_return = None
            c_return_var = c_advantage = c_return = None

        if self.gae_lambda == 1:  # GAE reduces to empirical discounted.
            return_ = discount_return(reward, done, bv, self.discount)
            advantage = return_ - value
        else:
            advantage, return_ = generalized_advantage_estimation(
                reward, value, done, bv, self.discount, self.gae_lambda)

        opt_info.exp_var.append(explained_variance(value, return_).item())
        opt_info.c_exp_var.append(explained_variance(c_value, c_return).item())

        if not self.mid_batch_reset or self.agent.recurrent:
            valid = valid_from_done(done)  # Recurrent: no reset during training.
            # "done" might stay True until env resets next batch.
            # Could probably do this formula directly on (1 - done) and use it
            # regardless of mid_batch_reset.
            ep_cost_mask = valid * (1 - torch.cat([valid[1:],
                torch.ones_like(valid[-1:])]))  # Find where valid turns OFF.
        else:
            valid = None  # OR: torch.ones_like(done)
            ep_cost_mask = done  # Everywhere a done, is episode final cost.
        ep_costs = samples.env.env_info.cum_cost[ep_cost_mask.type(torch.bool)]

        if self._ddp:
            world_size = torch.distributed.get_world_size()  # already have self.world_size
        if ep_costs.numel() > 0:  # Might not have any completed trajectories.
            ep_cost_avg = ep_costs.mean()
            self.cum_cost += ep_costs.sum().item()
            
            if not self.use_ep_cost_rms:
                ep_cost_avg /= self.cost_scale
                ep_cost_var = (ep_costs / self.cost_scale).var()
                if self._ddp:
                    eca = ep_cost_avg.to(self.agent.device)
                    torch.distributed.all_reduce(eca)
                    ep_cost_avg = eca.to("cpu")
                    ep_cost_avg /= world_size
                a = self.ep_cost_ema_alpha
                self._ep_cost_ema *= a
                self._ep_cost_ema += (1 - a) * ep_cost_avg
                self._ep_cost_var_ema *= a
                self._ep_cost_var_ema += (1 - a) * ep_cost_var
            else:
                # make the input shape [[a],[b],[c],...] such that max_count in ep_cost_rms counts
                # the number of trajectories
                rms_input = (ep_costs / self.cost_scale).flatten().unsqueeze(-1).cpu().numpy()
                self.ep_cost_rms.update(rms_input)
                self._ep_cost_ema = torch.tensor(self.ep_cost_rms.mean.item())
                self._ep_cost_var_ema = torch.tensor(self.ep_cost_rms.var.item())

        if self.normalize_advantage:
            if valid is not None:
                valid_mask = valid > 0
                adv_mean = advantage[valid_mask].mean()
                adv_std = advantage[valid_mask].std()
            else:
                adv_mean = advantage.mean()
                adv_std = advantage.std()
            if self._ddp:
                mean_std = torch.stack([adv_mean, adv_std])
                mean_std = mean_std.to(self.agent.device)
                torch.distributed.all_reduce(mean_std)
                mean_std = mean_std.to("cpu")
                mean_std /= world_size
                adv_mean, adv_std = mean_std[0], mean_std[1]
            advantage[:] = (advantage - adv_mean) / max(adv_std, 1e-6)

        if self.normalize_cost_advantage:
            if valid is not None:
                valid_mask = valid > 0
                cadv_mean = c_advantage[valid_mask].mean()
                cadv_std = c_advantage[valid_mask].std()
            else:
                cadv_mean = c_advantage.mean()
                cadv_std = c_advantage.std()
            if self._ddp:
                mean_std = torch.stack([cadv_mean, cadv_std])
                mean_std = mean_std.to(self.agent.device)
                torch.distributed.all_reduce(mean_std)
                mean_std = mean_std.to("cpu")
                mean_std /= world_size
                cadv_mean, cadv_std = mean_std[0], mean_std[1]
            c_advantage[:] = c_advantage - cadv_mean # only center, dont rescale (like in starter agents)

        return (return_, advantage, valid, c_return, c_advantage, c_return_var, c_var_value, cvar_return,
        cvar_return_offset, cvar_value_offset, self._ep_cost_ema, value, c_value)
