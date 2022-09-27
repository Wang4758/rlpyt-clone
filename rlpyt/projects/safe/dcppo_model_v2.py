#
# Modified version to support Worst-Case Actor
#

import numpy as np
import torch
import torch.nn.functional as F

from rlpyt.models.mlp import MlpModel
from rlpyt.models.running_mean_std import RunningMeanStdModel
from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.utils.collections import namedarraytuple

ValueInfo = namedarraytuple("ValueInfo", ["value", "c_value", "c_var_value"])
RnnState = namedarraytuple("RnnState", ["h", "c"])


class DCppoModelV2(torch.nn.Module):

    def __init__(
            self,
            observation_shape,
            action_size,
            hidden_sizes=None,
            lstm_size=None,
            lstm_skip=True,
            constraint=True,
            hidden_nonlinearity="tanh",  # or "relu"
            mu_nonlinearity="tanh",
            init_log_std=0.,
            normalize_observation=True,
            var_clip=1e-6,
            normalize_reward=False,
            a_long_shift=0.0,
            mu_input_mode="concat" # "add"
            ):
        super().__init__()

        # dont support old settings atm
        assert lstm_size == None, "lstm_size should be None in v2 model"
        assert constraint == True, "constraint should be True in v2 model"

        if hidden_nonlinearity == "tanh":  # So these can be strings in config file.
            hidden_nonlinearity = torch.nn.Tanh
        elif hidden_nonlinearity == "relu":
            hidden_nonlinearity = torch.nn.ReLU
        else:
            raise ValueError(f"Unrecognized hidden_nonlinearity string: {hidden_nonlinearity}")
        if mu_nonlinearity == "tanh":  # So these can be strings in config file.
            mu_nonlinearity = torch.nn.Tanh
        elif mu_nonlinearity == "relu":
            mu_nonlinearity = torch.nn.ReLU
        else:
            raise ValueError(f"Unrecognized mu_nonlinearity string: {mu_nonlinearity}")
        
        self._obs_ndim = len(observation_shape)
        input_size = int(np.prod(observation_shape))
        
        # reward model
        self.reward_body = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes or [256, 256],
            nonlinearity=hidden_nonlinearity,
        )

        # constraint model
        self.constraint_body = MlpModel(
            input_size=input_size,
            hidden_sizes=hidden_sizes or [256, 256],
            nonlinearity=hidden_nonlinearity,
        )

        last_size = self.reward_body.output_size
        
        # Value heads
        self.return_value = torch.nn.Linear(last_size, 1)
        self.constraint_value = torch.nn.Linear(last_size, 1)
        self.constraint_var = torch.nn.Linear(last_size, 1)

        # Policy head
        if mu_input_mode == "add":
            mu_linear = torch.nn.Linear(last_size, action_size)
            if mu_nonlinearity is not None:
                self.mu = torch.nn.Sequential(mu_linear, mu_nonlinearity())
            else:
                self.mu = mu_linear
        elif mu_input_mode == "concat":
            self.mu = MlpModel(2 * last_size, [last_size, action_size], nonlinearity=mu_nonlinearity)
            assert self.mu.output_size == action_size
        else:
            raise ValueError(f"Unrecongnized mu_input_mode '{mu_input_mode}' (Valid: 'add', 'concat')")
        
        self.log_std = torch.nn.Parameter(init_log_std * torch.ones(action_size))
        self.mu_input_mode = mu_input_mode
        self.a_long_shift: float = a_long_shift

        if normalize_observation:
            self.obs_rms = RunningMeanStdModel(observation_shape)
        if normalize_reward:
            reward_shape = (1,)
            self.rew_rms = RunningMeanStdModel(reward_shape)
        if normalize_observation or normalize_reward:
            self.var_clip = var_clip
        self.normalize_observation = normalize_observation
        self.normalize_reward = normalize_reward

    def forward(self, observation, prev_action, prev_reward, init_rnn_state=None):
        lead_dim, T, B, _ = infer_leading_dims(observation, self._obs_ndim)
        if self.normalize_observation:
            obs_var = self.obs_rms.var
            if self.var_clip is not None:
                obs_var = torch.clamp(obs_var, min=self.var_clip)
            
            # uniform_noise_strength = 2.0 * torch.rand_like(observation) - 1
            # max_noise_level_percent = 25.0
            # observation_noise_percent = uniform_noise_strength * max_noise_level_percent
            # observation += (0.01 * observation_noise_percent) * observation
            
            observation = torch.clamp((observation - self.obs_rms.mean) /
                obs_var.sqrt(), -10, 10)
        
        obs_reshaped = observation.view(T * B, -1)

        # reward body
        r_fc_x = self.reward_body(obs_reshaped)
        v_r = self.return_value(r_fc_x).squeeze(-1)
        
        # constraint body
        c_fc_x = self.constraint_body(obs_reshaped)
        v_c = self.constraint_value(c_fc_x).squeeze(-1)
        # variance should be positive (softplus) and also small in the beginning (*.1)
        v_c_var = F.softplus(self.constraint_var(c_fc_x).squeeze(-1)) * 0.1

        # policy
        if self.mu_input_mode == "add":
            policy_input = r_fc_x + c_fc_x
        elif self.mu_input_mode == "concat":
            # for shapes (n,128) and (n,128) --> we want (n, 256)
            policy_input = torch.concat([r_fc_x, c_fc_x], dim=1)
        mu = self.mu(policy_input)
        # this is specifically to prevent many violations of rule 2 in the beginning, use with care :)
        offset = torch.tensor([self.a_long_shift, 0]).to(mu.device)
        mu = mu + offset.repeat(len(mu), 1)
        log_std = self.log_std.repeat(T * B, 1)

        mu, log_std, v = restore_leading_dims((mu, log_std, v_r), lead_dim, T, B)
        c = restore_leading_dims(v_c, lead_dim, T, B)
        cv = restore_leading_dims(v_c_var, lead_dim, T, B)
        
        value = ValueInfo(value=v, c_value=c, c_var_value=cv)

        outputs = (mu, log_std, value)

        return outputs

    def update_obs_rms(self, observation):
        if not self.normalize_observation:
            return
        self.obs_rms.update(observation)

    def update_rew_rms(self, reward):
        if not self.normalize_reward:
            return
        self.rew_rms.update(reward)