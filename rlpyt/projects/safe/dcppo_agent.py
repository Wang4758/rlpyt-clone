#
# Modified version to support Worst-Case Actor
#

import torch

from rlpyt.distributions.gaussian import DistInfoStd
from rlpyt.agents.pg.mujoco import MujocoFfAgent, MujocoLstmAgent
from rlpyt.utils.buffer import buffer_to
from rlpyt.utils.collections import namedarraytuple

from rlpyt.projects.safe.dcppo_model import DCppoModel
from rlpyt.projects.safe.dcppo_model_v2 import DCppoModelV2

ValueInfo = namedarraytuple("ValueInfo", ["value", "c_value"])


class DCppoAgent(MujocoFfAgent):

    def __init__(
            self,
            ModelCls=DCppoModel,
            model_kwargs=None,
            initial_model_state_dict=None,
            ):
        if isinstance(ModelCls, str):
            # resolve model class from string
            if ModelCls == "DCppoModelV2":
                ModelCls = DCppoModelV2
            elif ModelCls == "DCppoModel":
                ModelCls = DCppoModel
            else:
                raise ValueError(f"Unrecognized model class: {ModelCls}")
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs,
            initial_model_state_dict=initial_model_state_dict)
        print(f"====== Using ModelCls {ModelCls}")
        self._ddp = False

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _mu, _log_std, value = self.model(*agent_inputs)
        return buffer_to(value, device="cpu")  # TODO: apply this to rlpyt.

    def update_obs_rms(self, observation):
        observation = buffer_to(observation, device=self.device)
        if self._ddp:
            self.model.module.update_obs_rms(observation)
        else:
            self.model.update_obs_rms(observation)

    def update_rew_rms(self, reward):
        reward = buffer_to(reward, device=self.device)
        if self._ddp:
            self.model.module.update_rew_rms(reward)
        else:
            self.model.update_rew_rms(reward)

    def data_parallel(self, *args, **kwargs):
        device_id = super().data_parallel(*args, **kwargs)
        self._ddp = True
        return device_id


class DCppoLstmAgent(MujocoLstmAgent):
    """But model_kwargs determines whether lstm is used."""

    def __init__(
            self,
            ModelCls=DCppoModel,  # I think can just swap in CppoConv
            model_kwargs=None,
            initial_model_state_dict=None,
            ):
        if isinstance(ModelCls, str):
            # resolve model class from string
            if ModelCls == "DCppoModelV2":
                ModelCls = DCppoModelV2
            elif ModelCls == "DCppoModel":
                ModelCls = DCppoModel
            else:
                raise ValueError(f"Unrecognized model class: {ModelCls}")
        super().__init__(ModelCls=ModelCls, model_kwargs=model_kwargs,
            initial_model_state_dict=initial_model_state_dict)
        self._ddp = False

    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        # self.distribution.set_std(0.)
        self.beta_r_model = self.ModelCls(**self.model_kwargs,
            **self.env_model_kwargs)
        self.beta_c_model = self.ModelCls(**self.model_kwargs,
            **self.env_model_kwargs)

    def to_device(self, cuda_idx=None):
        super().to_device(cuda_idx=cuda_idx)
        if cuda_idx is not None:
            self.beta_r_model.to(self.device)
            self.beta_c_model.to(self.device)

    @torch.no_grad()
    def value(self, observation, prev_action, prev_reward):
        agent_inputs = buffer_to((observation, prev_action, prev_reward),
            device=self.device)
        _mu, _log_std, value, _rnn_state = self.model(*agent_inputs,
            self.prev_rnn_state)
        return buffer_to(value, device="cpu")

    def update_obs_rms(self, observation):
        observation = buffer_to(observation, device=self.device)
        if self._ddp:
            self.model.module.update_obs_rms(observation)
        else:
            self.model.update_obs_rms(observation)

    def update_rew_rms(self, reward):
        reward = buffer_to(reward, device=self.device)
        if self._ddp:
            self.model.module.update_rew_rms(reward)
        else:
            self.model.update_rew_rms(reward)
    
    def data_parallel(self, *args, **kwargs):
        device_id = super().data_parallel(*args, **kwargs)
        self._ddp = True
        return device_id

    def beta_dist_infos(self, observation, prev_action, prev_reward,
            init_rnn_state):
        model_inputs = buffer_to((observation, prev_action, prev_reward,
            init_rnn_state), device=self.device)
        r_mu, r_log_std, _, _ = self.beta_r_model(*model_inputs)
        c_mu, c_log_std, _, _ = self.beta_c_model(*model_inputs)
        return buffer_to((DistInfoStd(mean=r_mu, log_std=r_log_std),
            DistInfoStd(mean=c_mu, log_std=c_log_std)), device="cpu")
