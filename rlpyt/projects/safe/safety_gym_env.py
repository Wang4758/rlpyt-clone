
# Requires installing OpenAI gym and safety gym.

from collections import namedtuple
from typing import List, Any
import numpy as np
from rlpyt.projects.safe.running_mean_std import RunningMeanStd

import safety_gym
import gym
from gym import Wrapper

from rlpyt.envs.gym import EnvInfoWrapper, GymEnvWrapper
from rlpyt.samplers.collections import TrajInfo

# To use: return a dict of keys and default values which sometimes appear in
# the wrapped env's env_info, so this env always presents those values (i.e.
# make keys and values keep the same structure and shape at all time steps.)
# Here, a dict of kwargs to be fed to `sometimes_info` should be passed as an
# env_kwarg into the `make` function, which should be used as the EnvCls.
def sometimes_info(*args, **kwargs):
    # e.g. Feed the env_id.
    # Return a dictionary (possibly nested) of keys: default_values
    # for this env.
    return dict(cost_exception=0, goal_met=False)


class SafetyGymEnvWrapper(Wrapper):

    def __init__(self, env, sometimes_info_kwargs, obs_prev_cost):
        super().__init__(env)
        self._sometimes_info = sometimes_info(**sometimes_info_kwargs)
        self._obs_prev_cost = obs_prev_cost
        self._prev_cost = 0.  # Concat this into the observation.
        obs = env.reset()
        # Some edited version of safexp envs defines observation space only
        # after reset, so expose it here (what base Wrapper does):
        self.observation_space = env.observation_space
        if isinstance(obs, dict):  # and "vision" in obs:
            self._prop_keys = [k for k in obs.keys() if k != "vision"]
            obs = self.observation(obs)
            prop_shape = obs["prop"].shape
            # if obs_prev_cost:
            #     assert len(prop_shape) == 1
            #     prop_shape = (prop_shape[0] + 1,)
            obs_space = dict(
                prop=gym.spaces.Box(-1e6, 1e6, prop_shape,
                    obs["prop"].dtype))
            if "vision" in obs:
                obs_space["vision"] = gym.spaces.Box(0, 1, obs["vision"].shape,
                    obs["vision"].dtype)
            # GymWrapper will in turn convert this to rlpyt.spaces.Composite.
            self.observation_space = gym.spaces.Dict(obs_space)
        elif obs_prev_cost:
            if isinstance(obs, dict):
                self.observation_space.spaces["prev_cost"] = gym.spaces.Box(
                    -1e6, 1e6, (1,), np.float32)
            else:
                obs_shape = obs.shape
                assert len(obs_shape) == 1
                obs_shape = (obs_shape[0] + 1,)
                self.observation_space = gym.spaces.Box(-1e6, 1e6, obs_shape,
                    obs.dtype)
        self._cum_cost = 0.

    def step(self, action):
        o, r, d, info = self.env.step(action)
        o = self.observation(o)  # Uses self._prev_cost
        self._prev_cost = info.get("cost", 0)
        self._cum_cost += self._prev_cost
        info["cum_cost"] = self._cum_cost
        # Try to make info dict same key structure at every step.
        info = infill_info(info, self._sometimes_info)
        for k, v in info.items():
            if isinstance(v, float):
                info[k] = np.dtype("float32").type(v)  # In case computing on.
        # Look inside safexp physics env for this logic on env horizon:
        info["timeout"] = info["is_time_out"] if "is_time_out" in info else ( d and (self.env.steps >= self.env.num_steps) )
        # info["timeout_next"] = not d and (
        #     self.env.steps == self.env.num_steps - 1)
        return o, r, d, info

    def reset(self):
        self._prev_cost = 0.
        self._cum_cost = 0.
        return self.observation(self.env.reset())

    def observation(self, obs):
        if isinstance(obs, dict):  # and "vision" in obs:
            # flatten everything else than vision.
            obs_ = dict(
                prop=np.concatenate([obs[k].reshape(-1)
                    for k in self._prop_keys])
            )
            if "vision" in obs:
                # [H,W,C] --> [C,H,W]
                obs_["vision"] = np.transpose(obs["vision"], (2, 0, 1))
            if self._obs_prev_cost:
                obs_["prop"] = np.append(obs_["prop"], self._prev_cost)
            obs = obs_
        elif self._obs_prev_cost:
            obs = np.append(obs, self._prev_cost)
        return obs


def infill_info(info, sometimes_info):
    for k, v in sometimes_info.items():
        if k not in info:
            info[k] = v
        elif isinstance(v, dict):
            infill_info(info[k], v)
    return info


def safety_gym_make(*args, sometimes_info_kwargs=None, obs_prev_cost=True,
        obs_version="default", **kwargs):
    assert obs_version in ["default", "vision", "vision_only", "no_lidar",
        "no_constraints"]
    if obs_version != "default":
        eid = kwargs["id"]  # Must provide as kwarg, not arg.
        names = dict(  # Map to my modification in safety-gym suite.
            vision="Vision",
            vision_only="Visonly",
            no_lidar="NoLidar",
            no_constraints="NoConstr",
        )
        name = names[obs_version]
        # e.g. Safexp-PointGoal1-v0 --> Safexp-PointGoal1Vision-v0
        kwargs["id"] = eid[:-3] + name + eid[-3:]
    
    should_normalize_reward: bool = kwargs.get("normalize_reward")
    del kwargs["normalize_reward"]

    env = gym.make(*args, **kwargs)
    if should_normalize_reward:
        inner_env = NormalizeReward(env)
    else:
        inner_env = env
    
    return GymEnvWrapper(SafetyGymEnvWrapper(
        inner_env,
        sometimes_info_kwargs=sometimes_info_kwargs or dict(),
        obs_prev_cost=obs_prev_cost),
    )

# ==============================================================
# 
# Everything ABOVE is previous code, everything BELOW 
# is my new code. SafetyGymTrajInfo was there before, but I
# extended it with all the commonroad related attributes.
# 
# ==============================================================

class SafetyGymTrajInfo(TrajInfo):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.IsGoalReached = 0
        self.IsCollision = 0
        self.IsOffroad = 0
        self.IsTimeout = 0
        self.Cost = 0
        self.R_G1 = 0
        self.R_G2 = 0
        self.R_G3 = 0
        self.CostSparse = 0
        self.NViolationsRG1 = 0
        self.NViolationsRG2 = 0
        self.NViolationsRG3 = 0

    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        self.Cost += getattr(env_info, "cost", 0)
        self.IsGoalReached += getattr(env_info, "is_goal_reached", 0) # for commonroad
        self.IsCollision += getattr(env_info, "is_collision", 0) # for commonroad
        self.IsOffroad += getattr(env_info, "is_off_road", 0) # for commonroad
        self.IsTimeout += getattr(env_info, "is_time_out", 0) # for commonroad
        self.R_G1 += getattr(env_info, "R_G1", 0) # for monitor
        self.R_G2 += getattr(env_info, "R_G2", 0) # for monitor
        self.R_G3 += getattr(env_info, "R_G3", 0) # for monitor
        self.CostSparse += int(getattr(env_info, "cost", 0) > 0)
        self.NViolationsRG1 += int(getattr(env_info, "R_G1", 0) < 0)
        self.NViolationsRG2 += int(getattr(env_info, "R_G2", 0) < 0)
        self.NViolationsRG3 += int(getattr(env_info, "R_G3", 0) < 0)

    def terminate(self, observation):
        del self.NonzeroRewards
        return super().terminate(observation)


class CommonroadEvalTrajInfo(SafetyGymTrajInfo):
    """Extend SafetyGymTrajInfo with BenchmarkId and Actions list"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.BenchmarkId: str = ""
        self.Actions: List[np.ndarray] = [] # gets filled in the eval collector 
        self.NTrafficRuleViolations = 0
        
        # to use these ego-states, go to commonroad_eval_collector and comment in the line:
        #       traj_infos[b].EgoStates.append(env.ego_state_list[-1])
        # (and comment in the definition of the attribute below)
        # self.EgoStates: List[Any] = [] # gets filled in the eval collector 
        
        # to use the robustness value traces, comment the definitions as well as the 
        # saving of values below in step() back in
        # self.R_G1s = []
        # self.R_G2s = []
        # self.R_G3s = []
    
    def step(self, observation, action, reward, done, agent_info, env_info):
        super().step(observation, action, reward, done, agent_info, env_info)
        # self.R_G1s.append(float(getattr(env_info, "R_G1")))
        # self.R_G2s.append(float(getattr(env_info, "R_G2")))
        # self.R_G3s.append(float(getattr(env_info, "R_G3")))
        if done: # num_traffic_rule_violation is accumulating! therefore we only need to safe it at the end!
            self.NTrafficRuleViolations += int(getattr(env_info, "num_traffic_rule_violation", 0)) #for monitor


class NormalizeObservation(gym.core.Wrapper):
    def __init__(
        self,
        env,
        epsilon=1e-8,
    ):
        super(NormalizeObservation, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        if self.is_vector_env:
            self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, dones, infos

    def reset(self):
        obs = self.env.reset()
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs

    def normalize(self, obs):
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon)


class NormalizeReward(gym.core.Wrapper):
    def __init__(
        self,
        env,
        gamma=0.99,
        epsilon=1e-8,
    ):
        super(NormalizeReward, self).__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.return_rms = RunningMeanStd(shape=())
        self.returns = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        obs, rews, dones, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        self.returns[dones] = 0.0
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, dones, infos

    def normalize(self, rews):
        self.return_rms.update(self.returns)
        return rews / np.sqrt(self.return_rms.var + self.epsilon)
