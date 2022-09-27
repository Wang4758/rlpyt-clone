"""
    Adapted trajectory collector class for evaluation that saves the actions (and ego state if uncommented)
    into a list. We need to do that here because we dont have access to that in the step function of
    the SafetyGymTrajInfo and CommonroadEvalTrajInfo classes
"""

from commonroad_rl.gym_commonroad.commonroad_env import CommonroadPlayNoScenarioException
import numpy as np

from rlpyt.samplers.collectors import BaseEvalCollector
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args

from rlpyt.projects.safe.safety_gym_env import CommonroadEvalTrajInfo

class CommonroadCpuEvalCollector(BaseEvalCollector):
    """Modified CpuEvalCollector for use with CpuSampler...
    """

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        observations = list()
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)

        try:

            for t in range(self.max_T):
                act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
                action = numpify_buffer(act_pyt)
                for b, env in enumerate(self.envs):
                    o, r, d, env_info = env.step(action[b])
                    traj_infos[b].step(observation[b], action[b], r, d,
                        agent_info[b], env_info)
                    traj_infos[b].Actions.append(action[b].tolist()) # SAVE ACTION (convert to list from ndarray bc of json later)
                    # comment this back in to save ego states into the trajectory info object
                    # traj_infos[b].EgoStates.append(env.ego_state_list[-1])
                    if getattr(env_info, "traj_done", d):
                        traj_infos[b].BenchmarkId = getattr(env, 'benchmark_id') # SAVE BENCHMARK ID BEFORE RESET
                        if traj_infos[b].IsGoalReached:
                            print("Goal reached!")
                        self.traj_infos_queue.put(traj_infos[b].terminate(o))
                        traj_infos[b] = self.TrajInfoCls()
                        o = env.reset()
                    if d:
                        action[b] = 0  # Next prev_action.
                        r = 0
                        self.agent.reset_one(idx=b)
                    observation[b] = o
                    reward[b] = r
                if self.sync.stop_eval.value:
                    break
            
        
        except CommonroadPlayNoScenarioException:
            # done, used all scenarios --> stop collecting
            print("All scenarios done")
            pass

        finally:
            self.traj_infos_queue.put(None)  # End sentinel.
