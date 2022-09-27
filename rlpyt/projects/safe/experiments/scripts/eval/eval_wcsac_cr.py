"""
    Created to evalutate WCSAC on commonroad / commonroad-monitor. 
    It is mostly a copy of the relevant parts from the training script in the starter-agents repo + the model loading part.

    NOTE: the evaluation results can be saved in two ways
        1) old way that was used in thesis: has less info, especially it is lacking the per-step robustness and ego state information
        2) new way for the animated plot in the paper: has commonroad.trajectory.State's for each scenario + per-step robustness values

    ==> to switch the argument parser has the '--new-eval-format' argument
"""

# Portions of the code are adapted from Safety Starter Agents and Spinning Up, released by OpenAI under the MIT license.
#!/usr/bin/env python

import warnings

import joblib

warnings.simplefilter(action="ignore", category=FutureWarning)

from functools import partial
import numpy as np
import tensorflow as tf
import gym
import time
from safe_rl.utils.logx import restore_tf_graph
from safe_rl.utils.mpi_tf import sync_all_params, MpiAdamOptimizer
from safe_rl.utils.mpi_tools import (
    mpi_fork,
    mpi_sum,
    proc_id,
    mpi_statistics_scalar,
    num_procs,
)
from safety_gym.envs.engine import Engine
from gym.envs.registration import register
from scipy.stats import norm
import argparse
import pathlib
import gzip
import json
from multiprocessing import Pool
import itertools

import commonroad_rl.gym_commonroad
from commonroad_rl.gym_commonroad.commonroad_env import (
    CommonroadPlayNoScenarioException,
)
import gym_monitor

def test_agent_pool(model_path: pathlib.Path, pool_size = 32, use_new_eval_format=False):
    with Pool(pool_size) as pool:
        args = [
            (model_path, pool_size, i, use_new_eval_format)
            for i in range(pool_size)
        ]
        results = pool.starmap(test_agent, args)
    return list(itertools.chain.from_iterable(results))

def test_agent(model_path: pathlib.Path, world_size = 1, rank = 0, use_new_eval_format = False):

    print('Loading environment...')
    # env = gym.make(
    #     "commonroad-v1", 
    #     test_env=True, 
    #     play=True, 
    #     logging_mode='INFO', 
    #     max_problems=-1, 
    #     mp_world_size=world_size, 
    #     mp_rank = rank
    # )
    env = gym.make(
        "cr-monitor-v0", 
        test_env=True, 
        play=True, 
        logging_mode='INFO', 
        max_problems=-1, 
        mp_world_size=world_size, 
        mp_rank = rank,
        active_rules = ["R_G1", "R_G2", "R_G3"],
        scenario_filter_file_path='/home/pillmayerc/mth/rlpyt/rlpyt/projects/safe/experiments/scripts/cr_util/results_R_G1R_G2R_G3_max_acc_ALLokayids.txt'
    )

    sess = tf.Session()

    tf_graph_io: dict = restore_tf_graph(sess, model_path.as_posix())
    mu, pi = tf_graph_io["mu"], tf_graph_io["pi"]
    x_ph = tf_graph_io["x"]

    def get_action(o, deterministic=False):
        act_op = mu if deterministic else pi
        return sess.run(act_op, feed_dict={x_ph: o.reshape(1, -1)})[0]

    max_ep_len = 1000
    render = False
    results = []
    for _ in range(len(env.all_problem_dict.keys())):
        try:
            o, r, d, ep_ret, ep_cost, ep_len, ep_goals = (env.reset(), 0, False, 0, 0, 0, 0)
            ep_cost_sparse = 0.0
            ep_r_g1 = 0.0
            ep_r_g2 = 0.0
            ep_r_g3 = 0.0
            ep_nv_g1 = 0.0
            ep_nv_g2 = 0.0
            ep_nv_g3 = 0.0
            actions = []
            ego_states = []
            r_g1s = []
            r_g2s = []
            r_g3s = []
        except CommonroadPlayNoScenarioException:
            print("Eval done")
            break
        while not (d or (ep_len == max_ep_len)):

            # for paper review: apply a bit of noise to simulate uncertainty
            # max_noise_level_percent = 25.0
            # observation_noise_frac = np.random.uniform(
            #     low = -max_noise_level_percent * 0.01,
            #     high = max_noise_level_percent * 0.01,
            #     size=o.shape
            # )
            # o += observation_noise_frac * o

            # Take deterministic actions at test time
            a = get_action(o, True)
            actions.append(a.tolist())
            o, r, d, info = env.step(a)
            if render and proc_id() == 0:
                env.render()
            ep_ret += r
            ep_cost += float(info.get("cost", 0))
            ep_len += 1
            ep_goals += int(info.get("is_goal_reached", False))

            ep_cost_sparse += int(info.get("cost", 0) > 0.)
            ep_r_g1 += float(info.get("R_G1", 0.0))
            ep_r_g2 += float(info.get("R_G2", 0.0))
            ep_r_g3 += float(info.get("R_G3", 0.0))
            ep_nv_g1 += int(info.get("R_G1", 0.0) < 0.0)
            ep_nv_g2 += int(info.get("R_G2", 0.0) < 0.0)
            ep_nv_g3 += int(info.get("R_G3", 0.0) < 0.0)
            ego_states.append(env.ego_state_list[-1])
            r_g1s.append(info.get("R_G1"))
            r_g2s.append(info.get("R_G2"))
            r_g3s.append(info.get("R_G3"))

            if d:
                # done!
                NTrafficRuleViolations = int(info.get("num_traffic_rule_violation"))
                result = {
                    "BenchmarkId": str(env.benchmark_id),
                    "Actions": actions,
                    "Length": ep_len,
                    "Return": ep_ret,
                    "Cost": ep_cost,
                    "IsGoalReached": info.get("is_goal_reached", 0),
                    "IsCollision": info.get("is_collision", 0),
                    "IsOffroad": info.get("is_off_road", 0),
                    "IsTimeout": info.get("is_time_out", 0),
                    "R_G1": ep_r_g1,
                    "R_G2": ep_r_g2,
                    "R_G3": ep_r_g3,
                    "CostSparse": ep_cost_sparse,
                    "NViolationsRG1": ep_nv_g1,
                    "NViolationsRG2": ep_nv_g2,
                    "NViolationsRG3": ep_nv_g3,
                    "NTrafficRuleViolations": NTrafficRuleViolations,
                }
                if use_new_eval_format:
                    result.update({
                        "EgoStates": ego_states,
                        "R_G1s": r_g1s,
                        "R_G2s": r_g2s,
                        "R_G3s": r_g3s,
                    })
                results.append(result)
                break

    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        type=pathlib.Path,
        help="Path to model dictionary",
        default="/home/pillmayerc/mth/data_to_keep/01_paper_data/wcsac_crm_0.5and0.9_2seeds",
    )
    parser.add_argument(
        "--new-eval-format",
        action="store_true"
    )
    args = parser.parse_args()
    path: pathlib.Path = args.d

    models_paths = path.glob('**/model_info.pkl')
    for model_path in models_paths:
        folder = model_path.parent
        if folder.name != 'simple_save14':
            # we only want the last model...
            continue
        run_id = int(folder.parent.name[-1])
        results = test_agent_pool(model_path = folder, use_new_eval_format=args.new_eval_format)
        # results = test_agent(model_path = folder)
        print(len(results))
        
        # jlresults_file = folder.parent.parent.joinpath(f'eval_results{run_id}_joblib.gz')
        # print("Saving results!")
        # joblib.dump(results, jlresults_file.as_posix(), compress=3)
        
        results_file = folder.parent.parent.joinpath(f'eval_results{run_id}_new.json.gz')
        with gzip.open(results_file, 'wb') as f:
            print("Saving results!")
            data: str = json.dumps(results)
            f.write(data.encode('utf-8'))